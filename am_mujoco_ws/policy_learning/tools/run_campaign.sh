#!/bin/bash
# ============================================================================
# run_campaign.sh —— 按"条件表"批量跑评估，一个条件×一个种子一次进程
# ----------------------------------------------------------------------------
# 设计目标（对应 §B 的环境固化 / §D 的运行规范）：
#   1) 每次运行只跑一个 EVAL_SEED（种子是进程级的，混在一起无法归因）；
#   2) 输出目录、ACADOS 构建目录按 条件×种子 隔离，避免并发时互相踩；
#   3) 固定环境变量（MUJOCO_GL / HF_ENDPOINT / setup_ee_mpc.sh）；
#   4) 固定 KEEP_FAILED_EPISODES=1：被重启/崩溃/丢弃的尝试归档而不是删除，
#      否则"这集试了 3 次"这件事在磁盘上完全看不出来；
#   5) EVAL_TAG 写进 condition.json，汇总脚本按 tag 分组，不靠猜目录名。
#
# 条件表格式（TSV，制表符分隔，第一行表头，'#' 开头为注释）：
#   tag        task         mode      value guided_steps rollouts seeds  ckpt extra
#   base       uam_cabinet  baseline  0     0            10       0,1,2  -    -
#   scale1.5   uam_cabinet  scale     1.5   1            10       0,1,2  -    -
#   guid1.5    uam_cabinet  guidance  1.5   15           10       0,1,2  -    --log_diffusion
#   base_d     uam_cabinet  baseline  0     0            10       0,1,2  -    --disturb
#   mode: baseline(不加引导) / scale(模式A: --scale) / guidance(模式B: --guidance)
#   ckpt: '-' 表示自动推导为 $CKPT_ROOT/umi_<task去掉uam_前缀>/checkpoints/latest.ckpt
#   extra: 追加的原始参数（如 --disturb、--log_diffusion），'-' 表示无
#
# 用法：
#   bash run_campaign.sh campaign_example.tsv                 # 正式跑
#   bash run_campaign.sh campaign_example.tsv --dry-run       # 只打印将执行的命令
#   bash run_campaign.sh campaign_example.tsv --only scale1.5 # 只跑指定 tag（可多次）
#   bash run_campaign.sh campaign_example.tsv --force         # 已完成的条件也重跑
#   PARALLEL=2 bash run_campaign.sh ...                       # 并发 2 个（单卡谨慎！）
# ============================================================================
set -u

SPEC=${1:?用法: bash run_campaign.sh <conditions.tsv> [--dry-run] [--force] [--only TAG]...}
shift
# 条件表转绝对路径：脚本后面会 cd 到 policy_learning，相对路径会失效
[ -f "$SPEC" ] || { echo "❌ 条件表不存在: $SPEC"; exit 2; }
SPEC=$(cd "$(dirname "$SPEC")" && pwd)/$(basename "$SPEC")

# ---- 路径与环境（云端默认值，可用环境变量覆盖） ----------------------------
WS=${WS:-/root/autodl-tmp/am_mujoco_ws}
PY=${PY:-/root/miniconda3/bin/python}
RESULTS_ROOT=${RESULTS_ROOT:-/root/autodl-tmp/results/eval}
CKPT_ROOT=${CKPT_ROOT:-/root/autodl-tmp/checkpoints}
ACADOS_BUILD_ROOT=${ACADOS_BUILD_ROOT:-/root/autodl-tmp/acados_build}
LOG_ROOT=${LOG_ROOT:-/root/autodl-tmp/logs/campaign}
PARALLEL=${PARALLEL:-1}
EST_WALL_PER_EPISODE=${EST_WALL_PER_EPISODE:-150}   # 秒；失败集更贵，仅用于预算提示

DRY_RUN=0; FORCE=0; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --force)   FORCE=1 ;;
    --only)    ONLY+=("$2"); shift ;;
    *) echo "未知参数: $1"; exit 2 ;;
  esac
  shift
done
selected() {  # 无 --only 则全选
  [ ${#ONLY[@]} -eq 0 ] && return 0
  local t; for t in "${ONLY[@]}"; do [ "$t" = "$1" ] && return 0; done
  return 1
}

export MUJOCO_GL=egl
export PATH="$(dirname "$PY"):$PATH"
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export KEEP_FAILED_EPISODES=1
# setup_ee_mpc.sh 内部写的是 `LD_LIBRARY_PATH="...:$LD_LIBRARY_PATH"`，
# 非交互 shell 里该变量常常未定义，配合本脚本的 `set -u` 会直接 `unbound variable` 退出
# （2026-09-19 冒烟实测踩到）。这里先兜底再临时关掉 -u 取 source。
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ "$DRY_RUN" = "0" ]; then
  set +u
  # shellcheck disable=SC1090
  source "$WS/am_trajectory_controller/setup_ee_mpc.sh"
  set -u
fi

cd "$WS/policy_learning" || { echo "❌ 找不到 $WS/policy_learning"; exit 1; }

# ---- 预算估算 --------------------------------------------------------------
total_runs=0; total_eps=0
while IFS=$'\t' read -r tag task mode value gsteps rollouts seeds ckpt extra; do
  [[ "$tag" =~ ^#.*$ || -z "$tag" || "$tag" = "tag" ]] && continue
  selected "$tag" || continue
  n_seeds=$(awk -F, '{print NF}' <<<"$seeds")
  total_runs=$((total_runs + n_seeds)); total_eps=$((total_eps + n_seeds * rollouts))
done < <(grep -v '^[[:space:]]*$' "$SPEC")
echo "============================================================"
echo "条件表: $SPEC"
echo "计划: $total_runs 次运行 / $total_eps 集（按每集 ${EST_WALL_PER_EPISODE}s 估 → 约 $((total_eps * EST_WALL_PER_EPISODE / 60)) 分钟，串行）"
echo "并发度: $PARALLEL   结果根目录: $RESULTS_ROOT"
echo "============================================================"
[ "$total_runs" = "0" ] && { echo "没有匹配的条件（检查 --only 拼写）"; exit 1; }

# ---- 主循环 ----------------------------------------------------------------
failures=0
pids=()
run_one() {  # tag task mode value gsteps rollouts seed ckpt extra
  local tag=$1 task=$2 mode=$3 value=$4 gsteps=$5 rollouts=$6 seed=$7 ckpt=$8 extra=$9
  local out_dir="$RESULTS_ROOT/$tag/seed$seed"
  local build_dir="$ACADOS_BUILD_ROOT/${tag}_seed${seed}"
  local log="$LOG_ROOT/${tag}_seed${seed}.log"

  if [ -f "$out_dir/experiment_summary.json" ] && [ "$FORCE" = "0" ]; then
    echo "⏭️  跳过（已完成）: $tag seed=$seed"; return 0
  fi

  # 引导参数：模式 A(scale) 与模式 B(guidance) 互斥；baseline 两者都不加
  local guide_args=()
  case "$mode" in
    baseline) ;;
    scale)    guide_args=(--scale "$value" --guided_steps "$gsteps") ;;
    guidance) guide_args=(--guidance "$value" --guided_steps "$gsteps") ;;
    *) echo "❌ 未知 mode: $mode（应为 baseline/scale/guidance）"; return 2 ;;
  esac
  local extra_args=()
  [ "$extra" != "-" ] && [ -n "$extra" ] && read -r -a extra_args <<<"$extra"

  if [ "$ckpt" = "-" ] || [ -z "$ckpt" ]; then
    ckpt="$CKPT_ROOT/umi_${task#uam_}/checkpoints/latest.ckpt"
  fi
  if [ ! -f "$ckpt" ]; then
    if [ "$DRY_RUN" = "1" ]; then
      echo "   ⚠️  dry-run: checkpoint 尚不存在（正式跑前请确认）: $ckpt"
    else
      echo "❌ checkpoint 不存在: $ckpt"; return 3
    fi
  fi

  local cmd=(xvfb-run -a "$PY" imitate_episodes.py
    --task_name "$task" --num_rollouts "$rollouts"
    --load_ckpt_file_path "$ckpt"
    --output_dir "$out_dir"
    --acados_build_dir "$build_dir"
    "${guide_args[@]}" "${extra_args[@]}")

  if [ "$DRY_RUN" = "1" ]; then
    echo "▶️  EVAL_TAG=$tag EVAL_SEED=$seed ${cmd[*]}"
    echo "     → $out_dir"
    return 0
  fi

  mkdir -p "$out_dir" "$build_dir" "$(dirname "$log")"
  echo "▶️  [$tag seed=$seed] → $out_dir（日志 $log）"
  EVAL_TAG="$tag" EVAL_SEED="$seed" "${cmd[@]}" >"$log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "❌ [$tag seed=$seed] 退出码 $rc（见 $log）"; return $rc
  fi
  echo "✅ [$tag seed=$seed] 完成 | $(grep 'Success rate' "$log" | tail -1)"
  return 0
}

while IFS=$'\t' read -r tag task mode value gsteps rollouts seeds ckpt extra; do
  [[ "$tag" =~ ^#.*$ || -z "$tag" || "$tag" = "tag" ]] && continue
  selected "$tag" || continue
  extra=${extra%$'\r'}; ckpt=${ckpt%$'\r'}; seeds=${seeds%$'\r'}   # 容忍 CRLF 条件表
  IFS=',' read -r -a seed_list <<<"$seeds"
  for seed in "${seed_list[@]}"; do
    if [ "$PARALLEL" -gt 1 ]; then
      while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n; done
      run_one "$tag" "$task" "$mode" "$value" "$gsteps" "$rollouts" "$seed" "$ckpt" "$extra" &
      pids+=($!)
    else
      run_one "$tag" "$task" "$mode" "$value" "$gsteps" "$rollouts" "$seed" "$ckpt" "$extra" || failures=$((failures+1))
    fi
  done
done < <(grep -v '^[[:space:]]*$' "$SPEC")

for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || failures=$((failures+1)); }; done

echo "============================================================"
echo "完成。失败运行数: $failures"
echo "汇总: $PY tools/aggregate_results.py $RESULTS_ROOT --out-dir $RESULTS_ROOT/_aggregate"
echo "============================================================"
exit $(( failures > 0 ? 1 : 0 ))

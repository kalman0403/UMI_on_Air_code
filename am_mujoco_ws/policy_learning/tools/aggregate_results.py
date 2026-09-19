#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
aggregate_results.py —— 把一堆评估输出目录汇总成"可读、可算、可画"的表。

为什么需要它（旧数据为什么难读）：
  * 每个 episode 一个 metrics.json，条件散落在目录名/命令历史里，没有 condition.json；
  * experiment_summary.json 只覆盖"本次进程"，跨种子/跨条件的比较要手工拼；
  * 旧数据里有若干"恒为 0"的字段（avg_ref_vs_mpc_* / mpc_data 等），
    汇总时必须显式标注"该字段无效"，否则很容易把 0 当成真实测量值；
  * 被重试的集目录原先被删除，"成功率的分母"和"实际尝试次数"对不上。

本脚本只依赖标准库（math/csv/json），因此可以直接在本地（不装 torch 的机器）跑下载回来的结果。

用法：
    python3 tools/aggregate_results.py <结果根目录> [--out-dir DIR] [--label 名字] [--quiet]
    # 例：python3 tools/aggregate_results.py /root/autodl-tmp/results/eval --label P1_cabinet
    #     python3 tools/aggregate_results.py ./downloaded_results --out-dir ./agg

识别方式（两种布局都支持）：
  新布局  <root>/<tag>/seed<k>/{condition.json, experiment_summary.json, episode_00X/}
  旧布局  <root>/<task>/<timestamp>/{experiment_summary.json, episode_00X/}   ← 无 condition.json

输出：
  <out-dir>/episodes.csv            每集一行（含条件列 + 关键指标 + 数据质量标记）
  <out-dir>/conditions.csv          每条件一行（n / 成功率 / Wilson 95% CI / 均值±std / 尝试与重启）
  <out-dir>/quality_report.txt      数据可信度清单（缺元数据、缺新日志字段、被重试次数…）
  控制台：条件级汇总表
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import sys

EPISODE_RE = re.compile(r'^episode_(\d+)$')
RETRY_RE = re.compile(r'^episode_(\d+)__retry-(.+)-(\d+)$')

# 旧数据里"结构上恒为 0"的字段：汇总时必须显式标注无效，避免被当成测量值
INVALID_IF_NO_DIFFUSION_LOG = (
    'avg_ref_vs_mpc_pos_rmse', 'avg_mpc_vs_actual_pos_rmse',
    'avg_ref_vs_mpc_orient_dist', 'avg_mpc_vs_actual_orient_dist',
)
# 只有打了 logging 补丁的数据才会有的字段
NEW_LOGGING_FIELDS = (
    'outcome', 'failure_reasons', 'episode_len',
    'index_attempts', 'index_restarts', 'first_success_step',
    'reward_series', 'inference_mpc_costs', 'num_timesteps',
)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（小样本成功率必报，避免 3/3=100% 的错觉）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (center - half) / denom), min(1.0, (center + half) / denom))


# 失败模式分类：§C 要求区分"被重启上限截断"与"跑满整集未达标"，
# 否则 λ 变大后成功率下降的原因无法归因（是策略变差，还是被重启机制截断）。
FULL_EPISODE_TOL = 0.9          # 步数 ≥ 90% 集长即视为"跑满"


def classify_outcome(row, episode_len=3000):
    """返回失败/成功模式标签（见下方各分支注释）"""
    if row.get('success'):
        return 'success'
    if row.get('crashed'):
        return 'crash'
    steps = row.get('num_timesteps')
    if steps is None:
        return 'legacy_unknown'
    if steps == 0:
        return 'aborted_no_steps'          # 每次尝试都在 step 0 被重启/崩溃掉（如 F_scale_g0）
    if steps >= FULL_EPISODE_TOL * episode_len:
        return 'timeout_full_episode'      # 跑满集长仍未达标
    return 'early_stop_other'              # 提前结束（掉罐等）


def load_json(path):
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def mean_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def std_or_none(xs):
    xs = [x for x in xs if x is not None]
    return statistics.pstdev(xs) if len(xs) > 1 else 0.0 if xs else None


def find_run_dirs(root: str) -> list[str]:
    """含 condition.json 或 per-episode metrics.json 的目录视为一次运行。"""
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ('_aggregate', '__pycache__')]
        has_cond = 'condition.json' in filenames
        has_summary = 'experiment_summary.json' in filenames
        has_eps = any(EPISODE_RE.match(d) for d in dirnames)
        if has_cond or has_summary or has_eps:
            runs.append(dirpath)
            dirnames[:] = []  # 不再往里走：episode 目录内部没有条件元数据
    return sorted(runs)


def collect_run(run_dir: str, root: str) -> tuple[list[dict], dict]:
    """返回 (每集行, 该运行的质量标记)"""
    cond = load_json(os.path.join(run_dir, 'condition.json')) or {}
    summary = load_json(os.path.join(run_dir, 'experiment_summary.json')) or {}
    rel = os.path.relpath(run_dir, root)
    group = cond.get('tag') or rel
    seed = cond.get('seed')
    if seed is None:
        m = re.search(r'seed(\d+)', rel)
        seed = int(m.group(1)) if m else None

    base = {
        'group': group, 'run_dir': rel, 'seed': seed,
        'episode_len': cond.get('episode_len'),
        'task_name': cond.get('task_name'), 'mode': cond.get('mode'),
        'scale': cond.get('scale'), 'guidance': cond.get('guidance'),
        'guided_steps': cond.get('guided_steps'), 'disturb': cond.get('disturb'),
        'log_diffusion': cond.get('log_diffusion'),
        'code_commit': cond.get('code_commit'),
        'ckpt_sha256_head1mb': cond.get('ckpt_sha256_head1mb'),
        'metadata_present': bool(cond),
    }

    rows = []
    for d in sorted(os.listdir(run_dir)):
        m = EPISODE_RE.match(d)
        if not m:
            continue
        metrics = load_json(os.path.join(run_dir, d, 'metrics.json'))
        if metrics is None:
            continue
        row = dict(base)
        row.update({
            'episode_id': int(m.group(1)),
            'success': bool(metrics.get('success')),
            'crashed': bool(metrics.get('crashed')),
            'episode_return': metrics.get('episode_return'),
            'highest_reward': metrics.get('highest_reward'),
            'num_timesteps': metrics.get('num_timesteps'),
            'episode_duration': metrics.get('episode_duration'),
            'avg_position_rmse': metrics.get('avg_position_rmse'),
            'avg_orientation_distance': metrics.get('avg_orientation_distance'),
            'avg_main_mpc_tracking_cost': metrics.get('avg_main_mpc_tracking_cost'),
            'index_attempts': metrics.get('index_attempts'),
            'index_restarts': metrics.get('index_restarts'),
            'first_success_step': metrics.get('first_success_step'),
            'first_inference_mpc_cost': metrics.get('first_inference_mpc_cost'),
            'avg_inference_mpc_cost': metrics.get('avg_inference_mpc_cost'),
            'max_inference_mpc_cost': metrics.get('max_inference_mpc_cost'),
            'has_new_logging': all(k in metrics for k in NEW_LOGGING_FIELDS),
            # 旧数据的"零值陷阱"标记：没开 log_diffusion 时这几个字段结构上就是 0
            'zero_trap_fields': (not cond.get('log_diffusion', False)),
            'failure_reasons': metrics.get('failure_reasons'),
        })
        row['outcome'] = classify_outcome(row, cond.get('episode_len') or 3000)
        rows.append(row)

    archived = [d for d in os.listdir(run_dir) if RETRY_RE.match(d)]
    retry_reasons = {}
    for d in archived:
        retry_reasons[RETRY_RE.match(d).group(2)] = retry_reasons.get(RETRY_RE.match(d).group(2), 0) + 1
    quality = {
        'group': group, 'run_dir': rel, 'metadata_present': bool(cond),
        'episodes_on_disk': len(rows),
        'episodes_in_summary': summary.get('experiment_summary', {}).get('total_episodes'),
        'archived_attempts': len(archived),
        'archived_by_reason': retry_reasons,
        'has_new_logging': all(r['has_new_logging'] for r in rows) if rows else False,
        'summary_total_restarts': summary.get('experiment_summary', {}).get('total_restarts'),
        'summary_total_attempts': summary.get('experiment_summary', {}).get('total_attempts'),
        'seed_recorded': seed is not None,
    }
    return rows, quality


def group_conditions(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(r['group'], []).append(r)

    out = []
    for g, rs in sorted(groups.items()):
        n = len(rs)
        k = sum(1 for r in rs if r['success'])
        lo, hi = wilson_ci(k, n)
        returns = [r['episode_return'] for r in rs if r['episode_return'] is not None]
        durs = [r['episode_duration'] for r in rs if r['episode_duration'] is not None]
        atts = [r['index_attempts'] for r in rs if r['index_attempts'] is not None]
        rsts = [r['index_restarts'] for r in rs if r['index_restarts'] is not None]
        firsts = [r['first_success_step'] for r in rs if r.get('first_success_step') is not None]
        first_cost = [r['first_inference_mpc_cost'] for r in rs if r.get('first_inference_mpc_cost') is not None]
        max_cost = [r['max_inference_mpc_cost'] for r in rs if r.get('max_inference_mpc_cost') is not None]
        first = rs[0]
        mode_counts = {}
        for r in rs:
            mode_counts[r['outcome']] = mode_counts.get(r['outcome'], 0) + 1
        out.append({
            'outcome_counts': '|'.join(f'{k}:{v}' for k, v in sorted(mode_counts.items())),
            'n_timeout_full': mode_counts.get('timeout_full_episode', 0),
            'n_aborted_no_steps': mode_counts.get('aborted_no_steps', 0),
            'n_crash': mode_counts.get('crash', 0),
            'group': g, 'n_episodes': n, 'n_success': k,
            'success_rate': k / n if n else 0.0,
            'wilson_lo': lo, 'wilson_hi': hi,
            'return_mean': mean_or_none(returns), 'return_std': std_or_none(returns),
            'duration_mean_s': mean_or_none(durs),
            'attempts_mean': mean_or_none(atts), 'restarts_mean': mean_or_none(rsts),
            'total_attempts': int(sum(atts)) if atts else None,
            'total_restarts': int(sum(rsts)) if rsts else None,
            'first_success_step_mean': mean_or_none(firsts),
            'first_mpc_cost_mean': mean_or_none(first_cost),
            'max_mpc_cost_mean': mean_or_none(max_cost),
            'task_name': first.get('task_name'), 'mode': first.get('mode'),
            'scale': first.get('scale'), 'guidance': first.get('guidance'),
            'guided_steps': first.get('guided_steps'), 'disturb': first.get('disturb'),
            'log_diffusion': first.get('log_diffusion'),
            'metadata_present': all(r['metadata_present'] for r in rs),
            'has_new_logging': all(r['has_new_logging'] for r in rs),
            'n_seeds': len({r['seed'] for r in rs if r['seed'] is not None}),
        })
    return out


EPISODE_FIELDS = [
    'group', 'seed', 'episode_id', 'success', 'crashed', 'episode_return', 'highest_reward',
    'num_timesteps', 'episode_duration', 'avg_position_rmse', 'avg_orientation_distance',
    'avg_main_mpc_tracking_cost',
    'outcome', 'failure_reasons', 'episode_len',
    'index_attempts', 'index_restarts', 'first_success_step',
    'first_inference_mpc_cost', 'avg_inference_mpc_cost', 'max_inference_mpc_cost',
    'task_name', 'mode', 'scale', 'guidance', 'guided_steps', 'disturb', 'log_diffusion',
    'code_commit', 'ckpt_sha256_head1mb', 'metadata_present', 'has_new_logging', 'run_dir',
]
CONDITION_FIELDS = [
    'group', 'n_episodes', 'n_success', 'success_rate', 'wilson_lo', 'wilson_hi',
    'outcome_counts', 'n_timeout_full', 'n_aborted_no_steps', 'n_crash',
    'return_mean', 'return_std', 'duration_mean_s', 'attempts_mean', 'restarts_mean',
    'total_attempts', 'total_restarts', 'first_success_step_mean',
    'first_mpc_cost_mean', 'max_mpc_cost_mean', 'n_seeds',
    'task_name', 'mode', 'scale', 'guidance', 'guided_steps', 'disturb', 'log_diffusion',
    'metadata_present', 'has_new_logging',
]


def write_csv(path, fields, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fields})


def fmt(x, nd=3):
    if x is None:
        return '   -  '
    if isinstance(x, float):
        return f'{x:.{nd}f}'
    return str(x)


def main():
    ap = argparse.ArgumentParser(description='汇总评估结果（纯标准库）')
    ap.add_argument('root', help='结果根目录（递归查找含 condition.json / experiment_summary.json 的目录）')
    ap.add_argument('--out-dir', default=None, help='输出目录（默认 <root>/_aggregate_<label>）')
    ap.add_argument('--label', default=None, help='这次汇总的标签，写进输出目录名')
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f'❌ 目录不存在: {args.root}')
        return 2

    run_dirs = find_run_dirs(args.root)
    if not run_dirs:
        print(f'❌ 在 {args.root} 下没找到任何运行目录（需要 condition.json / experiment_summary.json / episode_*/metrics.json）')
        return 2

    all_rows, qualities = [], []
    for rd in run_dirs:
        rows, q = collect_run(rd, args.root)
        all_rows.extend(rows)
        qualities.append(q)

    if not all_rows:
        print(f'❌ 找到 {len(run_dirs)} 个运行目录，但没有任何 episode_*/metrics.json')
        return 2

    conds = group_conditions(all_rows)
    out_dir = args.out_dir or os.path.join(args.root, '_aggregate' + (f'_{args.label}' if args.label else ''))
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, 'episodes.csv'), EPISODE_FIELDS, all_rows)
    write_csv(os.path.join(out_dir, 'conditions.csv'), CONDITION_FIELDS, conds)

    # ---- 质量报告：先看数据能不能用，再看成功率 ------------------------------
    q_lines = ['数据可信度清单（先读这个，再读成功率）', '=' * 78]
    no_meta = [q for q in qualities if not q['metadata_present']]
    no_new = [q for q in qualities if not q['has_new_logging']]
    mismatched = [q for q in qualities
                  if q['episodes_in_summary'] is not None and q['episodes_in_summary'] != q['episodes_on_disk']]
    archived_total = sum(q['archived_attempts'] for q in qualities)
    q_lines.append(f'运行目录数: {len(qualities)}   记录集数: {len(all_rows)}   条件数: {len(conds)}')
    q_lines.append(f'缺 condition.json 的运行: {len(no_meta)}'
                   + ('  ← 参数只能靠目录名/命令历史推断，跨条件比较不可靠' if no_meta else '  ✅'))
    q_lines.append(f'缺新日志字段的运行: {len(no_new)}'
                   + ('  ← 无逐步 reward / 尝试次数 / MPC 代价序列（补丁前数据）' if no_new else '  ✅'))
    q_lines.append(f'summary 与磁盘集数不一致的运行: {len(mismatched)}'
                   + ('  ← 有集被重试/删除，分母需人工确认' if mismatched else '  ✅'))
    q_lines.append(f'归档的被重试尝试目录: {archived_total}'
                   + ('  ← 这些是被重启/崩溃/丢弃的尝试（想算"每集平均尝试次数"要用它）' if archived_total else '  （KEEP_FAILED_EPISODES 未生效或没有重试）'))
    if no_meta:
        q_lines.append('')
        q_lines.append('缺元数据的运行目录：')
        q_lines += [f'  - {q["run_dir"]}' for q in no_meta[:20]]
    if mismatched:
        q_lines.append('')
        q_lines.append('summary/磁盘不一致的运行目录（summary_total | 磁盘实际）：')
        q_lines += [f'  - {q["run_dir"]}: {q["episodes_in_summary"]} | {q["episodes_on_disk"]}'
                    f'（归档尝试 {q["archived_attempts"]}）' for q in mismatched[:20]]
    q_lines.append('')
    q_lines.append('注意：avg_ref_vs_mpc_* / avg_mpc_vs_actual_* / main_mpc_tracking_costs 在未开')
    q_lines.append('      --log_diffusion 的旧数据里恒为 0，是"未采集"而不是"跟踪完美"。')
    q_text = '\n'.join(q_lines)
    with open(os.path.join(out_dir, 'quality_report.txt'), 'w', encoding='utf-8') as f:
        f.write(q_text + '\n')

    if not args.quiet:
        print(q_text)
        print()
        print('条件级汇总（成功率含 Wilson 95% CI）')
        print('=' * 78)
        hdr = (f'{"条件":<22}{"n":>4}{"成功":>6}{"成功率":>9}{"95% CI":>16}'
               f'{"return±std":>16}{"尝试":>6}{"重启":>6}  失败模式')
        print(hdr)
        print('-' * 78)
        for c in conds:
            ret = f'{fmt(c["return_mean"], 2)}±{fmt(c["return_std"], 2)}'
            ci = f'[{fmt(c["wilson_lo"], 2)},{fmt(c["wilson_hi"], 2)}]'
            print(f'{c["group"][:21]:<22}{c["n_episodes"]:>4}{c["n_success"]:>6}'
                  f'{c["success_rate"]*100:>8.1f}%{ci:>16}{ret:>16}'
                  f'{fmt(c["attempts_mean"], 2):>6}{fmt(c["restarts_mean"], 2):>6}  '
                  f'{c.get("outcome_counts", "")}')
        print('=' * 78)

    print(f'\n✅ 已写出:\n  {os.path.join(out_dir, "episodes.csv")}\n'
          f'  {os.path.join(out_dir, "conditions.csv")}\n'
          f'  {os.path.join(out_dir, "quality_report.txt")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_episode.py —— 单集深挖（失败归因/成功过程），纯标准库。

回答的问题：
  · 这一集是"跑满集长未达标"，还是"起点就被重启上限截断"，还是崩溃？
  · MPC 代价在失败前是否持续升高？有没有越过阈值（默认 10.0）？越过时是否还在重启窗口（默认 500 步）内？
  · 达标发生在第几步（`first_success_step`）？reward 序列里达标之后有没有掉回去？
  · 跟踪误差的均值与**末尾值**（末尾值更能说明"卡在哪"）。

用法：
    python3 tools/analyze_episode.py <episode_dir | metrics.json> [--threshold 10.0] [--window 500] [--csv out.csv]
例：
    python3 tools/analyze_episode.py /root/autodl-tmp/results/eval/d_base/seed0/episode_003
    python3 tools/analyze_episode.py ./d_base/seed0/episode_003 --csv cost.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys


def load(path):
    """读 metrics.json；episode_len 从 metrics → 同级 condition.json → 默认 3000 依次兜底。

    注意：`episode_len` 只写在 condition.json 里（metrics.json 没有），
    独立跑本工具时若不读 condition.json，就会把"跑满 3000 步"误判成"提前结束"。
    """
    if os.path.isdir(path):
        path = os.path.join(path, 'metrics.json')
    with open(path, encoding='utf-8') as f:
        m = json.load(f)
    if 'episode_len' not in m:
        cond_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))),
                                 'condition.json')
        if os.path.exists(cond_path):
            try:
                with open(cond_path, encoding='utf-8') as f:
                    m['episode_len'] = json.load(f).get('episode_len')
                m['_episode_len_source'] = os.path.basename(cond_path)
            except Exception:
                pass
    return m, path


def main():
    ap = argparse.ArgumentParser(description='单集深挖（失败归因）')
    ap.add_argument('path', help='episode 目录或 metrics.json')
    ap.add_argument('--threshold', type=float, default=10.0,
                    help='MPC 代价阈值（默认 10.0，对应 imitate_episodes.py 的 MPC_COST_THRESHOLD）')
    ap.add_argument('--window', type=int, default=500,
                    help='重启窗口步数（默认 500，对应 MPC_RESTART_WINDOW）')
    ap.add_argument('--csv', default=None, help='把每次推理的 (step, cost) 导出为 CSV')
    args = ap.parse_args()

    m, path = load(args.path)
    steps = m.get('num_timesteps')
    rs = m.get('reward_series') or []
    costs = m.get('inference_mpc_costs') or []
    events = m.get('index_restart_events') or []
    successes = [i for i, r in enumerate(rs) if r]

    print(f'📄 {path}')
    print('=' * 78)
    print(f'结果        : success={m.get("success")}  return={m.get("episode_return")}  '
          f'highest={m.get("highest_reward")}  crashed={m.get("crashed")}')
    print(f'步数/尝试   : num_timesteps={steps}  attempts={m.get("index_attempts")}  '
          f'restarts={m.get("index_restarts")}  episode_len={m.get("episode_len", "n/a")}')
    print(f'达标        : first_success_step={m.get("first_success_step")}  '
          f'reward_series 长度={len(rs)}  达标点数={len(successes)}')
    print(f'失败原因    : {m.get("failure_reasons") or "(空)"}')

    # 归因
    if m.get('success'):
        verdict = 'success'
    elif m.get('crashed'):
        verdict = 'crash'
    elif steps is None:
        verdict = 'legacy（补丁前数据，无 num_timesteps，无法归因）'
    elif steps == 0:
        verdict = (f'aborted_no_steps：所有尝试都在起点被重启/崩溃掉（attempts={m.get("index_attempts")}，'
                   f'最终没有跑出任何步数）')
    elif steps >= 0.9 * (m.get('episode_len') or 3000):
        src = '（集长来自 condition.json）' if m.get('_episode_len_source') else \
              '（未读到 condition.json，按默认集长 3000 判定）'
        verdict = f'timeout_full_episode：跑满 {steps} 步仍未达标{src}'
    else:
        verdict = f'early_stop_other：第 {steps} 步提前结束且未达标'
    print(f'归因        : {verdict}')

    if rs:
        print(f'reward 序列 : 达标点 {successes[:3]}{"..." if len(successes) > 3 else ""}  '
              f'末段 20 步求和={sum(rs[-20:])}  全程 1 的个数={sum(rs)}')
    print()

    # MPC 代价
    if costs:
        vals = [float(c) for _s, c in costs]
        early = [c for s, c in costs if s < args.window]
        late = [c for s, c in costs if s >= args.window]
        print(f'MPC 代价    : 推理 {len(vals)} 次  首={vals[0]:.3f}  末={vals[-1]:.3f}  '
              f'均={statistics.fmean(vals):.3f}  峰={max(vals):.3f}')
        print(f'  窗口内(<{args.window} 步) : n={len(early)} 均={statistics.fmean(early):.3f}' if early else
              f'  窗口内(<{args.window} 步) : 无')
        print(f'  窗口外(>={args.window} 步): n={len(late)} 均={statistics.fmean(late):.3f}'
              if late else f'  窗口外(>={args.window} 步): 无')
        over = [(s, c) for s, c in costs if c > args.threshold]
        print(f'  超过阈值 {args.threshold} 的次数: {len(over)}'
              + (f'  首次在 step {over[0][0]}（代价 {over[0][1]:.2f}）' if over else ''))
        if over and over[0][0] >= args.window:
            print('  ⚠️ 首次越阈发生在重启窗口之外 ⇒ 不会触发重启，只打印告警（这集的失败不是重启机制造成的）')
        # 代价趋势（前 3 / 后 3）
        print(f'  前 3 次: {[round(v,2) for v in vals[:3]]}   后 3 次: {[round(v,2) for v in vals[-3:]]}')
    else:
        print('MPC 代价    : 无记录（非 UAM 任务，或补丁前数据）')

    if events:
        print(f'重启触发点  : {len(events)} 次 → ' +
              ', '.join(f'(step={s}, cost={c:.2f})' for s, c in events[:6])
              + (' …' if len(events) > 6 else ''))
    print()

    # 跟踪误差
    ts = (m.get('timestep_data') or {})
    pos = ts.get('position_rmse') or []
    ori = ts.get('orientation_distance') or []
    if pos:
        print(f'位置 RMSE   : 均={statistics.fmean(pos):.5f}  末 10 步均={statistics.fmean(pos[-10:]):.5f}  '
              f'最大={max(pos):.5f}')
    if ori:
        print(f'姿态距离    : 均={statistics.fmean(ori):.5f}  末 10 步均={statistics.fmean(ori[-10:]):.5f}')
    q = (m.get('qpos_data') or {})
    tq, aq = q.get('target_qpos') or [], q.get('actual_qpos') or []
    if tq and aq:
        d = [abs(a - b) for a, b in zip(tq[-1][:3], aq[-1][:3])]
        print(f'末帧目标-实际(位置3轴绝对差): {[round(x,4) for x in d]}  最大值={max(d):.4f}')
    print('=' * 78)

    if args.csv and costs:
        with open(args.csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['step', 'mpc_cost'])
            w.writerows([[int(s), float(c)] for s, c in costs])
        print(f'✅ 代价序列已导出: {args.csv}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

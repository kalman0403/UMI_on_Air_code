#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_calibration.py —— 扰动标定判读：每档风力下的"基座跟踪误差"（cm）

论文口径：`--disturb` 应模拟硬件上 **~3 cm 的平均跟踪误差**（论文原文："inject noise into the
UAM base to simulate the ∼3 cm average tracking error observed on hardware when hovering near
a still target"）。本仓库实现是 **±8 N 三轴力偏置**（`constants.py:68`；`ee_sim_env.py` 阵风模型），
量级未必等价，所以需要先测：跑若干档 `EVAL_WIND_RANGE={0,8,16,24,32}`，取基座
`|actual_pos − target_pos|`（`qpos_data` 前 3 维）的统计量，选最接近 3.0 cm 的档位。

两种窗口都报：
  · 全程  ：整集平均（含接触/插入阶段，尾部尖峰会把均值抬高）
  · 前 N 步（默认 500，= 10 s）：接近论文说的"悬停在静止目标附近"阶段 ⇒ **主判据用这个**

用法：
    python3 analyze_calibration.py <结果根目录> [--window 500] [--target-cm 3.0]
    # 例：python3 analyze_calibration.py /root/autodl-tmp/results/eval --window 500
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys


def base_err_cm(metrics_path, window=None):
    """返回该集基座位置误差序列（cm）；window 只取前 N 步"""
    try:
        m = json.load(open(metrics_path, encoding='utf-8'))
    except Exception:
        return None
    q = m.get('qpos_data') or {}
    tq, aq = q.get('target_qpos') or [], q.get('actual_qpos') or []
    if not tq or not aq:
        return None
    n = min(len(tq), len(aq), window) if window else min(len(tq), len(aq))
    errs = []
    for t, a in zip(tq[:n], aq[:n]):
        d = ((a[0] - t[0]) ** 2 + (a[1] - t[1]) ** 2 + (a[2] - t[2]) ** 2) ** 0.5
        errs.append(d * 100.0)          # m → cm
    return errs or None


def pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(q * (len(xs) - 1)))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--window', type=int, default=500, help='只统计前 N 步（默认 500 = 10 s 悬停段）')
    ap.add_argument('--target-cm', type=float, default=3.0)
    ap.add_argument('--tag-prefix', default='cal_wind', help='只统计该前缀的条件（默认标定批）')
    args = ap.parse_args()

    # 收集每个 cal_wind* 条件的每集误差
    conds = {}
    for d in sorted(glob.glob(os.path.join(args.root, '**', 'condition.json'), recursive=True)):
        run_dir = os.path.dirname(d)
        try:
            cond = json.load(open(d, encoding='utf-8'))
        except Exception:
            continue
        tag = str(cond.get('tag') or '')
        if not tag.startswith(args.tag_prefix):
            continue
        wr = cond.get('wind_range')
        eps = []
        for mp in sorted(glob.glob(os.path.join(run_dir, 'episode_*', 'metrics.json'))):
            e = base_err_cm(mp, args.window)
            if e:
                eps.append((os.path.basename(os.path.dirname(mp)), e, mp))
        if eps:
            conds[tag] = dict(wind_range=wr, episodes=eps, run_dir=run_dir)

    if not conds:
        print(f'❌ 在 {args.root} 下没找到 tag 以 {args.tag_prefix!r} 开头的条件（先跑标定批）')
        return 2

    print(f'扰动标定判读（窗口=前 {args.window} 步；目标 ≈ {args.target_cm} cm）')
    print('=' * 104)
    print(f'{"条件":<14}{"wind(N)":>10}{"集数":>5}{"均值cm":>9}{"中位cm":>9}{"p90cm":>8}'
          f'{"对比0档增量cm":>14}   逐集均值')
    print('-' * 104)
    ref_mean = None
    rows = []
    for tag in sorted(conds, key=lambda t: (conds[t]['wind_range'][1] if conds[t]['wind_range'] else 0)):
        c = conds[tag]
        per_ep = [statistics.fmean(e) for _n, e, _p in c['episodes']]
        all_e = [x for _n, e, _p in c['episodes'] for x in e]
        m, md = statistics.fmean(per_ep), statistics.median(all_e)
        p90 = pct(all_e, 0.9)
        wr = c['wind_range'] or [None, None]
        if wr[1] == 0:
            ref_mean = m
        rows.append((tag, wr[1], len(per_ep), m, md, p90))
    for tag, w, n, m, md, p90 in rows:
        delta = f'{m - ref_mean:+.2f}' if ref_mean is not None else '  -'
        print(f'{tag:<14}{(w if w is not None else -1):>10.0f}{n:>5}{m:>9.2f}{md:>9.2f}{p90:>8.2f}'
              f'{delta:>14}   ' + ' '.join(f'{x:.2f}' for x in
                                           [statistics.fmean(e) for _n, e, _p in conds[tag]['episodes']]))
    print('=' * 104)

    # 选最接近目标的档位（用均值；若均值被接触尖峰抬高，中位列可作交叉验证）
    known = [r for r in rows if r[1] is not None]
    if not known:
        print('\n⚠️ 这些条件的 condition.json 里都没有 wind_range 字段（补丁前/旧数据）⇒ 无法自动选档。')
        print('   请用带该字段的新标定批（EVAL_WIND_RANGE=… 跑出来的）重新判读。')
        return 3
    best = min(known, key=lambda r: abs(r[3] - args.target_cm))
    print(f'\n最接近 {args.target_cm} cm 的档位：**{best[0]}**（wind=±{best[1]:.0f} N，均值 {best[3]:.2f} cm，'
          f'中位 {best[4]:.2f} cm）')
    if abs(best[3] - args.target_cm) > 1.0:
        lo = [r for r in known if r[3] < args.target_cm]
        hi = [r for r in known if r[3] > args.target_cm]
        print('⚠️ 与目标差 >1 cm：建议按线性插值补一档 —— '
              + (f'在 ±{lo[-1][1]:.0f} 与 ±{hi[0][1]:.0f} N 之间' if lo and hi
                 else f'需要更大的风力（当前最大档 ±{max(r[1] for r in known):.0f} N 仍偏小）'))
    print(f'\n下一步：用选定档位跑主批（关守卫）：')
    print(f'  EVAL_WIND_RANGE={best[1]:.0f} EVAL_MPC_COST_THRESHOLD=1e12 PARALLEL=2 \\')
    print(f'    bash tools/run_campaign.sh /root/autodl-tmp/campaign_fig7.tsv')
    return 0


if __name__ == '__main__':
    sys.exit(main())

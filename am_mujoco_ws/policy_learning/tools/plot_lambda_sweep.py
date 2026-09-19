#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_lambda_sweep.py —— 把 λ 扫描结果按 §C 的判读规则整理成表/图。

对应 `参数口径与实验设计_配置差异与复现策略.md` §5：
  · 主实验是模式 A（--scale λ --guided_steps 1），λ=0 即基线；
  · 模式 B（--guidance）单独报，不混算；
  · 报成功率（含 Wilson 95% CI）、尝试/重启次数、MPC 代价；
  · 条件间共用同一组种子 ⇒ 额外给"按种子配对的差值"。

输入可以是：
  1) aggregate_results.py 的输出目录（含 conditions.csv + episodes.csv）；
  2) 或直接给结果根目录（内部调用 aggregate_results 现场汇总）。
只依赖标准库；若装了 matplotlib 且加 --plot 才画图（复刻论文 Fig.7 形状）。

用法：
    python3 tools/plot_lambda_sweep.py <agg_dir|results_root> [--plot] [--out-dir DIR]
    python3 tools/plot_lambda_sweep.py /root/autodl-tmp/results/eval/_aggregate_matrix --plot
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate_results as agg  # noqa: E402  （同目录复用汇总逻辑与 Wilson 区间）


def load_from_agg_dir(d: str):
    """读 aggregate_results.py 的产物（conditions.csv / episodes.csv）"""
    conds_path = os.path.join(d, 'conditions.csv')
    eps_path = os.path.join(d, 'episodes.csv')
    if not os.path.exists(conds_path):
        raise SystemExit(f'❌ 找不到 {conds_path}（先跑 aggregate_results.py）')

    def num(x):
        if x in ('', None, 'None'):
            return None
        try:
            return float(x)
        except ValueError:
            return x

    with open(conds_path, encoding='utf-8') as f:
        conds = [{k: num(v) for k, v in row.items()} for row in csv.DictReader(f)]
    rows = []
    if os.path.exists(eps_path):
        with open(eps_path, encoding='utf-8') as f:
            rows = [{k: num(v) for k, v in row.items()} for row in csv.DictReader(f)]
    return rows, conds


def classify(conds):
    """拆成 (λ 序列, 模式 B 行, 其他行)"""
    lam, modeb, other = [], [], []
    for c in conds:
        mode = (c.get('mode') or '').strip()
        if mode in ('scale', 'baseline'):
            l = c.get('scale') or 0.0
            lam.append((float(l), c))
        elif mode == 'guidance':
            modeb.append(c)
        else:
            other.append(c)
    lam.sort(key=lambda t: t[0])
    return lam, modeb, other


def fmt(x, nd=3):
    return '  -  ' if x is None else f'{x:.{nd}f}'


def cnt(x):
    """计数列：整数值不带小数点（CSV 读回来是 float）"""
    if x is None:
        return '  -  '
    return str(int(x)) if float(x).is_integer() else f'{x:.1f}'


def per_seed_rates(rows, group):
    """按种子算成功率（配对用）"""
    by_seed = {}
    for r in rows:
        if r.get('group') != group:
            continue
        by_seed.setdefault(r.get('seed'), []).append(1 if r.get('success') in (True, 'True', 1, '1') else 0)
    return {s: (sum(v) / len(v) if v else None) for s, v in by_seed.items()}


def main():
    ap = argparse.ArgumentParser(description='λ 扫描判读（表 + 可选图）')
    ap.add_argument('target', help='aggregate 输出目录，或结果根目录')
    ap.add_argument('--plot', action='store_true', help='尝试用 matplotlib 画 λ–成功率 曲线')
    ap.add_argument('--out-dir', default=None, help='图/表输出目录（默认写回 target）')
    ap.add_argument('--ylabel-task', default=None, help='标题里的任务/场景说明')
    args = ap.parse_args()

    if os.path.exists(os.path.join(args.target, 'conditions.csv')):
        rows, conds = load_from_agg_dir(args.target)
        out_dir = args.out_dir or args.target
    else:                                   # 当成结果根目录，现场汇总
        run_dirs = agg.find_run_dirs(args.target)
        rows, _ = [], []
        for rd in run_dirs:
            r, _q = agg.collect_run(rd, args.target)
            rows.extend(r)
        conds = agg.group_conditions(rows)
        out_dir = args.out_dir or args.target
    os.makedirs(out_dir, exist_ok=True)

    lam, modeb, other = classify(conds)
    if not lam:
        print('❌ 没有 mode ∈ {baseline, scale} 的条件行，无法做 λ 扫描判读')
        return 2

    base_rate = next((c['success_rate'] for l, c in lam if l == 0.0), None)
    base_group = next((c['group'] for l, c in lam if l == 0.0), None)

    print('λ 扫描（模式 A：--scale λ --guided_steps 1；λ=0 即基线）')
    print('=' * 104)
    print(f'{"λ":>6}{"条件":>22}{"n":>5}{"成功":>6}{"成功率":>9}{"95% CI":>16}'
          f'{"尝试":>7}{"重启":>7}{"首代价":>10}{"峰值代价":>11}')
    print('-' * 104)
    for l, c in lam:
        ci = f'[{fmt(c.get("wilson_lo"),2)},{fmt(c.get("wilson_hi"),2)}]'
        print(f'{l:>6.2f}{str(c.get("group"))[:21]:>22}{cnt(c["n_episodes"]):>5}{cnt(c["n_success"]):>6}'
              f'{c["success_rate"]*100:>8.1f}%{ci:>16}'
              f'{fmt(c.get("attempts_mean"),2):>7}{fmt(c.get("restarts_mean"),2):>7}'
              f'{fmt(c.get("first_mpc_cost_mean"),2):>10}{fmt(c.get("max_mpc_cost_mean"),2):>11}')
    print('=' * 104)

    # 按种子配对：同种子下 λ 与基线的成功率之差
    base_seed = per_seed_rates(rows, base_group) if base_group else {}
    if base_seed and any(v is not None for v in base_seed.values()):
        print('\n按种子配对（同种子：λ − 基线，单位=成功率）')
        print('-' * 104)
        for l, c in lam:
            if l == 0.0:
                continue
            cur = per_seed_rates(rows, c['group'])
            deltas = [cur[s] - base_seed[s] for s in sorted(cur)
                      if s in base_seed and cur[s] is not None and base_seed[s] is not None]
            if deltas:
                print(f'  λ={l:<5} 各种子差值 {[round(d,3) for d in deltas]}  '
                      f'均值 {statistics.fmean(deltas):+.3f}'
                      + (f' ± {statistics.pstdev(deltas):.3f}' if len(deltas) > 1 else '')
                      + f'（种子 {sorted(cur)}）')

    if modeb:
        print('\n模式 B（--guidance，单列报告，不与模式 A 混算）')
        print('-' * 104)
        for c in modeb:
            print(f'  {str(c.get("group"))[:28]:<30} guidance={c.get("guidance")} steps={c.get("guided_steps")} '
                  f'n={cnt(c["n_episodes"])} 成功={cnt(c["n_success"])} 成功率={c["success_rate"]*100:.1f}% '
                  f'CI=[{fmt(c.get("wilson_lo"),2)},{fmt(c.get("wilson_hi"),2)}] '
                  f'尝试={fmt(c.get("attempts_mean"),2)} 重启={fmt(c.get("restarts_mean"),2)}')
    if other:
        print(f'\n其余条件 {len(other)} 个（可能是旧数据/无元数据）：'
              + ', '.join(str(c.get("group"))[:26] for c in other[:6]))

    # 结论文本（写文件，便于粘进论文/答复）
    txt = os.path.join(out_dir, 'lambda_sweep_summary.txt')
    with open(txt, 'w', encoding='utf-8') as f:
        f.write('λ 扫描判读（模式 A）\n')
        for l, c in lam:
            lo, hi = c.get("wilson_lo"), c.get("wilson_hi")
            ci = f'[{lo:.3f},{hi:.3f}]' if isinstance(lo, float) and isinstance(hi, float) else 'n/a'
            f.write(f'λ={l:<5} n={cnt(c["n_episodes"]):<3} 成功={cnt(c["n_success"]):<3} '
                    f'成功率={c["success_rate"]:.3f} Wilson95={ci} '
                    f'attempts={c.get("attempts_mean")} restarts={c.get("restarts_mean")} '
                    f'first_cost={c.get("first_mpc_cost_mean")} max_cost={c.get("max_mpc_cost_mean")}\n')
        if base_rate is not None:
            best = max(lam, key=lambda t: t[1]['success_rate'])
            f.write(f'\n基线成功率={base_rate:.3f}；本批最高 λ={best[0]}（{best[1]["success_rate"]:.3f}）\n')
            f.write('判读要点：CI 是否重叠、attempts/restarts 是否随 λ 单调上升、'
                    '失败集是被上限截断还是跑满 3000 步\n')
    print(f'\n✅ 结论文本: {txt}')

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:                                    # pragma: no cover
            print(f'⚠️ 未装 matplotlib，跳过画图（{e}）')
            return 0
        xs = [l for l, _ in lam]
        ys = [c['success_rate'] for _, c in lam]
        lo = [c['success_rate'] - (c.get('wilson_lo') or 0) for _, c in lam]
        hi = [(c.get('wilson_hi') or 0) - c['success_rate'] for _, c in lam]
        fig, ax = plt.subplots(figsize=(6, 4.2))
        ax.errorbar(xs, ys, yerr=[lo, hi], marker='o', capsize=4, color='#1f77b4')
        for l, c in lam:
            ax.annotate(f'{cnt(c["n_success"])}/{cnt(c["n_episodes"])}', (l, c['success_rate']),
                        textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8)
        ax.set_xlabel('λ (mode A: --scale λ, --guided_steps 1)')
        ax.set_ylabel('success rate')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
        ax.set_title('λ sweep' + (f' — {args.ylabel_task}' if args.ylabel_task else '')
                     + ' (error bars: Wilson 95% CI)')
        png = os.path.join(out_dir, 'lambda_sweep.png')
        fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)
        print(f'✅ 图: {png}')
    return 0


if __name__ == '__main__':
    sys.exit(main())

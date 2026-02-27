"""
批量对比 random vs adaptive 推理策略。

功能：
1) 对多个 PDB 和多个随机种子运行 inference.py --strategy both
2) 解析“策略对比汇总”指标
3) 保存明细 CSV
4) 打印均值/标准差和配对差值（adaptive - random）

使用示例：
python scripts/benchmark_inference_strategies.py \
  --model_path checkpoints_20000/best_model.pt \
  --pdb_dir raw_data \
  --max_samples 100 \
  --seeds 42,123,2026 \
  --num_iterations 12 \
  --min_commit_ratio 0.05 \
  --max_commit_ratio 0.20 \
  --output_csv benchmark_results.csv
"""

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import defaultdict
from statistics import mean, pstdev


SUMMARY_RE = re.compile(
    r"^\s*(random|adaptive)\s+\|\s+"
    r"FragAcc=([0-9.]+)\s+\|\s+"
    r"ResExact=([0-9.]+)\s+\|\s+"
    r"Coverage=([0-9.]+)\s+\|\s+"
    r"RMSD_all=([0-9.]+)\s+\|\s+"
    r"RMSD_matched=([0-9.]+)\s+\|\s+"
    r"Clash=([0-9.]+)\s*$"
)


def parse_summary(stdout: str):
    """从 inference.py 输出中解析 random/adaptive 指标。"""
    result = {}
    for line in stdout.splitlines():
        m = SUMMARY_RE.match(line)
        if not m:
            continue
        strategy = m.group(1)
        result[strategy] = {
            "frag_acc": float(m.group(2)),
            "res_exact": float(m.group(3)),
            "coverage": float(m.group(4)),
            "rmsd_all": float(m.group(5)),
            "rmsd_matched": float(m.group(6)),
            "clash": float(m.group(7)),
        }
    return result


def agg(values):
    if len(values) == 0:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), pstdev(values)


def main():
    parser = argparse.ArgumentParser(description="批量对比 random vs adaptive 推理策略")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="benchmark_inference_strategies.csv")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--num_iterations", type=int, default=12)
    parser.add_argument("--min_commit_ratio", type=float, default=0.05)
    parser.add_argument("--max_commit_ratio", type=float, default=0.20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save_fail_logs_dir", type=str, default="benchmark_fail_logs")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inference_py = os.path.join(project_root, "inference.py")

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(seeds) == 0:
        raise ValueError("--seeds 不能为空")

    if not os.path.isdir(args.pdb_dir):
        raise ValueError(f"pdb_dir 不存在: {args.pdb_dir}")

    pdb_files = sorted(
        os.path.join(args.pdb_dir, f)
        for f in os.listdir(args.pdb_dir)
        if f.endswith(".pdb")
    )
    if len(pdb_files) == 0:
        raise ValueError(f"未在目录中找到 .pdb 文件: {args.pdb_dir}")

    pdb_files = pdb_files[: args.max_samples]
    os.makedirs(args.save_fail_logs_dir, exist_ok=True)

    print("=" * 72)
    print("Batch Benchmark: random vs adaptive")
    print("=" * 72)
    print(f"模型: {args.model_path}")
    print(f"样本数: {len(pdb_files)}")
    print(f"种子: {seeds}")
    print(f"迭代: {args.num_iterations}, ratio=[{args.min_commit_ratio}, {args.max_commit_ratio}]")
    print("=" * 72)

    rows = []
    failed = 0
    total_jobs = len(pdb_files) * len(seeds)
    done = 0

    for pdb_path in pdb_files:
        pdb_name = os.path.basename(pdb_path)
        for seed in seeds:
            done += 1
            cmd = [
                sys.executable,
                inference_py,
                "--model_path", args.model_path,
                "--pdb_path", pdb_path,
                "--strategy", "both",
                "--num_iterations", str(args.num_iterations),
                "--min_commit_ratio", str(args.min_commit_ratio),
                "--max_commit_ratio", str(args.max_commit_ratio),
                "--seed", str(seed),
            ]
            if args.device:
                cmd += ["--device", args.device]

            proc = subprocess.run(
                cmd,
                cwd=project_root,
                capture_output=True,
                text=True,
            )

            if proc.returncode != 0:
                failed += 1
                fail_log = os.path.join(
                    args.save_fail_logs_dir,
                    f"{os.path.splitext(pdb_name)[0]}_seed{seed}.log"
                )
                with open(fail_log, "w", encoding="utf-8") as f:
                    f.write(proc.stdout)
                    f.write("\n\n[STDERR]\n")
                    f.write(proc.stderr)
                print(f"[{done}/{total_jobs}] FAIL {pdb_name} seed={seed} -> {fail_log}")
                continue

            parsed = parse_summary(proc.stdout)
            if "random" not in parsed or "adaptive" not in parsed:
                failed += 1
                fail_log = os.path.join(
                    args.save_fail_logs_dir,
                    f"{os.path.splitext(pdb_name)[0]}_seed{seed}_parse.log"
                )
                with open(fail_log, "w", encoding="utf-8") as f:
                    f.write(proc.stdout)
                    f.write("\n\n[STDERR]\n")
                    f.write(proc.stderr)
                print(f"[{done}/{total_jobs}] PARSE_FAIL {pdb_name} seed={seed} -> {fail_log}")
                continue

            for strategy in ("random", "adaptive"):
                r = parsed[strategy]
                rows.append({
                    "pdb": pdb_name,
                    "seed": seed,
                    "strategy": strategy,
                    **r,
                })
            print(f"[{done}/{total_jobs}] OK {pdb_name} seed={seed}")

    # 保存 CSV
    fieldnames = [
        "pdb", "seed", "strategy",
        "frag_acc", "res_exact", "coverage",
        "rmsd_all", "rmsd_matched", "clash",
    ]
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 72)
    print(f"完成。成功记录: {len(rows)} 行, 失败任务: {failed}, 输出: {args.output_csv}")

    # 按策略统计
    by_strategy = defaultdict(lambda: defaultdict(list))
    for r in rows:
        s = r["strategy"]
        for k in ("frag_acc", "res_exact", "coverage", "rmsd_all", "rmsd_matched", "clash"):
            by_strategy[s][k].append(float(r[k]))

    print("\n策略均值 ± 标准差:")
    for s in ("random", "adaptive"):
        if s not in by_strategy:
            continue
        print(f"  [{s}]")
        for k in ("frag_acc", "res_exact", "coverage", "rmsd_all", "rmsd_matched", "clash"):
            m, sd = agg(by_strategy[s][k])
            print(f"    {k:12s}: {m:.4f} ± {sd:.4f}")

    # 配对差值 adaptive - random
    paired = defaultdict(dict)
    for r in rows:
        key = (r["pdb"], int(r["seed"]))
        paired[key][r["strategy"]] = r

    deltas = defaultdict(list)
    for key, val in paired.items():
        if "random" not in val or "adaptive" not in val:
            continue
        rr = val["random"]
        aa = val["adaptive"]
        for k in ("frag_acc", "res_exact", "coverage", "rmsd_all", "rmsd_matched", "clash"):
            deltas[k].append(float(aa[k]) - float(rr[k]))

    if len(deltas) > 0 and len(deltas["frag_acc"]) > 0:
        print("\n配对差值 (adaptive - random):")
        for k in ("frag_acc", "res_exact", "coverage", "rmsd_all", "rmsd_matched", "clash"):
            m, sd = agg(deltas[k])
            print(f"  {k:12s}: {m:+.4f} ± {sd:.4f}")

    print("=" * 72)


if __name__ == "__main__":
    main()

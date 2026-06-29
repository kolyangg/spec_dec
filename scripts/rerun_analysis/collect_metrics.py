#!/usr/bin/env python3
"""Collect benchmark sweep metrics into notebook-ready files."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


THRESHOLDS = {
    "spec_bf16": 1250.0,
    "fp8": 1550.0,
    "fp8_spec": 1750.0,
}


def classify(stem: str) -> tuple[str, int | None] | None:
    if "warmup" in stem:
        return None
    if stem.startswith("fp8_spec_"):
        match = re.search(r"fp8_spec_n(\d+)", stem)
        return ("fp8_spec", int(match.group(1)) if match else None)
    if stem.startswith("spec_bf16_"):
        match = re.search(r"spec_bf16_n(\d+)", stem)
        return ("spec_bf16", int(match.group(1)) if match else None)
    if stem.startswith("baseline_bf16"):
        return ("baseline_bf16", None)
    if stem.startswith("fp8"):
        return ("fp8", None)
    return None


def load_rows(bench_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(bench_dir.glob("*.json")):
        classified = classify(path.stem)
        if classified is None:
            continue
        group, num_speculative_tokens = classified
        data = json.loads(path.read_text())
        row = {
            "file": path.name,
            "stem": path.stem,
            "group": group,
            "num_speculative_tokens": num_speculative_tokens,
            "output_throughput": data.get("output_throughput"),
            "total_token_throughput": data.get("total_token_throughput"),
            "request_throughput": data.get("request_throughput"),
            "mean_ttft_ms": data.get("mean_ttft_ms"),
            "p99_ttft_ms": data.get("p99_ttft_ms"),
            "mean_tpot_ms": data.get("mean_tpot_ms"),
            "p99_tpot_ms": data.get("p99_tpot_ms"),
            "mean_itl_ms": data.get("mean_itl_ms"),
            "acceptance_rate": data.get("spec_decode_acceptance_rate"),
            "acceptance_length": data.get("spec_decode_acceptance_length"),
            "drafts": data.get("spec_decode_num_drafts"),
            "draft_tokens": data.get("spec_decode_draft_tokens"),
            "accepted_tokens": data.get("spec_decode_accepted_tokens"),
            "per_position_acceptance": data.get(
                "spec_decode_per_position_acceptance_rates"
            ),
            "completed": data.get("completed"),
            "failed": data.get("failed"),
            "duration": data.get("duration"),
        }
        rows.append(row)
    return rows


def best_by_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for group in ["baseline_bf16", "spec_bf16", "fp8", "fp8_spec"]:
        candidates = [
            row
            for row in rows
            if row["group"] == group and row.get("output_throughput") is not None
        ]
        if candidates:
            best[group] = max(candidates, key=lambda row: row["output_throughput"])
    return best


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def text_block(bench_dir: Path, stem: str) -> str:
    txt_path = bench_dir / f"{stem}.txt"
    if not txt_path.exists():
        return f"No text output found for {stem}."
    text = txt_path.read_text(errors="replace")
    marker = "============ Serving Benchmark Result ============"
    idx = text.find(marker)
    return text[idx:].strip() if idx >= 0 else text.strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "stem",
        "group",
        "num_speculative_tokens",
        "output_throughput",
        "total_token_throughput",
        "mean_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "acceptance_rate",
        "acceptance_length",
        "drafts",
        "draft_tokens",
        "accepted_tokens",
        "completed",
        "failed",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(path: Path, bench_dir: Path, rows: list[dict[str, Any]]) -> None:
    best = best_by_group(rows)
    lines = [
        "# Tuned Benchmark Results",
        "",
        "## Best-Run Comparison",
        "",
        "| Configuration | Source run | Draft tokens | Output tok/s | Mean TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    labels = [
        ("baseline_bf16", "Baseline BF16"),
        ("spec_bf16", "EAGLE-3 BF16"),
        ("fp8", "FP8"),
        ("fp8_spec", "FP8 + EAGLE-3"),
    ]
    for group, label in labels:
        row = best.get(group)
        if not row:
            lines.append(f"| {label} | missing | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        acc = fmt(row["acceptance_rate"]) + "%" if row["acceptance_rate"] is not None else "N/A"
        lines.append(
            f"| {label} | `{row['stem']}` | {fmt(row['num_speculative_tokens'], 0)} | "
            f"`{fmt(row['output_throughput'])}` | `{fmt(row['mean_ttft_ms'])}` | "
            f"`{fmt(row['mean_tpot_ms'])}` | {acc} | `{fmt(row['acceptance_length'])}` |"
        )

    lines.extend(["", "## Threshold Check", ""])
    lines.append("| Configuration | Threshold | Observed | Result |")
    lines.append("| --- | ---: | ---: | --- |")
    for group, label in [("spec_bf16", "EAGLE-3 BF16"), ("fp8", "FP8"), ("fp8_spec", "FP8 + EAGLE-3")]:
        row = best.get(group)
        threshold = THRESHOLDS[group]
        observed = row.get("output_throughput") if row else None
        result = "pass" if observed is not None and observed > threshold else "below threshold"
        lines.append(
            f"| {label} | `> {threshold:.0f} tok/s` | `{fmt(observed)} tok/s` | {result} |"
        )

    lines.extend(["", "## All Measured Runs", ""])
    lines.append("| Run | Group | Draft tokens | Output tok/s | Mean TTFT ms | P99 TTFT ms | Mean TPOT ms | Acceptance rate | Acceptance length |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in sorted(rows, key=lambda item: (item["group"], item["num_speculative_tokens"] or -1, item["stem"])):
        acc = fmt(row["acceptance_rate"]) + "%" if row["acceptance_rate"] is not None else "N/A"
        lines.append(
            f"| `{row['stem']}` | `{row['group']}` | {fmt(row['num_speculative_tokens'], 0)} | "
            f"`{fmt(row['output_throughput'])}` | `{fmt(row['mean_ttft_ms'])}` | "
            f"`{fmt(row['p99_ttft_ms'])}` | `{fmt(row['mean_tpot_ms'])}` | "
            f"{acc} | `{fmt(row['acceptance_length'])}` |"
        )

    lines.extend(["", "## Notebook Text Blocks", ""])
    for group, title in [
        ("spec_bf16", "Speculative Decoding Benchmark Results"),
        ("fp8", "FP8 Quantization Benchmark Results"),
        ("fp8_spec", "FP8 + Speculative Decoding Benchmark Results"),
    ]:
        row = best.get(group)
        lines.append(f"### {title}")
        lines.append("")
        if row:
            lines.append("```text")
            lines.append(text_block(bench_dir, row["stem"]))
            lines.append("```")
        else:
            lines.append("No result found.")
        lines.append("")

    lines.extend(
        [
            "## Metrics Completeness",
            "",
            "- Step 9 table fields come from benchmark JSON files.",
            "- Step 10 benchmark text blocks come from matching TXT files.",
            "- Speculative acceptance fields are present when vLLM emits `spec_decode_*` metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-dir", type=Path, required=True)
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--results-md", type=Path, required=True)
    args = parser.parse_args()

    args.metrics_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.bench_dir)
    if not rows:
        raise SystemExit(f"No benchmark JSON files found in {args.bench_dir}")

    summary = {"best": best_by_group(rows), "all": rows}
    (args.metrics_dir / "task4_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    write_csv(args.metrics_dir / "task4_metrics.csv", rows)
    write_markdown(args.results_md, args.bench_dir, rows)
    print(f"Wrote {args.results_md}")
    print(f"Wrote {args.metrics_dir / 'task4_metrics.json'}")
    print(f"Wrote {args.metrics_dir / 'task4_metrics.csv'}")


if __name__ == "__main__":
    main()

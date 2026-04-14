"""memguard efficiency — GPU right-sizing report CLI.

Reads local telemetry from ``~/.memory-guard/telemetry.db`` and prints a
human-readable table showing the current GPU tier, recommended tier,
P94 peak, waste percentage, and estimated monthly savings for each
monitored source/model pair.

The report is built entirely from local telemetry accumulated by
``KVCacheMonitor``. No external service is required.

Usage
-----
  # Human-readable table (default)
  memguard-efficiency

  # Pipe-friendly JSON (for CI / alerting)
  memguard-efficiency --json

  # Fleet aggregate sorted by waste fraction
  memguard-efficiency --fleet

  # Change the lookback window (default: 30 days, max: 90)
  memguard-efficiency --lookback-days 7

  # Filter by source or model
  memguard-efficiency --source-id pod-a.i-1234abcd
  memguard-efficiency --model meta-llama/Llama-3-8B-Instruct

Exit codes
----------
  0  —  success (or no sources found)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "source_id":    28,
    "model_name":   36,
    "current_sku":  12,
    "recommended":  12,
    "p94_mb":        9,
    "waste":         7,
    "savings":      12,  # "  $10,000/mo" fits with padding
    "conf":          8,
    "n":             6,
}

_HEADER = (
    f"{'SOURCE':<{_COL_WIDTHS['source_id']}}"
    f"{'MODEL':<{_COL_WIDTHS['model_name']}}"
    f"{'CURRENT':<{_COL_WIDTHS['current_sku']}}"
    f"{'RECOMMENDS':<{_COL_WIDTHS['recommended']}}"
    f"{'P94 MB':>{_COL_WIDTHS['p94_mb']}}"
    f"{'WASTE':>{_COL_WIDTHS['waste']}}"
    f"{'SAVINGS/MO':>{_COL_WIDTHS['savings']}}"
    f"  {'CONF':<{_COL_WIDTHS['conf']}}"
    f"{'N':>{_COL_WIDTHS['n']}}"
)
_SEP = "─" * len(_HEADER)


def _truncate(s: str, width: int) -> str:
    return s if len(s) <= width else s[: width - 1] + "…"


def _format_source(row: Dict[str, Any]) -> str:
    source_id    = _truncate(str(row.get("source_id", "")),    _COL_WIDTHS["source_id"])
    model_name   = _truncate(str(row.get("model_name", "")),   _COL_WIDTHS["model_name"])
    # PR 73: render "4×A10G" for multi-GPU pods (device_count > 1)
    _sku         = str(row.get("current_sku") or "unknown")
    _dc          = int(row.get("device_count") or 1)
    current_sku  = f"{_dc}×{_sku}" if _dc > 1 else _sku
    current_sku  = _truncate(current_sku,                      _COL_WIDTHS["current_sku"])
    recommended  = str(row.get("recommended_sku") or "—")
    recommended  = _truncate(recommended,                      _COL_WIDTHS["recommended"])
    p94_mb       = f"{row.get('peak_p94_mb', 0):,.0f}"
    waste        = f"{row.get('waste_fraction', 0) * 100:.1f}%"
    savings_raw  = row.get("estimated_monthly_savings_usd", 0)
    savings      = f"${savings_raw}/mo" if savings_raw else "—"
    conf         = str(row.get("confidence", ""))[:3].upper()
    sample_n     = str(row.get("sample_size", 0))

    return (
        f"{source_id:<{_COL_WIDTHS['source_id']}}"
        f"{model_name:<{_COL_WIDTHS['model_name']}}"
        f"{current_sku:<{_COL_WIDTHS['current_sku']}}"
        f"{recommended:<{_COL_WIDTHS['recommended']}}"
        f"{p94_mb:>{_COL_WIDTHS['p94_mb']}}"
        f"{waste:>{_COL_WIDTHS['waste']}}"
        f"{savings:>{_COL_WIDTHS['savings']}}"
        f"  {conf:<{_COL_WIDTHS['conf']}}"  # 2-space separator before CONF
        f"{sample_n:>{_COL_WIDTHS['n']}}"
    )


def _print_table(sources: List[Dict[str, Any]], total_savings: Optional[int] = None) -> None:
    print(_SEP)
    print(_HEADER)
    print(_SEP)
    if not sources:
        print("  (no efficiency data found for the lookback window)")
    else:
        for row in sources:
            print(_format_source(row))
    print(_SEP)
    if total_savings is not None:
        print(f"  Total estimated monthly savings: ${total_savings}/mo")
        print(_SEP)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="memguard-efficiency",
        description="GPU right-sizing report — shows waste fraction and savings per source/model.",
    )
    parser.add_argument(
        "--fleet",
        action="store_true",
        help=(
            "Show all matching local sources sorted by waste fraction. "
            "Retained for CLI compatibility."
        ),
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="Emit raw JSON to stdout (pipe-friendly; compatible with CI / alerting scripts).",
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=30,
        metavar="N",
        help="Number of days of total_peak_mb history to include (default: 30, max: 90).",
    )
    parser.add_argument(
        "--source-id",
        default="",
        metavar="ID",
        help="Filter output to a single source_id (substring match on the client side).",
    )
    parser.add_argument(
        "--model",
        default="",
        metavar="NAME",
        help="Filter output to a single model_name (substring match on the client side).",
    )
    args = parser.parse_args()

    lookback = max(1, min(90, args.lookback_days))

    from ..local_efficiency import compute_local_efficiency_report

    result = compute_local_efficiency_report(
        lookback_days=lookback,
        source_id_filter=args.source_id,
        model_filter=args.model,
    )
    if result is None:
        print("No local telemetry yet — run guard_vllm for a few days, then retry.")
        sys.exit(0)

    sources: List[Dict[str, Any]] = result["sources"]
    total_savings: Optional[int] = result.get("total_estimated_monthly_savings_usd")

    if args.as_json:
        out: Dict[str, Any] = {"sources": sources}
        if total_savings is not None:
            out["total_estimated_monthly_savings_usd"] = total_savings
        print(json.dumps(out, indent=2))
    else:
        _print_table(sources, total_savings)


if __name__ == "__main__":
    main()

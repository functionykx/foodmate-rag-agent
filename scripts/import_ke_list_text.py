from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_parser import parse_ke_list_page


DEFAULT_INPUT = ROOT / "data" / "manual_ke" / "list_page.txt"
DEFAULT_OUTPUT = ROOT / "data" / "rental_candidates_cuhksz.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="把人工复制的贝壳列表页文本解析为租房候选CSV")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.exists():
        raise SystemExit(f"请先创建并粘贴列表页文本: {args.input}")

    records = parse_ke_list_page(args.input.read_text(encoding="utf-8"), datetime.now().astimezone())
    if not records:
        raise SystemExit("没有识别到房源。请确认文本中包含“整租·..._小区租房”这类列表标题。")
    frame = pd.DataFrame(records).drop_duplicates("candidate_id", keep="last")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(frame[["candidate_id", "title", "monthly_rent_min", "area_sqm_min", "tags"]].to_string(index=False))
    print(f"\n已解析 {len(frame)} 套候选房源: {args.output}")


if __name__ == "__main__":
    main()

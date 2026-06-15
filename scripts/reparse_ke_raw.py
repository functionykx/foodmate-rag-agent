from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_parser import RENTAL_COLUMNS, parse_ke_listing, valid_listing


DEFAULT_RAW = ROOT / "data" / "raw" / "ke_listings.jsonl"
DEFAULT_CSV = ROOT / "data" / "rental_listings_cuhksz.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="从本地贝壳原文JSONL重新生成结构化租房CSV")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--output", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not args.raw.exists():
        raise SystemExit(f"原始文件不存在: {args.raw}")

    records = []
    for line_number, line in enumerate(args.raw.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
            if not raw.get("raw_text"):
                continue
            fetched_at = datetime.fromisoformat(raw["fetched_at"])
            record = parse_ke_listing(
                raw["raw_text"],
                raw["url"],
                raw.get("card_text", ""),
                fetched_at,
            )
            if valid_listing(record):
                records.append(record)
            else:
                print(f"第 {line_number} 行仍缺少关键字段: {raw.get('url')}")
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"跳过第 {line_number} 行: {exc}")

    new_frame = pd.DataFrame(records, columns=RENTAL_COLUMNS)
    if args.output.exists():
        old_frame = pd.read_csv(args.output, encoding="utf-8")
        for column in RENTAL_COLUMNS:
            if column not in old_frame.columns:
                old_frame[column] = None
        new_frame = pd.concat([old_frame[RENTAL_COLUMNS], new_frame], ignore_index=True)
    new_frame["_key"] = new_frame["listing_id"].fillna(new_frame["source_url"])
    new_frame = new_frame.drop_duplicates("_key", keep="last").drop(columns="_key")
    new_frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"已输出 {len(new_frame)} 条房源: {args.output}")


if __name__ == "__main__":
    main()

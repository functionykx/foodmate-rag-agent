from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_parser import RENTAL_COLUMNS, parse_ke_listing, valid_listing


DEFAULT_MANIFEST = ROOT / "data" / "rental_top10_to_enrich.csv"
DEFAULT_OUTPUT = ROOT / "data" / "rental_listings_cuhksz.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="导入人工复制的Top10贝壳详情页文本")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = pd.read_csv(args.manifest, encoding="utf-8")
    records = []
    for _, row in manifest.iterrows():
        text_path = ROOT / str(row["detail_text_file"])
        if not text_path.exists():
            print(f"[缺失] {row['candidate_id']}: {text_path}")
            continue
        source_url = str(row.get("detail_url", ""))
        if source_url == "nan":
            source_url = ""
        record = parse_ke_listing(
            text_path.read_text(encoding="utf-8"),
            source_url,
            str(row.get("title", "")),
            datetime.now().astimezone(),
        )
        if valid_listing(record):
            records.append(record)
            print(f"[成功] {record['listing_id']} {record['title']}")
        else:
            print(f"[解析失败] {text_path}：缺少验真编号、标题或月租")

    if not records:
        raise SystemExit("没有可导入的完整详情记录。")
    new_frame = pd.DataFrame(records, columns=RENTAL_COLUMNS)
    if args.output.exists():
        old = pd.read_csv(args.output, encoding="utf-8")
        for column in RENTAL_COLUMNS:
            if column not in old.columns:
                old[column] = None
        new_frame = pd.concat([old[RENTAL_COLUMNS], new_frame], ignore_index=True)
    new_frame["_key"] = new_frame["listing_id"].fillna(new_frame["source_url"])
    new_frame = new_frame.drop_duplicates("_key", keep="last").drop(columns="_key")
    new_frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"\n正式租房知识库共 {len(new_frame)} 条: {args.output}")


if __name__ == "__main__":
    main()

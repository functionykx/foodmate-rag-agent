from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


STANDARD_COLUMNS = [
    "id",
    "name",
    "cuisine",
    "price_per_person",
    "rating",
    "distance_km",
    "location",
    "opening_hours",
    "tags",
    "best_for",
    "menu_highlights",
    "review_summary",
    "pros",
    "cons",
    "source",
]


COLUMN_ALIASES = {
    "name": ["店名", "商户名", "餐厅名", "name"],
    "cuisine": ["菜系", "分类", "品类", "cuisine"],
    "price_per_person": ["人均", "人均价格", "人均消费", "price_per_person"],
    "rating": ["评分", "星级", "rating"],
    "distance_km": ["距离公里", "距离", "distance_km"],
    "location": ["位置", "地址", "商圈", "location"],
    "opening_hours": ["营业时间", "opening_hours"],
    "tags": ["标签", "特色", "tags"],
    "best_for": ["适合场景", "场景", "best_for"],
    "menu_highlights": ["推荐菜", "招牌菜", "菜单", "menu_highlights"],
    "review_summary": ["评论摘要", "评价摘要", "review_summary"],
    "pros": ["优点", "pros"],
    "cons": ["缺点", "cons"],
    "source": ["来源", "source"],
}


def find_column(df: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in df.columns:
            return name
    return None


def clean_number(value, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else default


def normalize_dianping_csv(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    raw = pd.read_csv(input_path, encoding="utf-8")
    normalized = pd.DataFrame()
    normalized["id"] = range(1, len(raw) + 1)

    for standard_col, aliases in COLUMN_ALIASES.items():
        source_col = find_column(raw, aliases)
        if source_col:
            normalized[standard_col] = raw[source_col]
        else:
            normalized[standard_col] = ""

    normalized["name"] = normalized["name"].fillna("未知餐厅").astype(str)
    normalized["cuisine"] = normalized["cuisine"].replace("", "本地餐厅").fillna("本地餐厅")
    normalized["price_per_person"] = normalized["price_per_person"].apply(lambda x: clean_number(x, 50))
    normalized["rating"] = normalized["rating"].apply(lambda x: clean_number(x, 4.0))
    normalized["distance_km"] = normalized["distance_km"].apply(lambda x: clean_number(x, 1.0))
    normalized["location"] = normalized["location"].replace("", "学校附近").fillna("学校附近")
    normalized["opening_hours"] = normalized["opening_hours"].replace("", "未知").fillna("未知")
    normalized["source"] = normalized["source"].replace("", "manual_dianping").fillna("manual_dianping")

    for col in ["tags", "best_for", "menu_highlights", "review_summary", "pros", "cons"]:
        normalized[col] = normalized[col].fillna("").astype(str)

    normalized = normalized[STANDARD_COLUMNS]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, encoding="utf-8")
    return normalized


def parse_args():
    parser = argparse.ArgumentParser(description="Normalize manually collected Dianping-like restaurant CSV into FoodMate schema.")
    parser.add_argument("--input", required=True, help="Input CSV exported or manually prepared from allowed data.")
    parser.add_argument("--output", default="data/restaurants_real.csv", help="Output FoodMate-compatible CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = normalize_dianping_csv(args.input, args.output)
    print(f"Saved {len(df)} restaurants to {args.output}")


if __name__ == "__main__":
    main()

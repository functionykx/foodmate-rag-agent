from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "rental_candidates_cuhksz.csv"
DEFAULT_OUTPUT = ROOT / "data" / "rental_top10_to_enrich.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="从列表页候选中筛选需要人工打开详情的Top10")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--budget", type=float, required=True, help="月租上限")
    parser.add_argument("--bedrooms", type=int, default=None, help="期望卧室数")
    parser.add_argument("--rental-type", default="整租", choices=["整租", "合租", "不限"])
    parser.add_argument("--keywords", default="近地铁,精装", help="偏好关键词，用逗号分隔")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--include-ads", action="store_true")
    args = parser.parse_args()

    frame = pd.read_csv(args.input, encoding="utf-8")
    if not args.include_ads:
        frame = frame[frame["is_advertisement"].fillna(0).astype(int) == 0]
    if args.rental_type != "不限":
        frame = frame[frame["rental_type"] == args.rental_type]

    rent = frame["monthly_rent_min"].astype(float)
    frame["budget_score"] = (1 - (rent - args.budget).abs() / max(args.budget, 1)).clip(0, 1)
    frame.loc[rent > args.budget, "budget_score"] *= 0.35
    if args.bedrooms is None:
        frame["bedroom_score"] = 0.5
    else:
        frame["bedroom_score"] = (frame["bedrooms"].fillna(-99).astype(int) == args.bedrooms).astype(float)
    keywords = [item.strip() for item in args.keywords.split(",") if item.strip()]
    frame["tag_score"] = frame["tags"].fillna("").apply(
        lambda text: sum(keyword in text for keyword in keywords) / max(len(keywords), 1)
    )
    frame["freshness_score"] = pd.to_datetime(frame["maintenance_date"], errors="coerce").rank(pct=True).fillna(0.0)
    frame["preselect_score"] = (
        0.50 * frame["budget_score"]
        + 0.25 * frame["bedroom_score"]
        + 0.15 * frame["tag_score"]
        + 0.10 * frame["freshness_score"]
    )
    top = frame.sort_values("preselect_score", ascending=False).head(args.top_k).copy()
    top["detail_url"] = ""
    top["detail_text_file"] = top["candidate_id"].apply(lambda value: f"data/manual_ke/details/{value}.txt")
    top.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(top[["candidate_id", "title", "monthly_rent_min", "bedrooms", "preselect_score", "detail_text_file"]].to_string(index=False))
    print(f"\nTop{len(top)} 待补详情清单: {args.output}")


if __name__ == "__main__":
    main()

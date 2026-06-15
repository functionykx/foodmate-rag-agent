from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GEO_DATA_PATH = PROJECT_ROOT / "data" / "restaurants_cuhksz_geo.csv"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "restaurants_cuhksz.csv"


def resolve_data_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    env_path = os.getenv("FOODMATE_DATA_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_GEO_DATA_PATH if DEFAULT_GEO_DATA_PATH.exists() else DEFAULT_DATA_PATH


def load_restaurants(path: str | Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(resolve_data_path(path), encoding="utf-8")
    df["document"] = df.apply(build_document, axis=1)
    return df


def build_document(row: pd.Series) -> str:
    return (
        f"餐厅：{row['name']}。"
        f"菜系：{row['cuisine']}。"
        f"人均价格：{row['price_per_person']}元。"
        f"评分：{row['rating']}。"
        f"距离：{row['distance_km']}公里。"
        f"位置：{row['location']}。"
        f"标签：{row['tags']}。"
        f"适合场景：{row['best_for']}。"
        f"招牌菜：{row['menu_highlights']}。"
        f"评论摘要：{row['review_summary']}。"
        f"优点：{row['pros']}。"
        f"缺点：{row['cons']}。"
    )


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def contains_any(text: str, keywords: list[str]) -> bool:
    text = normalize_text(text)
    return any(keyword in text for keyword in keywords)

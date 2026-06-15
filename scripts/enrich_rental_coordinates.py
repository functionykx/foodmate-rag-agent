from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tools.baidu_map import BaiduMapError, BaiduMapTool


DEFAULT_INPUT = ROOT / "data" / "rental_listings_cuhksz.csv"
DEFAULT_OUTPUT = ROOT / "data" / "rental_listings_cuhksz_geo.csv"
BOUNDS = (22.60, 22.78, 114.09, 114.30)
GEO_COLUMNS = [
    "latitude", "longitude", "coordinate_type", "baidu_uid", "coordinate_source",
    "coordinate_query", "geocode_status", "geocode_error", "coordinate_updated_at",
]


def _in_bounds(point: dict) -> bool:
    return BOUNDS[0] <= float(point["latitude"]) <= BOUNDS[1] and BOUNDS[2] <= float(point["longitude"]) <= BOUNDS[3]


def _checkpoint(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, output)


def _resolve_community(tool: BaiduMapTool, community: str) -> tuple[dict, str]:
    queries = [f"深圳市龙岗区{community}", f"龙岗大运 {community}", community]
    errors = []
    for query in queries:
        try:
            point = tool.search_place(query, region="深圳市")
            if _in_bounds(point):
                return point, f"place:{query}"
            errors.append(f"{query}命中范围外坐标")
        except BaiduMapError as exc:
            errors.append(str(exc))
    try:
        address = f"深圳市龙岗区{community}"
        point = tool.geocode(address, city="深圳市")
        if _in_bounds(point):
            return point, f"geocode:{address}"
        errors.append(f"{address}地理编码范围异常")
    except BaiduMapError as exc:
        errors.append(str(exc))
    raise BaiduMapError("; ".join(errors))


def enrich(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    delay: float = 0.25,
    resume: bool = True,
    retry_failed: bool = False,
    limit: int | None = None,
    tool: BaiduMapTool | None = None,
) -> pd.DataFrame:
    tool = tool or BaiduMapTool()
    if not tool.available():
        raise SystemExit("缺少 BAIDU_MAP_AK，请先在PowerShell中设置。")
    source = output_path if resume and output_path.exists() else input_path
    frame = pd.read_csv(source, encoding="utf-8-sig")
    for column in GEO_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA

    communities = frame["community"].fillna("").astype(str).str.strip()
    unique_communities = [item for item in communities.unique().tolist() if item]
    processed = success = failed = skipped = 0
    for community in unique_communities:
        mask = communities == community
        existing = frame.loc[mask, ["latitude", "longitude"]].dropna()
        if not existing.empty:
            skipped += 1
            continue
        statuses = set(frame.loc[mask, "geocode_status"].dropna().astype(str))
        if "failed" in statuses and not retry_failed:
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        try:
            point, query = _resolve_community(tool, community)
            updates = {
                "latitude": float(point["latitude"]),
                "longitude": float(point["longitude"]),
                "coordinate_type": point.get("coordinate_type", "bd09ll"),
                "baidu_uid": point.get("baidu_uid", ""),
                "coordinate_source": point.get("source", "baidu"),
                "coordinate_query": query,
                "geocode_status": "success",
                "geocode_error": "",
                "coordinate_updated_at": timestamp,
            }
            for column, value in updates.items():
                frame.loc[mask, column] = value
            success += 1
            print(f"[{processed}] 成功 {community}: {point['latitude']},{point['longitude']}，回填{int(mask.sum())}套")
        except BaiduMapError as exc:
            frame.loc[mask, "geocode_status"] = "failed"
            frame.loc[mask, "geocode_error"] = str(exc)[:500]
            frame.loc[mask, "coordinate_updated_at"] = timestamp
            failed += 1
            print(f"[{processed}] 失败 {community}: {exc}")
        _checkpoint(frame, output_path)
        if delay:
            time.sleep(delay)

    _checkpoint(frame, output_path)
    print(f"\n唯一小区: {len(unique_communities)}，本轮成功: {success}，失败: {failed}，跳过: {skipped}")
    print(f"房源坐标完整: {frame['latitude'].notna().sum()}/{len(frame)}")
    print(f"输出文件: {output_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="按唯一小区补齐租房坐标")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None, help="只处理前N个唯一小区")
    parser.add_argument("--delay", type=float, default=0.25)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    enrich(args.input, args.output, max(args.delay, 0), not args.no_resume, args.retry_failed, args.limit)


if __name__ == "__main__":
    main()

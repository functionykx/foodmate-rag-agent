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


DEFAULT_INPUT = ROOT / "data" / "restaurants_cuhksz.csv"
DEFAULT_OUTPUT = ROOT / "data" / "restaurants_cuhksz_geo.csv"
COORDINATE_COLUMNS = [
    "latitude",
    "longitude",
    "coordinate_type",
    "baidu_uid",
    "coordinate_source",
    "coordinate_query",
    "geocode_status",
    "geocode_error",
    "coordinate_updated_at",
]
DEFAULT_BOUNDS = (22.60, 22.78, 114.09, 114.30)
COORDINATE_OVERRIDES = {
    # Place search can hit a similarly named POI far from the supplied address.
    "百草膳香鸡煲(坳背店)": {
        "latitude": 22.673868546960396,
        "longitude": 114.22229087052659,
        "coordinate_type": "bd09ll",
        "baidu_uid": "",
        "coordinate_source": "baidu_geocoding_verified",
        "coordinate_query": "geocode_address:深圳市龙岗区保安社区坳背路105号102",
    },
}


def _load_frame(input_path: Path, output_path: Path, resume: bool) -> pd.DataFrame:
    input_frame = pd.read_csv(input_path, encoding="utf-8-sig")
    if resume and output_path.exists():
        frame = pd.read_csv(output_path, encoding="utf-8-sig")
        existing_names = set(frame["name"].fillna("").astype(str))
        new_rows = input_frame[~input_frame["name"].fillna("").astype(str).isin(existing_names)].copy()
        if not new_rows.empty:
            frame = pd.concat([frame, new_rows], ignore_index=True, sort=False)
            print(f"从基础知识库同步新增餐厅: {len(new_rows)} 家")
    else:
        frame = input_frame
    for column in COORDINATE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _valid_coordinates(row: pd.Series) -> bool:
    try:
        latitude = float(row.get("latitude"))
        longitude = float(row.get("longitude"))
    except (TypeError, ValueError):
        return False
    return 3.0 <= latitude <= 54.0 and 73.0 <= longitude <= 136.0


def _within_bounds(point: dict, bounds: tuple[float, float, float, float]) -> bool:
    min_lat, max_lat, min_lng, max_lng = bounds
    latitude = float(point["latitude"])
    longitude = float(point["longitude"])
    return min_lat <= latitude <= max_lat and min_lng <= longitude <= max_lng


def _resolve_restaurant(
    tool: BaiduMapTool,
    name: str,
    address: str,
    bounds: tuple[float, float, float, float] = DEFAULT_BOUNDS,
) -> tuple[dict, str]:
    """Resolve a restaurant with increasingly broad queries."""
    queries = []
    if name and address:
        queries.append(("place_name_address", f"{name} {address}"))
    if name:
        queries.append(("place_name", name))
    errors = []
    for source, query in queries:
        try:
            point = tool.search_place(query, region="深圳市")
            if _within_bounds(point, bounds):
                return point, f"{source}:{query}"
            errors.append(
                f"{query} 命中范围外坐标 {point['latitude']},{point['longitude']}"
            )
        except BaiduMapError as exc:
            errors.append(str(exc))

    if address:
        full_address = address if "深圳" in address else f"深圳市龙岗区{address}"
        try:
            point = tool.geocode(full_address, city="深圳市")
            if _within_bounds(point, bounds):
                return point, f"geocode_address:{full_address}"
            errors.append(
                f"{full_address} 地理编码仍在范围外 {point['latitude']},{point['longitude']}"
            )
        except BaiduMapError as exc:
            errors.append(str(exc))
    raise BaiduMapError("; ".join(errors) or "缺少可查询的店名和地址")


def _checkpoint(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    os.replace(temporary, output_path)


def enrich(
    input_path: Path,
    output_path: Path,
    limit: int | None = None,
    delay: float = 0.25,
    resume: bool = True,
    retry_failed: bool = False,
    refresh_outliers: bool = False,
    tool: BaiduMapTool | None = None,
) -> pd.DataFrame:
    tool = tool or BaiduMapTool()
    if not tool.available():
        raise SystemExit("缺少 BAIDU_MAP_AK，请先在 PowerShell 中设置环境变量。")
    if not input_path.exists():
        raise SystemExit(f"输入文件不存在: {input_path}")

    frame = _load_frame(input_path, output_path, resume=resume)
    processed = 0
    success = 0
    failed = 0
    skipped = 0

    for idx, row in frame.iterrows():
        name = str(row.get("name", "")).strip()
        override = COORDINATE_OVERRIDES.get(name)
        if override:
            for column, value in override.items():
                frame.at[idx, column] = value
            frame.at[idx, "geocode_status"] = "success"
            frame.at[idx, "geocode_error"] = ""
            frame.at[idx, "coordinate_updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

        existing_point = {"latitude": row.get("latitude"), "longitude": row.get("longitude")}
        is_outlier = _valid_coordinates(row) and not _within_bounds(existing_point, DEFAULT_BOUNDS)
        if _valid_coordinates(row) and not (refresh_outliers and is_outlier):
            skipped += 1
            continue
        if not retry_failed and str(row.get("geocode_status", "")) == "failed":
            skipped += 1
            continue
        if limit is not None and processed >= limit:
            break

        address = str(row.get("location", "")).strip()
        if address.lower() == "nan":
            address = ""
        processed += 1
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")

        try:
            point, query = _resolve_restaurant(tool, name, address)
            frame.at[idx, "latitude"] = float(point["latitude"])
            frame.at[idx, "longitude"] = float(point["longitude"])
            frame.at[idx, "coordinate_type"] = point.get("coordinate_type", "bd09ll")
            frame.at[idx, "baidu_uid"] = point.get("baidu_uid", "")
            frame.at[idx, "coordinate_source"] = point.get("source", "baidu")
            frame.at[idx, "coordinate_query"] = query
            frame.at[idx, "geocode_status"] = "success"
            frame.at[idx, "geocode_error"] = ""
            frame.at[idx, "coordinate_updated_at"] = timestamp
            success += 1
            print(
                f"[{processed}] 成功 {name}: "
                f"{point['latitude']},{point['longitude']} ({query.split(':', 1)[0]})"
            )
        except (BaiduMapError, KeyError, TypeError, ValueError) as exc:
            frame.at[idx, "geocode_status"] = "failed"
            frame.at[idx, "geocode_error"] = str(exc)[:500]
            frame.at[idx, "coordinate_updated_at"] = timestamp
            failed += 1
            print(f"[{processed}] 失败 {name}: {exc}")

        # Save every row so an interrupted run can resume without losing API results.
        _checkpoint(frame, output_path)
        if delay > 0:
            time.sleep(delay)

    _checkpoint(frame, output_path)
    coordinate_count = int(frame.apply(_valid_coordinates, axis=1).sum())
    print("\n坐标补全完成")
    print(f"输出文件: {output_path}")
    print(f"本轮成功: {success}，失败: {failed}，跳过: {skipped}")
    print(f"坐标完整: {coordinate_count}/{len(frame)}")
    if failed:
        print("失败记录已写入 geocode_status/geocode_error，可使用 --retry-failed 重试。")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="使用百度地图AK批量补齐餐厅坐标")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="原始餐厅CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="带坐标的输出CSV")
    parser.add_argument("--limit", type=int, default=None, help="只处理前N条缺失记录，用于测试")
    parser.add_argument("--delay", type=float, default=0.25, help="每家餐厅请求后的等待秒数")
    parser.add_argument("--no-resume", action="store_true", help="忽略已有输出，从输入文件重新开始")
    parser.add_argument("--retry-failed", action="store_true", help="重新查询上次失败的记录")
    parser.add_argument("--refresh-outliers", action="store_true", help="重新查询大运/龙岗范围外的可疑坐标")
    args = parser.parse_args()
    enrich(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
        delay=max(args.delay, 0.0),
        resume=not args.no_resume,
        retry_failed=args.retry_failed,
        refresh_outliers=args.refresh_outliers,
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils import PROJECT_ROOT


GEOCODING_URL = "https://api.map.baidu.com/geocoding/v3/"
PLACE_SEARCH_URL = "https://api.map.baidu.com/place/v2/search"
ROUTE_MATRIX_URLS = {
    "walking": "https://api.map.baidu.com/routematrix/v2/walking",
    "riding": "https://api.map.baidu.com/routematrix/v2/riding",
    "driving": "https://api.map.baidu.com/routematrix/v2/driving",
}
TRANSPORT_LABELS = {"walking": "步行", "riding": "骑行", "driving": "驾车"}
DEFAULT_CACHE_PATH = PROJECT_ROOT / "vector_store" / "baidu_geocode_cache.json"


class BaiduMapError(RuntimeError):
    """Raised when Baidu Map returns an unsuccessful response."""


class BaiduMapTool:
    def __init__(self, ak: str | None = None, cache_path: str | Path | None = None):
        self.ak = ak or os.getenv("BAIDU_MAP_AK", "")
        self.city = os.getenv("FOODMATE_MAP_CITY", "深圳市")
        self.timeout = float(os.getenv("FOODMATE_MAP_TIMEOUT", "8"))
        self.batch_size = max(1, int(os.getenv("FOODMATE_MAP_BATCH_SIZE", "10")))
        self.cache_path = Path(cache_path or DEFAULT_CACHE_PATH)
        self._cache = self._load_cache()

    def available(self) -> bool:
        return bool(self.ak)

    def _request(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        if not self.ak:
            raise BaiduMapError("缺少 BAIDU_MAP_AK")
        query = dict(params)
        query["ak"] = self.ak
        request_url = f"{url}?{urllib.parse.urlencode(query)}"
        try:
            with urllib.request.urlopen(request_url, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise BaiduMapError(f"百度地图请求失败: {exc}") from exc
        if payload.get("status") != 0:
            message = payload.get("message") or payload.get("msg") or "unknown error"
            raise BaiduMapError(f"百度地图返回错误 status={payload.get('status')}: {message}")
        return payload

    def _load_cache(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def geocode(self, address: str, city: str | None = None, use_cache: bool = True) -> dict[str, Any]:
        address = str(address).strip()
        city = city or self.city
        cache_key = f"geocode::{city}::{address}"
        if use_cache and cache_key in self._cache:
            return dict(self._cache[cache_key])

        payload = self._request(
            GEOCODING_URL,
            {"address": address, "city": city, "output": "json", "ret_coordtype": "bd09ll"},
        )
        result = payload["result"]
        location = result["location"]
        point = {
            "latitude": float(location["lat"]),
            "longitude": float(location["lng"]),
            "coordinate_type": "bd09ll",
            "precise": int(result.get("precise", 0)),
            "confidence": int(result.get("confidence", 0)),
            "level": str(result.get("level", "")),
            "source": "baidu_geocoding",
        }
        self._cache[cache_key] = point
        self._save_cache()
        return dict(point)

    def search_place(self, name: str, region: str | None = None, use_cache: bool = True) -> dict[str, Any]:
        name = str(name).strip()
        region = region or self.city
        cache_key = f"place::{region}::{name}"
        if use_cache and cache_key in self._cache:
            return dict(self._cache[cache_key])

        payload = self._request(
            PLACE_SEARCH_URL,
            {
                "query": name,
                "region": region,
                "city_limit": "true",
                "output": "json",
                "scope": 1,
                "page_size": 10,
                "page_num": 0,
                "coord_type": 3,
                "ret_coordtype": "bd09ll",
            },
        )
        results = payload.get("results", [])
        if not results:
            raise BaiduMapError(f"未找到地点: {name}")
        poi = results[0]
        location = poi.get("location") or {}
        point = {
            "name": poi.get("name", name),
            "address": poi.get("address", ""),
            "latitude": float(location["lat"]),
            "longitude": float(location["lng"]),
            "coordinate_type": "bd09ll",
            "baidu_uid": poi.get("uid", ""),
            "source": "baidu_place_search",
        }
        self._cache[cache_key] = point
        self._save_cache()
        return dict(point)

    def resolve_location(self, text: str, city: str | None = None) -> dict[str, Any]:
        """Resolve either a POI name or a structured address to BD-09 coordinates."""
        try:
            return self.search_place(text, region=city)
        except BaiduMapError:
            return self.geocode(text, city=city)

    def route_matrix(
        self,
        origin: tuple[float, float],
        destinations: list[tuple[float, float]],
        transport_mode: str = "walking",
    ) -> list[dict[str, float]]:
        if transport_mode not in ROUTE_MATRIX_URLS:
            raise BaiduMapError(f"不支持的出行方式: {transport_mode}")
        if not destinations:
            return []
        all_routes: list[dict[str, float]] = []
        origin_text = f"{origin[0]},{origin[1]}"
        for start in range(0, len(destinations), self.batch_size):
            batch = destinations[start : start + self.batch_size]
            destination_text = "|".join(f"{lat},{lng}" for lat, lng in batch)
            params = {
                "origins": origin_text,
                "destinations": destination_text,
                "output": "json",
                "coord_type": "bd09ll",
            }
            if transport_mode == "driving":
                params["tactics"] = 11
            payload = self._request(ROUTE_MATRIX_URLS[transport_mode], params)
            results = payload.get("result", [])
            if len(results) != len(batch):
                raise BaiduMapError(
                    f"百度{TRANSPORT_LABELS[transport_mode]}算路返回数量与目的地数量不一致"
                )
            for item in results:
                distance = item.get("distance") or {}
                duration = item.get("duration") or {}
                all_routes.append(
                    {
                        "distance_km": round(float(distance.get("value", 0)) / 1000, 3),
                        "duration_min": round(float(duration.get("value", 0)) / 60, 1),
                    }
                )
            if start + self.batch_size < len(destinations):
                time.sleep(float(os.getenv("FOODMATE_MAP_BATCH_DELAY", "0.05")))
        return all_routes

    def walking_matrix(
        self,
        origin: tuple[float, float],
        destinations: list[tuple[float, float]],
    ) -> list[dict[str, float]]:
        """Backward-compatible wrapper for the original walking-only API."""
        return self.route_matrix(origin, destinations, transport_mode="walking")

    def route_many_origins_to_destination(
        self,
        origins: list[tuple[float, float]],
        destination: tuple[float, float],
        transport_mode: str = "walking",
    ) -> list[dict[str, float]]:
        """Calculate multiple house-to-work routes in one matrix request per batch."""
        if transport_mode not in ROUTE_MATRIX_URLS:
            raise BaiduMapError(f"不支持的出行方式: {transport_mode}")
        if not origins:
            return []
        all_routes: list[dict[str, float]] = []
        destination_text = f"{destination[0]},{destination[1]}"
        for start in range(0, len(origins), self.batch_size):
            batch = origins[start : start + self.batch_size]
            origin_text = "|".join(f"{lat},{lng}" for lat, lng in batch)
            params = {
                "origins": origin_text,
                "destinations": destination_text,
                "output": "json",
                "coord_type": "bd09ll",
            }
            if transport_mode == "driving":
                params["tactics"] = 11
            payload = self._request(ROUTE_MATRIX_URLS[transport_mode], params)
            results = payload.get("result", [])
            if len(results) != len(batch):
                raise BaiduMapError(
                    f"百度{TRANSPORT_LABELS[transport_mode]}算路返回数量与起点数量不一致"
                )
            for item in results:
                distance = item.get("distance") or {}
                duration = item.get("duration") or {}
                all_routes.append(
                    {
                        "distance_km": round(float(distance.get("value", 0)) / 1000, 3),
                        "duration_min": round(float(duration.get("value", 0)) / 60, 1),
                    }
                )
            if start + self.batch_size < len(origins):
                time.sleep(float(os.getenv("FOODMATE_MAP_BATCH_DELAY", "0.05")))
        return all_routes

    def update_candidate_distances(
        self,
        user_location: str,
        candidates: pd.DataFrame,
        transport_mode: str = "walking",
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        if transport_mode not in ROUTE_MATRIX_URLS:
            transport_mode = "walking"
        result = candidates.copy()
        result["effective_distance_km"] = result["distance_km"].astype(float)
        result["distance_source"] = "school_default"
        result["estimated_duration_min"] = pd.Series([None] * len(result), index=result.index, dtype="object")
        result["transport_mode"] = transport_mode

        origin = self.resolve_location(user_location)
        valid_rows: list[int] = []
        destinations: list[tuple[float, float]] = []
        for idx, row in result.iterrows():
            try:
                latitude = float(row.get("latitude"))
                longitude = float(row.get("longitude"))
            except (TypeError, ValueError):
                continue
            if pd.isna(latitude) or pd.isna(longitude):
                continue
            valid_rows.append(idx)
            destinations.append((latitude, longitude))

        if destinations:
            routes = self.route_matrix(
                (float(origin["latitude"]), float(origin["longitude"])),
                destinations,
                transport_mode=transport_mode,
            )
            for idx, route in zip(valid_rows, routes):
                result.at[idx, "effective_distance_km"] = route["distance_km"]
                result.at[idx, "estimated_duration_min"] = route["duration_min"]
                result.at[idx, "distance_source"] = f"baidu_{transport_mode}"

        context = {
            "enabled": True,
            "user_location": user_location,
            "transport_mode": transport_mode,
            "transport_label": TRANSPORT_LABELS[transport_mode],
            "origin": origin,
            "routed_candidates": len(destinations),
            "fallback_candidates": len(result) - len(destinations),
        }
        return result, context

from __future__ import annotations

import re
import hashlib
from datetime import date, datetime, timedelta
from typing import Any


RENTAL_COLUMNS = [
    "listing_id", "title", "community", "city", "district", "monthly_rent",
    "payment_method", "deposit", "service_fee", "agency_fee", "rental_type",
    "bedrooms", "living_rooms", "bathrooms", "area_sqm", "decoration",
    "orientation", "floor_level", "total_floors", "elevator", "parking",
    "water_type", "electricity_type", "gas", "heating", "broadband",
    "move_in_date", "lease_term", "viewing", "facilities", "tags",
    "verification_status", "publisher_compliance", "maintenance_date",
    "availability_status", "latitude", "longitude", "coordinate_type",
    "source", "source_url",
]


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", str(text).replace("\r", "")).strip()


def _search(pattern: str, text: str, flags: int = 0) -> str | None:
    match = re.search(pattern, text, flags)
    return _clean(match.group(1)) if match else None


def _number(value: str | None, kind=float) -> int | float | None:
    if value is None:
        return None
    try:
        parsed = kind(value)
        return parsed
    except (TypeError, ValueError):
        return None


def _yes_no(text: str, label: str) -> str:
    value = _search(rf"{re.escape(label)}[：:]?\s*([^\n]+)", text)
    if not value:
        return "暂无数据"
    if value.startswith("有") or value.startswith("是"):
        return "是"
    if value.startswith("无") or value.startswith("否"):
        return "否"
    return value[:20]


def _maintenance_date(text: str, fetched_at: datetime) -> str | None:
    explicit = _search(r"房源维护时间[：:]?\s*(\d{4}-\d{2}-\d{2})", text)
    if explicit:
        return explicit
    if re.search(r"(?:维护[：:]?|房源维护时间[：:]?)\s*今天", text):
        return fetched_at.date().isoformat()
    days = _search(r"(\d+)\s*天前维护", text)
    if days:
        return (fetched_at.date() - timedelta(days=int(days))).isoformat()
    return None


def _extract_title(text: str, card_text: str = "") -> str | None:
    candidates = []
    for source in [text, card_text]:
        for line in source.splitlines():
            line = _clean(line)
            if re.match(r"^(?:整租|合租|独栋)[·・]", line) and "_" not in line:
                candidates.append(line)
    if not candidates:
        return None
    return min(candidates, key=len)


def _extract_community(title: str | None, card_text: str) -> str | None:
    community = _search(r"龙岗区[-－]大运新城[-－]([^/\n]+)", card_text)
    if community:
        return community
    if title:
        match = re.search(r"[·・]([^\d]+?)\s+\d室", title)
        if match:
            return _clean(match.group(1))
    return None


def _extract_fees(text: str, rent: int | None) -> dict[str, Any]:
    fees = {"payment_method": None, "deposit": None, "service_fee": None, "agency_fee": None}
    match = re.search(
        r"付款方式.*?租金\s*\(元/月\).*?押金\s*\(元\).*?服务费\s*\([^)]*\).*?中介费\s*\(元\)\s*"
        r"([^\n]+)\s*\n\s*([^\n]+)\s*\n\s*([^\n]+)\s*\n\s*([^\n]+)\s*\n\s*([^\n]+)",
        text,
        re.S,
    )
    if not match:
        return fees
    payment, listed_rent, deposit, service_fee, agency_fee = [_clean(match.group(i)) for i in range(1, 6)]
    fees["payment_method"] = payment
    fees["deposit"] = _number(deposit, int) if deposit.isdigit() else deposit
    fees["service_fee"] = _number(service_fee, int) if service_fee.isdigit() else service_fee
    fees["agency_fee"] = _number(agency_fee, int) if agency_fee.isdigit() else agency_fee
    if rent is None and listed_rent.isdigit():
        fees["monthly_rent"] = int(listed_rent)
    return fees


def parse_ke_listing(
    raw_text: str,
    source_url: str,
    card_text: str = "",
    fetched_at: datetime | None = None,
) -> dict[str, Any]:
    """Parse visible Ke listing text into the project's rental knowledge schema."""
    fetched_at = fetched_at or datetime.now().astimezone()
    text = _clean(raw_text)
    card_text = _clean(card_text)
    title = _extract_title(text, card_text)
    listing_id = _search(r"房源验真编号[：:]?\s*(SZ\d+)", text)
    monthly_rent = _number(_search(r"(\d+)\s*元/月", text), int)

    house = re.search(r"房屋类型[：:]?\s*(\d+)室(\d+)厅(\d+)卫\s*([\d.]+)㎡\s*([^\n]+)", text)
    if not house:
        house = re.search(r"(\d+)室(\d+)厅(\d+)卫.*?([\d.]+)㎡", f"{text}\n{card_text}")
    bedrooms = int(house.group(1)) if house else None
    living_rooms = int(house.group(2)) if house else None
    bathrooms = int(house.group(3)) if house else None
    area_sqm = float(house.group(4)) if house else _number(_search(r"([\d.]+)㎡", text))
    decoration = _clean(house.group(5)).split()[0] if house and house.lastindex and house.lastindex >= 5 else None

    orientation = _search(r"(?:基本信息)?面积[：:].*?朝向[：:]\s*([^\n]+?)\s+维护[：:]", text, re.S)
    if not orientation:
        orientation = _search(r"朝向楼层[：:]?\s*([^\n]+?)\s+(?:低|中|高)楼层", text)
    floor_level = _search(r"楼层[：:]?\s*((?:低|中|高)楼层)/\d+层", text)
    total_floors = _number(_search(r"楼层[：:]?\s*(?:低|中|高)楼层/(\d+)层", text), int)

    rental_type = _search(r"租赁方式[：:]?\s*([^\n]+)", text)
    move_in = _search(r"入住[：:]?\s*([^\n]+?)\s+楼层[：:]", text)
    lease_term = _search(r"租期[：:]?\s*([^\n]+)", text)
    viewing = _search(r"看房[：:]?\s*([^\n]+)", text)

    facility_names = ["洗衣机", "空调", "衣柜", "电视", "冰箱", "热水器", "床", "暖气", "宽带", "天然气"]
    facilities = []
    facility_block = _search(r"配套设施\s*(.*?)(?:房源描述|费用详情|付款方式|$)", text, re.S) or ""
    for name in facility_names:
        if re.search(rf"{name}(?!\s*无)", facility_block):
            facilities.append(name)

    tags_source = f"{text}\n{card_text}"
    known_tags = [
        "贝壳省心租", "贝壳优选", "必看好房", "自营", "月租", "近地铁", "精装",
        "押一付一", "随时看房", "业主自荐", "有阳台", "开放厨房", "拎包入住",
    ]
    tags = [tag for tag in known_tags if tag in tags_source]

    if "房源发布人未登记" in text or "未合规备案" in text:
        compliance = "发布人未登记且机构未合规备案"
    elif "营业执照" in text:
        compliance = "营业执照已展示"
    else:
        compliance = "暂无数据"

    fees = _extract_fees(text, monthly_rent)
    monthly_rent = monthly_rent or fees.pop("monthly_rent", None)
    record = {
        "listing_id": listing_id,
        "title": title,
        "community": _extract_community(title, card_text),
        "city": "深圳市",
        "district": "龙岗区" if "龙岗区" in tags_source or "大运" in tags_source else None,
        "monthly_rent": monthly_rent,
        **fees,
        "rental_type": rental_type,
        "bedrooms": bedrooms,
        "living_rooms": living_rooms,
        "bathrooms": bathrooms,
        "area_sqm": area_sqm,
        "decoration": decoration,
        "orientation": orientation,
        "floor_level": floor_level,
        "total_floors": total_floors,
        "elevator": _yes_no(text, "电梯"),
        "parking": _search(r"车位[：:]?\s*([^\n]+)", text),
        "water_type": _search(r"用水[：:]?\s*([^\s\n]+)", text),
        "electricity_type": _search(r"用电[：:]?\s*([^\s\n]+)", text),
        "gas": _yes_no(text, "燃气"),
        "heating": _yes_no(text, "采暖"),
        "broadband": "否" if re.search(r"宽带\s*无", facility_block) else ("是" if "宽带" in facilities else "暂无数据"),
        "move_in_date": move_in,
        "lease_term": lease_term,
        "viewing": viewing,
        "facilities": ";".join(facilities),
        "tags": ";".join(tags),
        "verification_status": "贝壳验真" if listing_id else "未提取到验真编号",
        "publisher_compliance": compliance,
        "maintenance_date": _maintenance_date(text, fetched_at),
        "availability_status": "在租",
        "latitude": None,
        "longitude": None,
        "coordinate_type": None,
        "source": "贝壳",
        "source_url": source_url,
    }
    return {column: record.get(column) for column in RENTAL_COLUMNS}


def valid_listing(record: dict[str, Any]) -> bool:
    return bool(record.get("listing_id") and record.get("title") and record.get("monthly_rent"))


def parse_ke_list_page(raw_text: str, fetched_at: datetime | None = None) -> list[dict[str, Any]]:
    """Parse copied list-page text into lightweight rental candidates."""
    fetched_at = fetched_at or datetime.now().astimezone()
    text = _clean(raw_text)
    title_pattern = re.compile(r"^((?:整租|合租|独栋)[·・].+?)_(?:.+?租房(?:广告)?)$", re.M)
    matches = list(title_pattern.finditer(text))
    candidates = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        linked_title = _clean(match.group(1))
        title = _extract_title(block) or linked_title

        rent_range = re.search(r"(\d+)\s*[-–—至到]\s*(\d+)\s*元/月", block)
        rent = _number(_search(r"(\d+)\s*元/月", block), int)
        rent_min = int(rent_range.group(1)) if rent_range else rent
        rent_max = int(rent_range.group(2)) if rent_range else rent
        area_range = re.search(r"([\d.]+)\s*[-–—至到]\s*([\d.]+)㎡", block)
        area = _number(_search(r"([\d.]+)㎡", block))
        area_min = float(area_range.group(1)) if area_range else area
        area_max = float(area_range.group(2)) if area_range else area

        house = re.search(r"(\d+)室(\d+)厅(\d+)卫", block)
        community = _search(r"龙岗区[-－]大运新城[-－]([^/\n]+)", block)
        if not community and "_" in match.group(0):
            community = _clean(match.group(0).rsplit("_", 1)[-1].replace("租房广告", "").replace("租房", ""))
        orientation = _search(r"㎡\s*/\s*([^/\n]+)\s*/\s*\d+室", block)
        maintenance_date = _maintenance_date(block, fetched_at)
        known_tags = [
            "贝壳优选", "必看好房", "自营", "月租", "近地铁", "精装", "押一付一",
            "随时看房", "业主自荐", "有阳台", "开放厨房", "拎包入住", "独栋公寓",
            "可短租", "不接受短租", "不短租", "民水民电", "免中介",
        ]
        tags = [tag for tag in known_tags if tag in block]
        candidate_key = f"{title}|{community}|{rent_min}|{area_min}"
        candidate_id = "LIST-" + hashlib.sha1(candidate_key.encode("utf-8")).hexdigest()[:12].upper()
        candidates.append(
            {
                "candidate_id": candidate_id,
                "title": title,
                "community": community,
                "district": "龙岗区",
                "area_sqm_min": area_min,
                "area_sqm_max": area_max,
                "orientation": orientation,
                "bedrooms": int(house.group(1)) if house else None,
                "living_rooms": int(house.group(2)) if house else None,
                "bathrooms": int(house.group(3)) if house else None,
                "monthly_rent_min": rent_min,
                "monthly_rent_max": rent_max,
                "rental_type": title.split("·", 1)[0] if "·" in title else None,
                "tags": ";".join(tags),
                "maintenance_date": maintenance_date,
                "is_advertisement": int("广告" in match.group(0)),
                "detail_status": "待人工补充",
                "source": "贝壳列表页人工复制",
            }
        )
    return candidates

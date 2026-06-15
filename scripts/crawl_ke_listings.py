from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import sys
import urllib.robotparser
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.rental_parser import RENTAL_COLUMNS, parse_ke_listing, valid_listing


DEFAULT_LIST_URL = "https://sz.zu.ke.com/zufang/longgangqu/rs%E5%A4%A7%E8%BF%90/"
DEFAULT_RAW = ROOT / "data" / "raw" / "ke_listings.jsonl"
DEFAULT_LINKS = ROOT / "data" / "raw" / "ke_listing_links.json"
DEFAULT_CSV = ROOT / "data" / "rental_listings_cuhksz.csv"
DEFAULT_PROFILE = ROOT / "work" / "ke_browser_profile"
BLOCK_MARKERS = ["访问异常", "安全验证", "验证码", "请求过于频繁", "操作过于频繁"]


def check_robots(url: str) -> tuple[bool, str]:
    robots_url = "https://sz.zu.ke.com/robots.txt"
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        parser.read()
        return parser.can_fetch("FoodMateResearchCrawler", url), robots_url
    except Exception as exc:
        return True, f"{robots_url}（读取失败，仅提示：{exc}）"


def load_raw_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def merge_csv(output_path: Path, records: list[dict[str, Any]]) -> pd.DataFrame:
    new_frame = pd.DataFrame(records, columns=RENTAL_COLUMNS)
    if output_path.exists():
        old_frame = pd.read_csv(output_path, encoding="utf-8")
        for column in RENTAL_COLUMNS:
            if column not in old_frame.columns:
                old_frame[column] = None
        combined = pd.concat([old_frame[RENTAL_COLUMNS], new_frame], ignore_index=True)
    else:
        combined = new_frame
    combined["_key"] = combined["listing_id"].fillna(combined["source_url"])
    combined = combined.drop_duplicates(subset=["_key"], keep="last").drop(columns=["_key"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8-sig")
    return combined


async def visible_text(page) -> str:
    try:
        return await page.locator("body").inner_text(timeout=10000)
    except Exception:
        return ""


async def save_debug_snapshot(page, prefix: str) -> tuple[Path, Path]:
    debug_dir = ROOT / "data" / "raw" / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = debug_dir / f"{prefix}_{timestamp}.html"
    screenshot_path = debug_dir / f"{prefix}_{timestamp}.png"
    html_path.write_text(await page.content(), encoding="utf-8")
    await page.screenshot(path=str(screenshot_path), full_page=True)
    return html_path, screenshot_path


async def wait_for_manual_ready(page, reason: str) -> None:
    print(f"\n{reason}")
    print(f"当前页面：{page.url}")
    print("请在脚本打开的 Chromium 中完成验证，并等待租房列表和价格正常显示。")
    input("页面准备好后回到此 PowerShell，按 Enter 继续；按 Ctrl+C 可停止：")
    await page.wait_for_timeout(1500)


async def handle_verification(page, interactive: bool) -> None:
    text = await visible_text(page)
    if not any(marker in text for marker in BLOCK_MARKERS):
        return
    if not interactive:
        raise RuntimeError("页面出现验证或访问限制，已停止采集")
    await wait_for_manual_ready(page, "页面出现验证或访问限制；脚本不会自动绕过验证。")
    text = await visible_text(page)
    if any(marker in text for marker in BLOCK_MARKERS):
        raise RuntimeError("验证仍未解除，已停止采集")


async def extract_listing_cards(page) -> list[dict[str, str]]:
    return await page.locator('a[href*="/zufang/"]').evaluate_all(
        """
        elements => elements.map(anchor => {
          let node = anchor;
          let best = anchor.innerText || '';
          for (let i = 0; i < 7 && node; i++, node = node.parentElement) {
            const text = (node.innerText || '').trim();
            if (text.includes('元/月') && text.includes('㎡') && text.length < 1500) best = text;
          }
          return {url: anchor.href, anchor_text: (anchor.innerText || '').trim(), card_text: best};
        })
        """
    )


async def collect_links(
    page,
    list_url: str,
    max_pages: int,
    interactive: bool,
    manual_ready: bool = False,
) -> list[dict[str, str]]:
    await page.goto(list_url, wait_until="domcontentloaded", timeout=30000)
    await page.wait_for_timeout(2500)
    await handle_verification(page, interactive)
    if manual_ready and interactive:
        await wait_for_manual_ready(page, "首次运行使用人工确认模式。")
    collected: dict[str, dict[str, str]] = {}

    for page_number in range(1, max_pages + 1):
        await page.wait_for_timeout(1500)
        cards = await extract_listing_cards(page)
        if not cards and interactive:
            await wait_for_manual_ready(
                page,
                "当前页面尚未发现房源链接，可能仍在验证、跳转或加载中。",
            )
            await handle_verification(page, interactive)
            cards = await extract_listing_cards(page)
        for item in cards:
            url = str(item.get("url", "")).split("?")[0]
            if re.search(r"/zufang/SZ\d+\.html$", url):
                collected[url] = {"url": url, "anchor_text": item.get("anchor_text", ""), "card_text": item.get("card_text", "")}
        print(f"列表第 {page_number} 页：累计发现 {len(collected)} 个普通房源详情链接")

        if not collected and interactive:
            await wait_for_manual_ready(
                page,
                "页面中存在导航链接，但还没有识别到普通房源详情链接。请确认验证已完成且房源卡片已显示。",
            )
            await handle_verification(page, interactive)
            cards = await extract_listing_cards(page)
            for item in cards:
                url = str(item.get("url", "")).split("?")[0]
                if re.search(r"/zufang/SZ\d+\.html$", url):
                    collected[url] = {
                        "url": url,
                        "anchor_text": item.get("anchor_text", ""),
                        "card_text": item.get("card_text", ""),
                    }
            print(f"人工确认后：累计发现 {len(collected)} 个普通房源详情链接")

        if not collected:
            html_path, screenshot_path = await save_debug_snapshot(page, "ke_list_zero_links")
            raise RuntimeError(
                "仍未发现普通房源详情链接。"
                f"已保存调试页面：{html_path}；截图：{screenshot_path}"
            )

        if page_number >= max_pages:
            break
        next_link = page.get_by_text("下一页", exact=True)
        if await next_link.count() == 0:
            break
        href = await next_link.first.get_attribute("href")
        disabled = await next_link.first.get_attribute("class") or ""
        if not href or "disabled" in disabled:
            break
        await next_link.first.click()
        await page.wait_for_load_state("domcontentloaded", timeout=30000)
        await handle_verification(page, interactive)
        await asyncio.sleep(random.uniform(2.0, 3.5))
    return list(collected.values())


async def crawl(args) -> None:
    allowed, robots_info = check_robots(args.list_url)
    print(f"robots 检查：{robots_info}")
    if not allowed and not args.ignore_robots:
        raise SystemExit("robots 规则不允许该自动访问。请停止采集，或先取得网站授权。")

    try:
        from playwright.async_api import async_playwright
    except ImportError as exc:
        raise SystemExit("缺少 Playwright。运行：pip install playwright；python -m playwright install chromium") from exc

    raw_rows = load_raw_rows(args.raw_output)
    completed_urls = {row.get("url") for row in raw_rows if row.get("status") == "ok"}
    parsed_records = [row["parsed"] for row in raw_rows if row.get("status") == "ok" and row.get("parsed")]

    async with async_playwright() as playwright:
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(args.profile_dir),
            headless=args.headless,
            viewport={"width": 1440, "height": 900},
            locale="zh-CN",
        )
        page = context.pages[0] if context.pages else await context.new_page()
        page.set_default_timeout(15000)

        if args.links_file.exists() and args.use_saved_links:
            links = json.loads(args.links_file.read_text(encoding="utf-8"))
            print(f"读取已保存链接：{len(links)} 条")
        else:
            links = await collect_links(
                page,
                args.list_url,
                args.max_pages,
                not args.headless,
                manual_ready=args.manual_ready,
            )
            args.links_file.parent.mkdir(parents=True, exist_ok=True)
            args.links_file.write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"链接清单已保存：{args.links_file}")

        pending = [item for item in links if item["url"] not in completed_urls]
        if args.max_details:
            pending = pending[: args.max_details]
        print(f"本次待抓取详情页：{len(pending)} 条")

        for index, item in enumerate(pending, start=1):
            url = item["url"]
            fetched_at = datetime.now().astimezone()
            print(f"[{index}/{len(pending)}] {url}")
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await handle_verification(page, not args.headless)
                await page.wait_for_timeout(1000)
                text = await visible_text(page)
                parsed = parse_ke_listing(text, url, item.get("card_text", ""), fetched_at)
                status = "ok" if valid_listing(parsed) else "parse_failed"
                raw_row = {
                    "url": url,
                    "fetched_at": fetched_at.isoformat(),
                    "status": status,
                    "card_text": item.get("card_text", ""),
                    "page_title": await page.title(),
                    "raw_text": text,
                    "parsed": parsed,
                }
                append_jsonl(args.raw_output, raw_row)
                if status == "ok":
                    parsed_records.append(parsed)
                    print(f"  成功：{parsed['listing_id']}，{parsed['monthly_rent']} 元/月")
                else:
                    print("  页面已保存，但关键字段解析失败，请检查原文")
            except Exception as exc:
                append_jsonl(
                    args.raw_output,
                    {"url": url, "fetched_at": fetched_at.isoformat(), "status": "error", "error": str(exc)},
                )
                print(f"  失败：{exc}")
                if any(marker in str(exc) for marker in ["验证", "访问限制"]):
                    break
            await asyncio.sleep(random.uniform(args.delay_min, args.delay_max))

        await context.close()

    combined = merge_csv(args.csv_output, parsed_records)
    print(f"\n结构化知识库：{args.csv_output}")
    print(f"原始页面归档：{args.raw_output}")
    print(f"知识库当前共 {len(combined)} 条房源")


def parse_args():
    parser = argparse.ArgumentParser(description="低频采集贝壳公开租房详情页并写入 FoodMate 租房知识库")
    parser.add_argument("--list-url", default=DEFAULT_LIST_URL)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--max-details", type=int, default=3, help="单次最多访问的详情页数量；0 表示不限制")
    parser.add_argument("--delay-min", type=float, default=5.0)
    parser.add_argument("--delay-max", type=float, default=8.0)
    parser.add_argument("--profile-dir", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--raw-output", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--links-file", type=Path, default=DEFAULT_LINKS)
    parser.add_argument("--csv-output", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--use-saved-links", action="store_true")
    parser.add_argument("--headless", action="store_true", help="无界面运行；遇到验证时无法人工处理")
    parser.add_argument(
        "--manual-ready",
        action="store_true",
        help="列表页打开后固定等待人工完成验证并确认页面已加载",
    )
    parser.add_argument("--ignore-robots", action="store_true", help="仅在确认已取得网站授权时使用")
    args = parser.parse_args()
    if args.max_details == 0:
        args.max_details = None
    if args.delay_min < 1 or args.delay_max < args.delay_min:
        parser.error("delay 参数不合法；建议保持至少 5 秒间隔")
    return args


if __name__ == "__main__":
    asyncio.run(crawl(parse_args()))

from __future__ import annotations

import unicodedata

from src.agent import FoodMateAgent


DEMO_CASES = [
    "工作日中午和同学吃饭，想找有套餐或优惠的店，人均80元以内，港中深附近。",
    "晚上想和朋友吃火锅，人均110元以内，离地铁近一点。",
    "下午想找地方自习办公，预算50元以内，最好有咖啡。",
]


SCORE_COLUMNS = [
    "name",
    "final_score",
    "semantic_score",
    "budget_score",
    "distance_score",
    "cuisine_score",
    "scene_score",
    "deal_score",
]


def display_width(text: object) -> int:
    width = 0
    for char in str(text):
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def pad_right(text: object, width: int) -> str:
    text = str(text)
    return text + " " * max(width - display_width(text), 0)


def pad_left(text: object, width: int) -> str:
    text = str(text)
    return " " * max(width - display_width(text), 0) + text


def format_score(value: object) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def format_score_table(df) -> str:
    rows = []
    for _, row in df[SCORE_COLUMNS].iterrows():
        rows.append(
            {
                "name": str(row["name"]),
                "final_score": format_score(row["final_score"]),
                "semantic_score": format_score(row["semantic_score"]),
                "budget_score": format_score(row["budget_score"]),
                "distance_score": format_score(row["distance_score"]),
                "cuisine_score": format_score(row["cuisine_score"]),
                "scene_score": format_score(row["scene_score"]),
                "deal_score": format_score(row.get("deal_score", 0)),
            }
        )

    widths = {}
    for col in SCORE_COLUMNS:
        widths[col] = max(display_width(col), *(display_width(row[col]) for row in rows))

    header = "  ".join(
        pad_right(col, widths[col]) if col == "name" else pad_left(col, widths[col])
        for col in SCORE_COLUMNS
    )
    divider = "  ".join("-" * widths[col] for col in SCORE_COLUMNS)
    body = [
        "  ".join(
            pad_right(row[col], widths[col]) if col == "name" else pad_left(row[col], widths[col])
            for col in SCORE_COLUMNS
        )
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def main() -> None:
    for idx, text in enumerate(DEMO_CASES, start=1):
        print("=" * 100)
        print(f"Demo case {idx}: {text}")
        print()
        agent = FoodMateAgent()
        result = agent.handle(text)
        if result["type"] == "followup":
            print("Agent 追问:", result["message"])
            if "预算" in result["message"]:
                result = agent.handle("人均 80 元以内")
            elif "场景" in result["message"]:
                result = agent.handle("朋友聚餐")
        print(result["message"])
        if not result["recommendations"].empty:
            print("\nScore table:")
            print(format_score_table(result["recommendations"]))


if __name__ == "__main__":
    main()

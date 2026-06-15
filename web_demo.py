from __future__ import annotations

import html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from src.agent import FoodMateAgent


EXAMPLES = [
    "今晚想和朋友聚餐，人均 80 元以内，想吃亚洲菜，不要太辣，离学校近一点。",
    "想找一家适合约会的餐厅，预算 180 元以内，希望环境安静、有氛围。",
    "中午只有 30 分钟，预算 45 元，想吃健康一点，最好能快速出餐。",
    "下午想找个安静的地方自习办公，人均 50 元以内，最好有咖啡和甜品。",
]


def page(query: str = "") -> str:
    agent = FoodMateAgent()
    result = agent.handle(query) if query else None
    examples = "".join(
        f"<li><a href='/?q={html.escape(example, quote=True)}'>{html.escape(example)}</a></li>"
        for example in EXAMPLES
    )
    output = ""
    if result:
        output = f"<section><h2>Agent 回复</h2><pre>{html.escape(result['message'])}</pre></section>"
        if result["type"] == "recommendation":
            rows = ""
            for _, row in result["recommendations"].iterrows():
                rows += (
                    "<tr>"
                    f"<td>{html.escape(str(row['name']))}</td>"
                    f"<td>{html.escape(str(row['cuisine']))}</td>"
                    f"<td>{row['price_per_person']} 元</td>"
                    f"<td>{row['distance_km']} 公里</td>"
                    f"<td>{row['rating']}</td>"
                    f"<td>{row['final_score']:.3f}</td>"
                    f"<td>{html.escape(str(row['review_summary']))}</td>"
                    "</tr>"
                )
            output += (
                "<section><h2>评分表</h2>"
                "<table><thead><tr><th>店名</th><th>菜系</th><th>人均</th><th>距离</th>"
                "<th>评分</th><th>综合分</th><th>引用证据</th></tr></thead>"
                f"<tbody>{rows}</tbody></table></section>"
            )

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FoodMate 中文餐厅推荐 Agent</title>
  <style>
    body {{ margin: 0; font-family: Arial, "Microsoft YaHei", sans-serif; color: #17202a; background: #f7f8fa; }}
    header {{ background: #1f6f8b; color: white; padding: 24px 32px; }}
    main {{ max-width: 1120px; margin: 0 auto; padding: 24px; }}
    form {{ display: flex; gap: 8px; margin: 16px 0; }}
    input {{ flex: 1; padding: 12px; border: 1px solid #ccd3db; border-radius: 6px; font-size: 15px; }}
    button {{ padding: 12px 18px; border: 0; border-radius: 6px; background: #1f6f8b; color: white; font-weight: 700; }}
    section {{ background: white; border: 1px solid #e1e5ea; border-radius: 8px; padding: 18px; margin-top: 16px; }}
    pre {{ white-space: pre-wrap; line-height: 1.55; font-family: Consolas, "Microsoft YaHei", monospace; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ border-bottom: 1px solid #e1e5ea; padding: 10px; text-align: left; vertical-align: top; }}
    th {{ background: #eef5f7; }}
    li {{ margin: 8px 0; }}
    a {{ color: #1f6f8b; }}
  </style>
</head>
<body>
  <header>
    <h1>FoodMate 中文餐厅 RAG 推荐 Agent</h1>
  </header>
  <main>
    <form method="get">
      <input name="q" value="{html.escape(query, quote=True)}" placeholder="输入预算、菜系、距离、用餐场景，例如：人均80元，朋友聚餐，中餐，不要辣">
      <button type="submit">推荐</button>
    </form>
    <section>
      <h2>演示输入</h2>
      <ul>{examples}</ul>
    </section>
    {output}
  </main>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query).get("q", [""])[0]
        body = page(query).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), Handler)
    print("FoodMate web demo running at http://127.0.0.1:8000")
    server.serve_forever()


if __name__ == "__main__":
    main()

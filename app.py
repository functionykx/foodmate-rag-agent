from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from src.llm import llm_enabled
from src.multi_agent import SupervisorAgent


st.set_page_config(page_title="打工人生活助手", layout="wide")


@st.cache_resource
def get_agent() -> SupervisorAgent:
    return SupervisorAgent()


agent = get_agent()

st.title("打工人生活助手")
st.caption("一个入口完成附近餐厅推荐与租房筛选，支持多轮补充条件、RAG 检索和可解释排序。")

with st.sidebar:
    st.subheader("运行状态")
    st.caption(f"召回候选：Top {os.getenv('FOODMATE_RECALL_TOP_K', '30')}")
    st.caption(f"餐厅 Pipeline：{os.getenv('FOODMATE_PIPELINE_MODE', 'hybrid')}")
    default_restaurant_data = "data/restaurants_cuhksz_geo.csv" if Path("data/restaurants_cuhksz_geo.csv").exists() else "data/restaurants_cuhksz.csv"
    restaurant_data = Path(os.getenv("FOODMATE_DATA_PATH", default_restaurant_data)).name
    st.caption(f"餐厅数据源：{restaurant_data}")
    st.caption(f"LLM：{'已启用' if llm_enabled() else '未启用，使用规则与模板'}")
    independent_mode = st.checkbox("每次输入独立推荐", value=False)
    user_location = st.text_input(
        "地点（餐厅出发点 / 租房通勤目的地）",
        placeholder="例如：大运中心地铁站C口或深圳北站",
    )
    transport_label = st.selectbox("出行方式", ["步行", "骑行", "驾车"], disabled=not bool(user_location))

    st.subheader("演示示例")
    examples = [
        "预算80元，和同事吃工作日晚餐，想吃辣一点的中餐",
        "想找安静的咖啡店坐坐，人均50元以内",
        "想在大运附近整租一室一厅，月租3000元以内，要近地铁、精装修",
        "租房预算4500元，至少两室，面积70平以上，需要电梯和天然气",
    ]
    selected = st.selectbox("选择示例", examples)
    if st.button("使用示例", use_container_width=True):
        st.session_state["pending_input"] = selected
    if st.button("重置对话", use_container_width=True):
        agent.reset()
        st.session_state["messages"] = []
        st.session_state["awaiting_followup"] = False

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "awaiting_followup" not in st.session_state:
    st.session_state["awaiting_followup"] = False

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])

pending = st.session_state.pop("pending_input", None)
user_input = pending or st.chat_input("描述你想吃什么，或输入月租预算、户型、面积和设施要求")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    if independent_mode and not st.session_state.get("awaiting_followup", False):
        agent.reset()

    agent_input = user_input
    if user_location:
        rental_tokens = ["租房", "房源", "月租", "租金", "整租", "合租", "公寓", "户型"]
        rental_context = (
            any(token in user_input for token in rental_tokens)
            or (
                st.session_state.get("awaiting_followup", False)
                and agent.last_intent == "rental"
            )
        )
        if rental_context:
            agent_input += f"。通勤目的地：{user_location}。出行方式：{transport_label}"
        else:
            agent_input += f"。当前位置：{user_location}。出行方式：{transport_label}"

    result = agent.run(agent_input)
    st.session_state["awaiting_followup"] = result["type"] == "followup"
    st.session_state["messages"].append({"role": "assistant", "content": result["message"]})
    with st.chat_message("assistant"):
        st.write(result["message"])

    recs = result.get("recommendations")
    if result["type"] == "recommendation" and recs is not None and not recs.empty:
        st.subheader("餐厅 Top 5")
        display_cols = [
            "name", "cuisine", "price_per_person", "rating", "distance_km", "final_score",
            "semantic_score", "budget_score", "distance_score", "cuisine_score", "scene_score", "deal_score",
        ]
        display_cols = [column for column in display_cols if column in recs.columns]
        st.dataframe(recs[display_cols], use_container_width=True, hide_index=True)
        st.subheader("知识库引用")
        for _, row in recs.iterrows():
            with st.expander(str(row["name"])):
                st.write(f"推荐菜：{row.get('menu_highlights', '暂无')} ")
                st.write(f"评价摘要：{row.get('review_summary', '暂无')}")
                st.write(f"优点：{row.get('pros', '暂无')}")
                st.write(f"注意：{row.get('cons', '暂无')}")

    if result["type"] == "rental_recommendation" and recs is not None and not recs.empty:
        st.subheader("租房 Top 5")
        rental_cols = [
            "title", "monthly_rent", "bedrooms", "living_rooms", "area_sqm", "orientation",
            "facilities", "verification_status", "final_score", "semantic_score", "budget_score",
            "bedroom_score", "area_score", "facility_score", "verified_score",
            "rental_type_score", "move_in_score", "lease_score",
            "commute_distance_km", "commute_duration_min", "commute_score", "commute_source",
        ]
        rental_cols = [column for column in rental_cols if column in recs.columns]
        st.dataframe(recs[rental_cols], use_container_width=True, hide_index=True)
        st.subheader("房源依据与风险提示")
        for _, row in recs.iterrows():
            with st.expander(str(row["title"])):
                st.write(f"房源编号：{row.get('listing_id', '未提供')}")
                st.write(f"标签：{row.get('tags', '暂无')}")
                st.write(f"设施：{row.get('facilities', '暂无')}")
                st.write(f"入住：{row.get('move_in_date', '需咨询')}；看房：{row.get('viewing', '需咨询')}")
                st.write(f"验真状态：{row.get('verification_status', '未知')}")
                if pd.notna(row.get("commute_duration_min")):
                    st.write(
                        f"通勤：{row.get('commute_distance_km')}公里，"
                        f"约{row.get('commute_duration_min')}分钟（{row.get('commute_source')}）"
                    )
                st.warning("房源价格、费用、设施和在租状态可能变化，签约前请向平台或经纪人再次核验。")

    if result["type"] == "experiment":
        st.subheader("A/B 评估")
        table = result.get("metrics", {}).get("table", [])
        if table:
            st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("Agent 调度与工具轨迹"):
        left, right = st.columns(2)
        with left:
            st.write("Supervisor Intent")
            st.code(result.get("supervisor_intent", "unknown"))
            st.write("Active Agent")
            st.code(result.get("active_agent", "unknown"))
            st.write("Plan")
            st.json(result.get("supervisor_plan", []))
        with right:
            st.write("Critic")
            st.json(result.get("multi_agent_critic", {}))
            st.write("Latency")
            st.code(f"{result.get('latency_ms', 0.0):.2f} ms")
        st.json(result.get("multi_agent_actions", []))

with st.sidebar:
    st.subheader("当前任务")
    st.code(agent.state.intent)
    st.subheader("当前偏好")
    st.json(agent.state.preferences)

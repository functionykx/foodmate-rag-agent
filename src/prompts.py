SYSTEM_STYLE = """
你是 FoodMate，一个面向中文本地生活场景的餐厅推荐 Agent。
推荐理由必须来自检索到的餐厅知识库事实，并且要同时说明优点和可能缺点。
"""


def build_query(preferences: dict, user_message: str) -> str:
    pieces = [user_message]
    for key in ["cuisine", "scene"]:
        if preferences.get(key):
            pieces.append(str(preferences[key]))
    if preferences.get("avoid_spicy"):
        pieces.append("不辣 清淡 少辣")
    if preferences.get("vegetarian"):
        pieces.append("素食 植物基")
    if preferences.get("deal_preference"):
        pieces.append("优惠 团购 套餐 工作日 午市 学生")
    if preferences.get("wants_spicy"):
        pieces.append("要辣 辣 重口味 麻辣 香辣 湘菜 川菜 重庆火锅")
    if preferences.get("cuisine") == "甜品":
        pieces.append("甜品 甜点 蛋糕 面包 下午茶 奶茶 双皮奶 舒芙蕾 漏奶华")
    if preferences.get("scene") == "独自用餐":
        pieces.append("一人食 单人 独自用餐 自己吃 小吃 下午茶")
    if preferences.get("min_budget") and preferences.get("budget"):
        pieces.append(f"人均 {preferences['min_budget']} 到 {preferences['budget']} 元")
    if preferences.get("budget"):
        pieces.append(f"人均 {preferences['budget']} 元以内")
    if preferences.get("max_distance_km"):
        pieces.append(f"{preferences['max_distance_km']} 公里以内 学校附近")
    return " ".join(pieces)

"""
Gradio Workout Logger + 教練機器人（Groq）— 單檔可執行 app.py
需求：
1) Date 預設今天、可修改
2) 6 個 item；輸入過的動作會記憶成選項（可自訂）
3) 每個 item 有 5 組 set（每組 kg + reps）
4) 每個 item 有 Note 欄
5) Save 會把資料追加存到 CSV，Records 分頁可查詢
6) Coach 分頁：gr.Chatbot + Groq 串流回覆（API key 走 os.getenv('groq_key')）

執行方式：
    pip install gradio pandas python-dateutil
    python app.py
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List
from datetime import datetime, date

# 3) groq 安裝（照使用者指定寫法）
try:
    from groq import Groq
except ImportError:  # 若沒裝就安裝
    os.system('pip install groq')
    from groq import Groq

import gradio as gr
import pandas as pd

# ------------ 常數與檔案路徑 ------------
APP_TITLE = "Workout Logger"
RECORDS_CSV = Path("workout_records.csv")
ITEMS_JSON = Path("known_items.json")
NUM_ITEMS = 6
NUM_SETS = 5

# ------------ Groq（教練機器人）設定 ------------
GROQ_API_KEY = os.getenv("groq_key")  # 依需求使用此環境變數名稱
try:
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception:
    groq_client = None

SYSTEM_PROMPT = (
    "你是一個講繁體中文(Zh-tw)的健身教練，你很樂觀、會鼓勵人，也會講有趣的笑話。"
    "無論學生問什麼問題，都盡量把話題引導至運動與健身。請用口語、短段落，"
    "提供具體可行的訓練建議（動作/組數/重量或RPE），並適度提醒安全與暖身放鬆。"
)
GROQ_MODEL = "llama-3.3-70b-versatile"

# ------------ 資料存取工具 ------------
def load_known_items() -> List[str]:
    if ITEMS_JSON.exists():
        try:
            return json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def save_known_items(items: List[str]):
    # 去重、移除空白
    uniq = []
    for it in items:
        it = (it or "").strip()
        if it and it not in uniq:
            uniq.append(it)
    ITEMS_JSON.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_records_csv():
    if not RECORDS_CSV.exists():
        cols = [
            "date", "item",
        ]
        # set1_kg, set1_reps ... set5_kg, set5_reps
        for s in range(1, NUM_SETS+1):
            cols += [f"set{s}_kg", f"set{s}_reps"]
        cols += ["note", "total_volume_kg", "created_at"]
        pd.DataFrame(columns=cols).to_csv(RECORDS_CSV, index=False, encoding="utf-8")


def append_records(rows: List[dict]):
    ensure_records_csv()
    if not rows:
        return
    df_old = pd.read_csv(RECORDS_CSV)
    df_new = pd.DataFrame(rows)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(RECORDS_CSV, index=False, encoding="utf-8")

# ------------ 業務邏輯：儲存紀錄 ------------

def compute_total_volume(kg_list: List[float|None], reps_list: List[int|None]) -> float:
    total = 0.0
    for k, r in zip(kg_list, reps_list):
        if k is None or r is None:
            continue
        try:
            total += float(k) * int(r)
        except Exception:
            pass
    return round(total, 2)


def save_button_clicked(date_str: str, *flat_inputs):
    """flat_inputs 依序包含 6 個 item 的：
    [item_name, set1_kg, set1_reps, ..., set5_kg, set5_reps, note] * 6
    """
    # 解析日期
    try:
        dt = pd.to_datetime(date_str).date()
    except Exception:
        return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), pd.DataFrame()

    block_size = 1 + (NUM_SETS * 2) + 1
    rows = []
    all_new_item_names = []

    for i in range(NUM_ITEMS):
        start = i * block_size
        end = start + block_size
        chunk = list(flat_inputs[start:end])
        item_name = (chunk[0] or "").strip()
        if not item_name:
            continue
        all_new_item_names.append(item_name)

        kg_vals, reps_vals = [], []
        sets_kv = {}
        pos = 1
        for s in range(1, NUM_SETS+1):
            kg = chunk[pos]; reps = chunk[pos+1]
            pos += 2
            kg = None if kg in ("", None) else float(kg)
            reps = None if reps in ("", None) else int(reps)
            sets_kv[f"set{s}_kg"] = kg
            sets_kv[f"set{s}_reps"] = reps
            kg_vals.append(kg)
            reps_vals.append(reps)
        note = chunk[pos] if pos < len(chunk) else ""

        total_volume = compute_total_volume(kg_vals, reps_vals)
        rows.append({
            "date": dt.isoformat(),
            "item": item_name,
            **sets_kv,
            "note": note,
            "total_volume_kg": total_volume,
            "created_at": datetime.now().isoformat(timespec="seconds")
        })

    if not rows:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), pd.DataFrame()

    append_records(rows)

    # 更新已知 item 清單
    known = load_known_items()
    merged = list(dict.fromkeys([*known, *all_new_item_names]))
    save_known_items(merged)

    df = pd.read_csv(RECORDS_CSV)
    return (f"已儲存 {len(rows)} 筆（日期：{dt.isoformat()}）。", gr.update(choices=merged), df.tail(20))


# ------------ Records 搜尋 ------------

def search_records(date_from: str, date_to: str, item_filter: str):
    ensure_records_csv()
    if not RECORDS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RECORDS_CSV)

    if date_from:
        try:
            df = df[df["date"] >= pd.to_datetime(date_from).date().isoformat()]
        except Exception:
            pass
    if date_to:
        try:
            df = df[df["date"] <= pd.to_datetime(date_to).date().isoformat()]
        except Exception:
            pass

    if item_filter:
        df = df[df["item"].astype(str).str.contains(item_filter, case=False, na=False)]

    if not df.empty:
        df = df.sort_values(["date", "created_at"], ascending=[False, False])
    return df


# ------------ 教練機器人：串流回覆 ------------

def coach_chat_stream(history: list[list[str]], user_msg: str):
    """以 generator 串流更新 gr.Chatbot。history 形如 [[user, bot], ...]"""
    msg = (user_msg or "").strip()
    if not msg:
        yield history, ""
        return

    if groq_client is None:
        bot_text = "（尚未設定環境變數 groq_key，請設定後重試。）"
        history = history + [[msg, bot_text]]
        yield history, ""
        return

    # 組 messages
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, b in history:
        if u:
            messages.append({"role": "user", "content": u})
        if b:
            messages.append({"role": "assistant", "content": b})
    messages.append({"role": "user", "content": msg})

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            stream=True,
            stop=None,
        )
        bot_resp = ""
        history = history + [[msg, ""]]
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                bot_resp += delta
                history[-1][1] = bot_resp
                yield history, ""
        return
    except Exception as e:
        history = history + [[msg, f"抱歉，Groq 呼叫失敗：{e}"]]
        yield history, ""


# ------------ 建立介面 ------------
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏋️‍♂️ Workout Logger + 🤖 Coach
快速記錄重量訓練與查詢歷史，並附帶教練機器人提供訓練建議。")

    with gr.Tabs():
        # ---- Log 分頁 ----
        with gr.TabItem("Log"):
            today_str = date.today().isoformat()
            date_in = gr.Textbox(value=today_str, label="Date (YYYY-MM-DD)")

            known_items = load_known_items()

            item_dropdowns = []
            set_inputs = []
            note_inputs = []

            for i in range(NUM_ITEMS):
                with gr.Group():
                    gr.Markdown(f"### Item {i+1}")
                    dd = gr.Dropdown(choices=known_items, allow_custom_value=True, value=None,
                                     label=f"Item {i+1} Name")
                    item_dropdowns.append(dd)

                    row_inputs = []
                    for s in range(1, NUM_SETS+1):
                        with gr.Row():
                            kg = gr.Number(label=f"Set {s} — kg", precision=2)
                            reps = gr.Number(label=f"Set {s} — reps", precision=0)
                            row_inputs += [kg, reps]
                    set_inputs.append(row_inputs)

                    note = gr.Textbox(label="Note", placeholder="RPE、感覺、下次調整…")
                    note_inputs.append(note)

            save_btn = gr.Button("💾 Save", variant="primary")
            status_md = gr.Markdown("")
            latest_df = gr.Dataframe(headers=None, value=pd.DataFrame(), wrap=True, interactive=False, label="最近 20 筆紀錄")

            flat_all_inputs = []
            for i in range(NUM_ITEMS):
                flat_all_inputs.append(item_dropdowns[i])
                flat_all_inputs += set_inputs[i]
                flat_all_inputs.append(note_inputs[i])

            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, *flat_all_inputs],
                outputs=[status_md, item_dropdowns[0], latest_df],
            )

        # ---- Records 分頁 ----
        with gr.TabItem("Records"):
            gr.Markdown("### 搜尋歷史紀錄")
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                q_item = gr.Textbox(label="Item 包含（關鍵字）")
            query_btn = gr.Button("🔎 Search")
            out_df = gr.Dataframe(headers=None, value=pd.DataFrame(), wrap=True, interactive=False, label="搜尋結果")
            query_btn.click(search_records, inputs=[q_from, q_to, q_item], outputs=out_df)

        # ---- Coach 分頁 ----
        with gr.TabItem("Coach"):
            gr.Markdown("""
            ### 🤖 教練機器人（Groq）
            - 會用繁體中文，用幽默與鼓勵口吻，並盡量把話題拉回運動與健身。
            - **請先設定環境變數 `groq_key`**（你的 Groq API Key）。
            - 模型：`llama-3.3-70b-versatile`，支援串流輸出。
            """)
            chatbot = gr.Chatbot(height=420)
            user_in = gr.Textbox(placeholder="輸入你的問題，按 Enter 或點送出…", label="訊息")
            with gr.Row():
                send_btn = gr.Button("送出", variant="primary")
                clear_btn = gr.Button("清空")

            send_btn.click(coach_chat_stream, inputs=[chatbot, user_in], outputs=[chatbot, user_in])
            user_in.submit(coach_chat_stream, inputs=[chatbot, user_in], outputs=[chatbot, user_in])
            clear_btn.click(lambda: ([], ""), None, [chatbot, user_in], queue=False)

    gr.Markdown("---
**Tips**
- Item 名稱可直接輸入新文字，下次會出現在下拉選單。
- 空白的 Item 不會儲存。
- Total Volume = ∑(kg × reps)。")

if __name__ == "__main__":
    ensure_records_csv()
    demo.launch()

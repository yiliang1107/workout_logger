"""
Gradio Workout Logger + 你的教練（Groq）— 單檔可執行 app.py
更新：
- 每次只記錄 1 個 Item（多次 Save 以追加）
- kg / reps 輸入預設為空（不顯示 0）
- Item 下拉會列出過去紀錄中的動作名稱（亦可自訂新名稱）
- Coach 分頁名稱改為「你的教練」，不顯示多餘說明文字
- Records/最新紀錄：調整 Note 欄位為最寬（以 CSS 力度加強），並確保 Note 放在最後一欄

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

# 依需求：groq 安裝/匯入
try:
    from groq import Groq
except ImportError:
    os.system('pip install groq')
    from groq import Groq

import gradio as gr
import pandas as pd

# ------------ 常數與檔案路徑 ------------
APP_TITLE = "Workout Logger"
RECORDS_CSV = Path("workout_records.csv")
ITEMS_JSON = Path("known_items.json")
NUM_ITEMS = 1            # 每次只記錄 1 個 Item
NUM_SETS = 5

# ------------ Groq（教練機器人）設定 ------------
GROQ_API_KEY = os.getenv("groq_key")  # 用 secret: groq_key
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

# ------------ Data I/O 工具 ------------
def load_known_items() -> List[str]:
    if ITEMS_JSON.exists():
        try:
            return json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_known_items(items: List[str]):
    uniq = []
    for it in items:
        it = (it or "").strip()
        if it and it not in uniq:
            uniq.append(it)
    ITEMS_JSON.write_text(json.dumps(uniq, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_records_csv():
    if not RECORDS_CSV.exists():
        cols = ["date", "item"]
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


def get_all_item_choices() -> List[str]:
    """合併 JSON 與 CSV 中出現過的 item；依歷史出現頻率排序。"""
    seen = []
    # 從 CSV 抓 item 次數
    if RECORDS_CSV.exists():
        try:
            df = pd.read_csv(RECORDS_CSV)
            counts = (
                df["item"].dropna().astype(str).str.strip().value_counts()
                if "item" in df.columns else pd.Series(dtype=int)
            )
            seen += [x for x in counts.index.tolist() if x]
        except Exception:
            pass
    # 加入 JSON 中的 known_items（去重）
    for it in load_known_items():
        if it and it not in seen:
            seen.append(it)
    return seen

# ------------ 儲存紀錄邏輯 ------------

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
    """flat_inputs 內容：
    [item_name, set1_kg, set1_reps, ..., set5_kg, set5_reps, note]
    """
    try:
        dt = pd.to_datetime(date_str).date()
    except Exception:
        return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), pd.DataFrame()

    block_size = 1 + (NUM_SETS * 2) + 1
    # 僅一個 item
    chunk = list(flat_inputs[:block_size])
    item_name = (chunk[0] or "").strip()
    if not item_name:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), pd.DataFrame()

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
    row = {
        "date": dt.isoformat(),
        "item": item_name,
        **sets_kv,
        "note": note,
        "total_volume_kg": total_volume,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    append_records([row])

    # 更新已知 item 清單
    known = load_known_items()
    if item_name not in known:
        known.append(item_name)
        save_known_items(known)

    # 重新抓選單（含 CSV 歷史）
    merged_choices = get_all_item_choices()

    # 最新 20 筆，並把 note 放最後一欄
    df = pd.read_csv(RECORDS_CSV)
    if not df.empty and "note" in df.columns:
        cols = [c for c in df.columns if c != "note"] + ["note"]
        df = df[cols]

    return (f"已儲存 1 筆（日期：{dt.isoformat()}）。", gr.update(choices=merged_choices), df.tail(20))


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
        # 讓 note 放在最後一欄
        if "note" in df.columns:
            cols = [c for c in df.columns if c != "note"] + ["note"]
            df = df[cols]
    return df


# ------------ 教練機器人：串流回覆 ------------

def coach_chat_stream(history: list[list[str]], user_msg: str):
    msg = (user_msg or "").strip()
    if not msg:
        yield history, ""
        return

    if groq_client is None:
        bot_text = "（尚未設定環境變數 groq_key，請設定後重試。）"
        history = history + [[msg, bot_text]]
        yield history, ""
        return

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


# ------------ CSS：擴大 Note 欄位寬度 ------------
CSS = """
#records_df table, #latest_df table { table-layout: fixed; width: 100%; }
#records_df table th:last-child, #records_df table td:last-child,
#latest_df table th:last-child, #latest_df table td:last-child { width: 48% !important; }
"""

# ------------ 建立介面 ------------
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# 🏋️‍♂️ Workout Logger + 🤖 你的教練
快速記錄重量訓練與查詢歷史。")

    with gr.Tabs():
        # ---- Log 分頁（單一 Item） ----
        with gr.TabItem("Log"):
            today_str = date.today().isoformat()
            date_in = gr.Textbox(value=today_str, label="Date (YYYY-MM-DD)")

            # 選單：合併歷史（CSV）與已知（JSON）
            item_choices = get_all_item_choices()

            gr.Markdown("### Item 1")
            item_dd = gr.Dropdown(choices=item_choices, allow_custom_value=True, value=None, label="Item 名稱")

            set_inputs = []
            for s in range(1, NUM_SETS+1):
                with gr.Row():
                    kg = gr.Number(label=f"Set {s} — kg", precision=2, value=None, placeholder="kg")
                    reps = gr.Number(label=f"Set {s} — reps", precision=0, value=None, placeholder="reps")
                    set_inputs += [kg, reps]
            note_in = gr.Textbox(label="Note", placeholder="RPE、感覺、下次調整…")

            save_btn = gr.Button("💾 Save", variant="primary")
            status_md = gr.Markdown("")
            latest_df = gr.Dataframe(headers=None, value=pd.DataFrame(), wrap=True, interactive=False,
                                     label="最近 20 筆紀錄", elem_id="latest_df")

            flat_inputs = [item_dd, *set_inputs, note_in]
            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, *flat_inputs],
                outputs=[status_md, item_dd, latest_df],
            )

        # ---- Records 分頁 ----
        with gr.TabItem("Records"):
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                q_item = gr.Textbox(label="Item 包含（關鍵字）")
            query_btn = gr.Button("🔎 Search")
            out_df = gr.Dataframe(headers=None, value=pd.DataFrame(), wrap=True, interactive=False, label="搜尋結果", elem_id="records_df")
            query_btn.click(search_records, inputs=[q_from, q_to, q_item], outputs=out_df)

        # ---- 你的教練（無說明文字） ----
        with gr.TabItem("你的教練"):
            chatbot = gr.Chatbot(height=420)
            user_in = gr.Textbox(placeholder="輸入你的問題，按 Enter 或點送出…", label="訊息")
            with gr.Row():
                send_btn = gr.Button("送出", variant="primary")
                clear_btn = gr.Button("清空")
            send_btn.click(coach_chat_stream, inputs=[chatbot, user_in], outputs=[chatbot, user_in])
            user_in.submit(coach_chat_stream, inputs=[chatbot, user_in], outputs=[chatbot, user_in])
            clear_btn.click(lambda: ([], ""), None, [chatbot, user_in], queue=False)

    gr.Markdown("""---
**Tips**
- Item 名稱可直接輸入新文字，下次會出現在下拉選單。
- 空白的數值欄會保持空白（不顯示 0）。
- Total Volume = ∑(kg × reps)。
""")

if __name__ == "__main__":
    ensure_records_csv()
    demo.launch()

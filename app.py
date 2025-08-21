"""
Gradio Workout Logger + 你的教練（Groq）— 單檔可執行 app.py
更新：
- 每次只記錄 1 個 Item（多次 Save 以追加）
- kg / reps 輸入預設為空（不顯示 0）
- Item 下拉會列出「雲端紀錄（Google Sheet）」與本地已知動作
- Save 時：
  * 直接以 Google Sheet 作為資料來源與顯示來源（<cloud record>）
  * 若內容與最近一次（同 item、同日期）完全相同 ⇒ 不儲存並停用 Save 按鈕
  * 若 10 分鐘內同 item 同日期有舊紀錄 ⇒ 以新內容覆蓋（刪舊寫新）
  * 每次存檔後，整表同步至 Google Sheet；本地 CSV 僅作為備援鏡像
- Records / 最新紀錄：直接讀取 Google Sheet；Note 欄位最寬

執行方式：
    pip install gradio pandas python-dateutil
    python app.py
授權方式（Google Sheet）：
    建議使用 Service Account，並將該帳戶的 email 分享為試算表的「可編輯」
    1) 設定環境變數 `gspread_service_json` 為 service account JSON 內容（整段字串）
       或設定 `GOOGLE_APPLICATION_CREDENTIALS` 指向本機 JSON 檔案
"""
from __future__ import annotations
import os
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime, date, timedelta
import hashlib

# 依需求：groq 安裝/匯入
try:
    from groq import Groq
except ImportError:
    os.system('pip install groq')
    from groq import Groq

# Google Sheets 相依
try:
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
except ImportError:
    os.system('pip install gspread gspread_dataframe')
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe

import gradio as gr
import pandas as pd

# ------------ 常數與檔案路徑 ------------
APP_TITLE = "Workout Logger"
RECORDS_CSV = Path("workout_records.csv")
ITEMS_JSON = Path("known_items.json")
NUM_ITEMS = 1            # 每次只記錄 1 個 Item
NUM_SETS = 5
WINDOW_MINUTES = 10      # 10 分鐘內可覆寫
SHEET_ID = "1qWH-FQKqAMLXdN2uV4fcLIk5URRjBwY7nELznZ352og"
SHEET_TITLE = "records"

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

# ------------ Google Sheets 工具 ------------

def _gs_client() -> Optional[gspread.Client]:
    try:
        sa_json = os.getenv("gspread_service_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            creds_dict = json.loads(sa_json)
            return gspread.service_account_from_dict(creds_dict)
        # 否則走 GOOGLE_APPLICATION_CREDENTIALS
        return gspread.service_account()
    except Exception:
        return None


def _open_or_create_ws(client: gspread.Client):
    sh = client.open_by_key(SHEET_ID)
    try:
        ws = sh.worksheet(SHEET_TITLE)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=SHEET_TITLE, rows=1000, cols=30)
    # 確保表頭
    ensure_records_header(ws)
    return ws


def ensure_records_header(ws):
    cols = ["date", "item"]
    for s in range(1, NUM_SETS+1):
        cols += [f"set{s}_kg", f"set{s}_reps"]
    cols += ["note", "total_volume_kg", "created_at"]
    try:
        first_row = ws.row_values(1)
    except Exception:
        first_row = []
    if first_row != cols:
        ws.clear()
        ws.update([cols])


def read_cloud_df() -> Optional[pd.DataFrame]:
    client = _gs_client()
    if not client:
        return None
    try:
        ws = _open_or_create_ws(client)
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        # 移除全空列；確保欄位名
        df = df.dropna(how='all')
        if df.empty:
            # 建立空 DataFrame 但有正確欄位
            cols = ["date", "item"]
            for s in range(1, NUM_SETS+1):
                cols += [f"set{s}_kg", f"set{s}_reps"]
            cols += ["note", "total_volume_kg", "created_at"]
            df = pd.DataFrame(columns=cols)
        # 轉字串型態以避免 NaN 問題（除數值欄）
        return df
    except Exception:
        return None


def write_cloud_df(df: pd.DataFrame) -> bool:
    client = _gs_client()
    if not client:
        return False
    try:
        ws = _open_or_create_ws(client)
        ws.clear()
        set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
        return True
    except Exception:
        return False

# ------------ 本地 CSV 備援 ------------

def ensure_records_csv():
    if not RECORDS_CSV.exists():
        cols = ["date", "item"]
        for s in range(1, NUM_SETS+1):
            cols += [f"set{s}_kg", f"set{s}_reps"]
        cols += ["note", "total_volume_kg", "created_at"]
        pd.DataFrame(columns=cols).to_csv(RECORDS_CSV, index=False, encoding="utf-8")


def load_local_df() -> pd.DataFrame:
    ensure_records_csv()
    try:
        return pd.read_csv(RECORDS_CSV)
    except Exception:
        return pd.DataFrame()


def write_local_df(df: pd.DataFrame):
    df.to_csv(RECORDS_CSV, index=False, encoding="utf-8")


# ------------ 輔助：來源優先雲端 ------------

def load_records_df() -> pd.DataFrame:
    df = read_cloud_df()
    if df is not None:
        return df
    return load_local_df()


def save_records_df(df: pd.DataFrame) -> bool:
    ok_cloud = write_cloud_df(df)
    write_local_df(df)
    return ok_cloud


# ------------ 其他工具 ------------

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


def get_all_item_choices() -> List[str]:
    seen: List[str] = []
    # 從雲端讀
    df = read_cloud_df()
    if df is not None and not df.empty and "item" in df.columns:
        counts = df["item"].dropna().astype(str).str.strip().value_counts()
        seen += [x for x in counts.index.tolist() if x]
    else:
        # 從本地備援
        if RECORDS_CSV.exists():
            try:
                df_local = pd.read_csv(RECORDS_CSV)
                if "item" in df_local.columns:
                    counts = df_local["item"].dropna().astype(str).str.strip().value_counts()
                    seen += [x for x in counts.index.tolist() if x]
            except Exception:
                pass
    # 加上 JSON known
    for it in load_known_items():
        if it and it not in seen:
            seen.append(it)
    return seen


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


def hash_entry(row: dict) -> str:
    key = json.dumps({k: row.get(k) for k in [
        "date", "item",
        *[f"set{i}_kg" for i in range(1, NUM_SETS+1)],
        *[f"set{i}_reps" for i in range(1, NUM_SETS+1)],
        "note"
    ]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ------------ 儲存動作（含 10 分鐘覆寫 & 重複檢查） ------------

def save_button_clicked(date_str: str, *flat_inputs):
    # 解析日期
    try:
        dt = pd.to_datetime(date_str).date()
    except Exception:
        return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), pd.DataFrame(), gr.update()

    # 展平：單一 item
    block_size = 1 + (NUM_SETS * 2) + 1
    chunk = list(flat_inputs[:block_size])
    item_name = (chunk[0] or "").strip()
    if not item_name:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), pd.DataFrame(), gr.update()

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
    now = datetime.now()
    new_row = {
        "date": dt.isoformat(),
        "item": item_name,
        **sets_kv,
        "note": note,
        "total_volume_kg": total_volume,
        "created_at": now.isoformat(timespec="seconds"),
    }
    new_hash = hash_entry(new_row)

    # 載入來源（優先雲端）
    df = load_records_df()

    # 篩同日同 item 最近一筆
    idx_recent = None
    recent_row = None
    if not df.empty:
        try:
            df_tmp = df.copy()
            if "created_at" in df_tmp.columns:
                df_tmp["created_at_dt"] = pd.to_datetime(df_tmp["created_at"], errors="coerce")
            else:
                df_tmp["created_at_dt"] = pd.NaT
            mask = (df_tmp["date"].astype(str) == new_row["date"]) & (df_tmp["item"].astype(str) == new_row["item"])
            df_same = df_tmp[mask].sort_values("created_at_dt", ascending=False)
            if not df_same.empty:
                idx_recent = df_same.index[0]
                recent_row = df.loc[idx_recent].to_dict()
        except Exception:
            pass

    # 若內容未變更：不儲存，並停用 Save
    if recent_row is not None:
        recent_hash = hash_entry(recent_row)
        if recent_hash == new_hash:
            merged_choices = get_all_item_choices()
            latest = load_records_df()
            if not latest.empty and "note" in latest.columns:
                cols = [c for c in latest.columns if c != "note"] + ["note"]
                latest = latest[cols]
            return ("內容未變更：未儲存。", gr.update(choices=merged_choices), latest.tail(20), gr.update(interactive=False))

    # 若 10 分鐘內有舊紀錄：覆寫（刪舊寫新）
    replaced = False
    if recent_row is not None:
        try:
            t_recent = pd.to_datetime(recent_row.get("created_at"), errors="coerce")
            if pd.notna(t_recent) and (now - t_recent.to_pydatetime()) <= timedelta(minutes=WINDOW_MINUTES):
                # 刪除舊行
                df = df.drop(index=idx_recent)
                replaced = True
        except Exception:
            pass

    # 追加新行
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # 讓 note 放最後一欄
    if "note" in df.columns:
        cols = [c for c in df.columns if c != "note"] + ["note"]
        df = df[cols]

    # 同步寫回雲端與本地
    save_records_df(df)

    # 更新已知 item 清單
    known = load_known_items()
    if item_name not in known:
        known.append(item_name)
        save_known_items(known)

    merged_choices = get_all_item_choices()
    latest = load_records_df()
    if not latest.empty and "note" in latest.columns:
        cols = [c for c in latest.columns if c != "note"] + ["note"]
        latest = latest[cols]

    msg = ("已覆寫最近 10 分鐘內的舊紀錄。" if replaced else "已儲存 1 筆。") + f"（日期：{dt.isoformat()}）"
    return (msg, gr.update(choices=merged_choices), latest.tail(20), gr.update(interactive=True))


# ------------ Records 搜尋（直接讀雲端，失敗則備援） ------------

def search_records(date_from: str, date_to: str, item_filter: str):
    df = load_records_df()

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
        # created_at 由新到舊
        try:
            df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
            df = df.sort_values(["date", "created_at_dt"], ascending=[False, False])
            df = df.drop(columns=["created_at_dt"], errors="ignore")
        except Exception:
            pass
        # 讓 note 放最後一欄
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
    gr.Markdown("""# 🏋️‍♂️ Workout Logger + 🤖 你的教練
快速記錄重量訓練與查詢歷史。""")

    with gr.Tabs():
        # ---- Log 分頁（單一 Item） ----
        with gr.TabItem("Log"):
            today_str = date.today().isoformat()
            date_in = gr.Textbox(value=today_str, label="Date (YYYY-MM-DD)")

            # 選單：合併雲端與已知
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
            latest_df = gr.Dataframe(headers=None, value=load_records_df().tail(20) if not load_records_df().empty else pd.DataFrame(),
                                     wrap=True, interactive=False, label="最近 20 筆紀錄", elem_id="latest_df")

            flat_inputs = [item_dd, *set_inputs, note_in]
            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, *flat_inputs],
                outputs=[status_md, item_dd, latest_df, save_btn],
            )

        # ---- Records 分頁 ----
        with gr.TabItem("Records"):
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                q_item = gr.Textbox(label="Item 包含（關鍵字）")
            query_btn = gr.Button("🔎 Search")
            out_df = gr.Dataframe(headers=None, value=load_records_df(), wrap=True, interactive=False, label="搜尋結果", elem_id="records_df")
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
    # 建立本地備援檔
    if not RECORDS_CSV.exists():
        ensure_records_csv()
    demo.launch()

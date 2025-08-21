"""
Gradio Workout Logger + 你的教練（Groq）— 單檔可執行 app.py（雲端修正版）
重點更新：
- 直接使用 Google Sheet 做資料來源與顯示來源（<cloud record>）。
- 自動偵測分頁名稱：優先 `SHEET_TITLE`（環境變數）→ `records` → `record` → 第一個分頁。
- UI 顯示 Cloud 連線狀態、目標分頁名稱與目前行數；Save 後訊息會顯示雲端是否成功與總列數。
- 10 分鐘內同日期+同 item 覆寫、內容相同不再重存（並暫時停用 Save）。

執行：
    pip install -r requirements.txt  # 或直接 pip install gradio pandas python-dateutil groq gspread gspread_dataframe google-auth google-auth-oauthlib
    python app.py
環境變數：
    groq_key=...                 # Groq API Key
    gspread_service_json=...     # 貼整段 Service Account JSON（或使用 GOOGLE_APPLICATION_CREDENTIALS 指向檔案）
    SHEET_TITLE=records          # 可選，指定要用的 worksheet 名稱
"""
from __future__ import annotations
import os, json, hashlib
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, date, timedelta

# ---- 依需求：groq 安裝/匯入 ----
try:
    from groq import Groq
except ImportError:
    os.system('pip install groq')
    from groq import Groq

# ---- Google Sheets 相依 ----
try:
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
except ImportError:
    os.system('pip install gspread gspread_dataframe google-auth google-auth-oauthlib')
    import gspread
    from gspread_dataframe import set_with_dataframe, get_as_dataframe

import gradio as gr
import pandas as pd

# ------------ 常數與檔案路徑 ------------
APP_TITLE = "Workout Logger"
RECORDS_CSV = Path("workout_records.csv")  # 本地備援
ITEMS_JSON = Path("known_items.json")
NUM_SETS = 5
WINDOW_MINUTES = 10      # 10 分鐘內可覆寫
SHEET_ID = "1qWH-FQKqAMLXdN2uV4fcLIk5URRjBwY7nELznZ352og"
SHEET_TITLE_ENV = os.getenv("SHEET_TITLE", "records")  # 可用環境變數覆寫

# ------------ Groq（教練機器人）設定 ------------
GROQ_API_KEY = os.getenv("groq_key")
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

# 追蹤雲端狀態
CLOUD_LAST_ERROR = ""
CLOUD_WS_TITLE = None

# ------------ Google Sheets 工具 ------------

def _gs_client() -> Optional[gspread.Client]:
    try:
        sa_json = os.getenv("gspread_service_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            creds_dict = json.loads(sa_json)
            return gspread.service_account_from_dict(creds_dict)
        return gspread.service_account()  # 走 GOOGLE_APPLICATION_CREDENTIALS
    except Exception:
        return None


def _get_target_ws(sh: gspread.Spreadsheet) -> gspread.Worksheet:
    """
    目標 worksheet 決策：
    1) SHEET_TITLE_ENV
    2) 'records'
    3) 'record'
    4) 第一個現有分頁
    若都沒有，建立 SHEET_TITLE_ENV。
    """
    global CLOUD_WS_TITLE
    # 先嘗試直接取
    preferred = [SHEET_TITLE_ENV, "records", "record"]
    titles = [ws.title for ws in sh.worksheets()]
    for name in preferred:
        if name in titles:
            CLOUD_WS_TITLE = name
            return sh.worksheet(name)
    # 沒找到就用第一個
    if titles:
        CLOUD_WS_TITLE = titles[0]
        return sh.worksheet(titles[0])
    # 若竟然沒有分頁，建立一個
    CLOUD_WS_TITLE = SHEET_TITLE_ENV
    return sh.add_worksheet(title=SHEET_TITLE_ENV, rows=1000, cols=30)


def _open_or_create_ws(client: gspread.Client):
    sh = client.open_by_key(SHEET_ID)
    ws = _get_target_ws(sh)
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
    """改用 get_all_values 讀取，避免 gspread_dataframe 造成的空白列問題。"""
    global CLOUD_LAST_ERROR
    client = _gs_client()
    if not client:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return None
    try:
        ws = _open_or_create_ws(client)
        rows = ws.get_all_values()  # 2D list
        if not rows:
            return None
        header = rows[0] if rows else []
        data = rows[1:] if len(rows) > 1 else []
        if not header:
            return None
        if not data:
            # 回傳空 DF 但保留欄位
            df = pd.DataFrame(columns=header)
        else:
            df = pd.DataFrame(data, columns=header)
        # 對數值欄嘗試轉型，空白保持空字串
        for s in range(1, NUM_SETS+1):
            for sub in ("kg", "reps"):
                col = f"set{s}_{sub}"
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
        if "total_volume_kg" in df.columns:
            df["total_volume_kg"] = pd.to_numeric(df["total_volume_kg"], errors="coerce")
        CLOUD_LAST_ERROR = ""
        return df
    except Exception as e:
        CLOUD_LAST_ERROR = f"讀取雲端失敗：{e}"
        return None
    try:
        ws = _open_or_create_ws(client)
        df = get_as_dataframe(ws, evaluate_formulas=True, header=0)
        df = df.dropna(how='all')
        if df.empty:
            cols = ["date", "item"]
            for s in range(1, NUM_SETS+1):
                cols += [f"set{s}_kg", f"set{s}_reps"]
            cols += ["note", "total_volume_kg", "created_at"]
            df = pd.DataFrame(columns=cols)
        CLOUD_LAST_ERROR = ""
        return df
    except Exception as e:
        CLOUD_LAST_ERROR = f"讀取雲端失敗：{e}"
        return None


def write_cloud_df(df: pd.DataFrame) -> Tuple[bool, int]:
    """不用 gspread_dataframe，直接 ws.update(range_name='A1', values=...).
    另外強制欄位順序，並把 NaN 轉為空字串，避免整列被視為空白。"""
    global CLOUD_LAST_ERROR
    client = _gs_client()
    if not client:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return False, 0
    try:
        ws = _open_or_create_ws(client)
        # 欄位順序
        cols = ["date", "item"] + sum(([f"set{s}_kg", f"set{s}_reps"] for s in range(1, NUM_SETS+1)), []) + ["note", "total_volume_kg", "created_at"]
        out_df = df.copy()
        # 若缺欄位補空、並重排
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df = out_df[cols]
        out_df = out_df.fillna("")
        # 轉成純 Python 基本型別
        raw_values = out_df.values.tolist()
        values: list[list] = []
        for row in raw_values:
            new_row = []
            for x in row:
                if isinstance(x, (int, float, str)):
                    new_row.append(x)
                else:
                    new_row.append(str(x) if x is not None else "")
            values.append(new_row)
        # 清空+寫入
        ws.clear()
        ws.update(range_name="A1", values=[cols] + values)
        # 調整大小
        try:
            ws.resize(rows=max(2, len(values) + 1), cols=len(cols))
        except Exception:
            pass
        CLOUD_LAST_ERROR = ""
        return True, len(values)
    except Exception as e:
        CLOUD_LAST_ERROR = f"寫入雲端失敗：{e}"
        return False, 0
    try:
        ws = _open_or_create_ws(client)
        # 準備資料：將 NaN 轉成空字串，確保會寫出列
        out_df = df.copy()
        out_df = out_df.fillna("")
        header = list(out_df.columns)
        values = out_df.values.tolist()
        ws.clear()
        ws.update("A1", [header] + values)
        # 最後調整表格大小
        try:
            ws.resize(rows=max(2, len(values) + 1), cols=len(header))
        except Exception:
            pass
        CLOUD_LAST_ERROR = ""
        return True, len(values)
    except Exception as e:
        CLOUD_LAST_ERROR = f"寫入雲端失敗：{e}"
        return False, 0
    try:
        ws = _open_or_create_ws(client)
        ws.clear()
        set_with_dataframe(ws, df, include_index=False, include_column_header=True, resize=True)
        CLOUD_LAST_ERROR = ""
        # 重新抓一次行數
        total_rows = len(df.index)
        return True, total_rows
    except Exception as e:
        CLOUD_LAST_ERROR = f"寫入雲端失敗：{e}"
        return False, 0

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


# ------------ 優先雲端 ------------

def load_records_df() -> pd.DataFrame:
    df = read_cloud_df()
    if df is not None:
        return df
    return load_local_df()


def save_records_df(df: pd.DataFrame) -> Tuple[bool, int]:
    ok_cloud, total_rows = write_cloud_df(df)
    write_local_df(df)
    return ok_cloud, total_rows


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
    df = read_cloud_df()
    if df is not None and not df.empty and "item" in df.columns:
        counts = df["item"].dropna().astype(str).str.strip().value_counts()
        seen += [x for x in counts.index.tolist() if x]
    else:
        if RECORDS_CSV.exists():
            try:
                df_local = pd.read_csv(RECORDS_CSV)
                if "item" in df_local.columns:
                    counts = df_local["item"].dropna().astype(str).str.strip().value_counts()
                    seen += [x for x in counts.index.tolist() if x]
            except Exception:
                pass
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


# ------------ 儲存（含覆寫與重複判斷） ------------

def save_button_clicked(date_str: str, item_name: str,
                        set1kg, set1reps, set2kg, set2reps, set3kg, set3reps, set4kg, set4reps, set5kg, set5reps,
                        note: str):
    # 解析日期
    try:
        dt = pd.to_datetime(date_str).date()
    except Exception:
        return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), pd.DataFrame(), gr.update()

    item_name = (item_name or "").strip()
    if not item_name:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), pd.DataFrame(), gr.update()

    # 解析數值
    def to_f(x):
        return None if x in ("", None) else float(x)
    def to_i(x):
        return None if x in ("", None) else int(x)

    kg_vals = [to_f(set1kg), to_f(set2kg), to_f(set3kg), to_f(set4kg), to_f(set5kg)]
    reps_vals = [to_i(set1reps), to_i(set2reps), to_i(set3reps), to_i(set4reps), to_i(set5reps)]

    sets_kv = {}
    for idx, (kg, rp) in enumerate(zip(kg_vals, reps_vals), start=1):
        sets_kv[f"set{idx}_kg"] = kg
        sets_kv[f"set{idx}_reps"] = rp

    total_volume = compute_total_volume(kg_vals, reps_vals)
    now = datetime.now()
    new_row = {
        "date": dt.isoformat(),
        "item": item_name,
        **sets_kv,
        "note": note or "",
        "total_volume_kg": total_volume,
        "created_at": now.isoformat(timespec="seconds"),
    }
    new_hash = hash_entry(new_row)

    # 讀現有
    df = load_records_df()

    # 找最近同日+同 item
    idx_recent = None
    recent_row = None
    if not df.empty:
        try:
            df_tmp = df.copy()
            df_tmp["created_at_dt"] = pd.to_datetime(df_tmp.get("created_at"), errors="coerce")
            mask = (df_tmp["date"].astype(str) == new_row["date"]) & (df_tmp["item"].astype(str) == new_row["item"])
            df_same = df_tmp[mask].sort_values("created_at_dt", ascending=False)
            if not df_same.empty:
                idx_recent = df_same.index[0]
                recent_row = df.loc[idx_recent].to_dict()
        except Exception:
            pass

    if recent_row is not None and hash_entry(recent_row) == new_hash:
        merged_choices = get_all_item_choices()
        latest = load_records_df()
        if not latest.empty and "note" in latest.columns:
            cols = [c for c in latest.columns if c != "note"] + ["note"]
            latest = latest[cols]
        return ("內容未變更：未儲存。", gr.update(choices=merged_choices), latest.tail(20), gr.update(interactive=False))

    replaced = False
    if recent_row is not None:
        try:
            t_recent = pd.to_datetime(recent_row.get("created_at"), errors="coerce")
            if pd.notna(t_recent) and (now - t_recent.to_pydatetime()) <= timedelta(minutes=WINDOW_MINUTES):
                df = df.drop(index=idx_recent)
                replaced = True
        except Exception:
            pass

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    if "note" in df.columns:
        cols = [c for c in df.columns if c != "note"] + ["note"]
        df = df[cols]

    ok_cloud, total_rows = save_records_df(df)

    msg = ("已覆寫最近 10 分鐘內的舊紀錄。" if replaced else "已儲存 1 筆。") + f"（日期：{dt.isoformat()}）"
    if ok_cloud:
        msg += f"｜雲端同步✅｜分頁：{CLOUD_WS_TITLE}｜總列數：{total_rows}"
    else:
        extra = f"（{CLOUD_LAST_ERROR}）" if CLOUD_LAST_ERROR else ""
        msg += f"｜雲端同步❌ {extra}"

    known = load_known_items()
    if item_name not in known:
        known.append(item_name)
        save_known_items(known)

    merged_choices = get_all_item_choices()
    latest = load_records_df()
    if not latest.empty and "note" in latest.columns:
        cols = [c for c in latest.columns if c != "note"] + ["note"]
        latest = latest[cols]

    return (msg, gr.update(choices=merged_choices), latest.tail(20), gr.update(interactive=True))


# ------------ 搜尋（直接讀雲端，失敗則備援） ------------

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
        try:
            df["created_at_dt"] = pd.to_datetime(df["created_at"], errors="coerce")
            df = df.sort_values(["date", "created_at_dt"], ascending=[False, False])
            df = df.drop(columns=["created_at_dt"], errors="ignore")
        except Exception:
            pass
        if "note" in df.columns:
            cols = [c for c in df.columns if c != "note"] + ["note"]
            df = df[cols]
    return df


# ------------ 教練機器人（串流） ------------

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

# ------------ 介面 ------------
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("""# 🏋️‍♂️ Workout Logger + 🤖 你的教練
快速記錄重量訓練與查詢歷史。""")

    # 雲端狀態提示（包含目標分頁）
    _df_probe = read_cloud_df()
    target = CLOUD_WS_TITLE or SHEET_TITLE_ENV
    cloud_status = "已連線至雲端試算表 ✅" if _df_probe is not None else f"未連線至雲端（改用本機備援）❌  {CLOUD_LAST_ERROR}"
    rows_info = f"，分頁：{target}，目前列數：{len(_df_probe) if _df_probe is not None else 0}"
    gr.Markdown(f"**Cloud**：{cloud_status}{rows_info}")

    with gr.Tabs():
        # ---- Log ----
        with gr.TabItem("Log"):
            today_str = date.today().isoformat()
            date_in = gr.Textbox(value=today_str, label="Date (YYYY-MM-DD)")

            item_choices = get_all_item_choices()
            gr.Markdown("### Item")
            item_dd = gr.Dropdown(choices=item_choices, allow_custom_value=True, value=None, label="Item 名稱")

            with gr.Row():
                set1kg = gr.Number(label="Set 1 — kg", precision=2, value=None, placeholder="kg")
                set1rp = gr.Number(label="Set 1 — reps", precision=0, value=None, placeholder="reps")
            with gr.Row():
                set2kg = gr.Number(label="Set 2 — kg", precision=2, value=None, placeholder="kg")
                set2rp = gr.Number(label="Set 2 — reps", precision=0, value=None, placeholder="reps")
            with gr.Row():
                set3kg = gr.Number(label="Set 3 — kg", precision=2, value=None, placeholder="kg")
                set3rp = gr.Number(label="Set 3 — reps", precision=0, value=None, placeholder="reps")
            with gr.Row():
                set4kg = gr.Number(label="Set 4 — kg", precision=2, value=None, placeholder="kg")
                set4rp = gr.Number(label="Set 4 — reps", precision=0, value=None, placeholder="reps")
            with gr.Row():
                set5kg = gr.Number(label="Set 5 — kg", precision=2, value=None, placeholder="kg")
                set5rp = gr.Number(label="Set 5 — reps", precision=0, value=None, placeholder="reps")

            note_in = gr.Textbox(label="Note", placeholder="RPE、感覺、下次調整…")

            save_btn = gr.Button("💾 Save", variant="primary")
            status_md = gr.Markdown("")
            current_df = load_records_df()
            latest_df = gr.Dataframe(headers=None, value=current_df.tail(20) if not current_df.empty else pd.DataFrame(),
                                     wrap=True, interactive=False, label="最近 20 筆紀錄", elem_id="latest_df")

            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, item_dd,
                        set1kg, set1rp, set2kg, set2rp, set3kg, set3rp, set4kg, set4rp, set5kg, set5rp,
                        note_in],
                outputs=[status_md, item_dd, latest_df, save_btn],
            )

        # ---- Records ----
        with gr.TabItem("Records"):
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                q_item = gr.Textbox(label="Item 包含（關鍵字）")
            query_btn = gr.Button("🔎 Search")
            out_df = gr.Dataframe(headers=None, value=load_records_df(), wrap=True, interactive=False, label="搜尋結果", elem_id="records_df")
            query_btn.click(search_records, inputs=[q_from, q_to, q_item], outputs=out_df)

        # ---- 你的教練 ----
        with gr.TabItem("你的教練"):
            chatbot = gr.Chatbot(height=420, type="messages")
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
    if not RECORDS_CSV.exists():
        ensure_records_csv()
    demo.launch()

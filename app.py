"""
Gradio Workout Logger + 你的教練（Groq）— app.py（穩定版）
- Records 的 Item 改為「下拉＋可輸入」，並新增「🔄 更新選單」按鈕
- 修正所有可能的未結束字串（unterminated string literal）
- 保留功能：雲端同步、10 分鐘覆寫、Item 下拉記憶、行動版五行顯示、Note 另行＋台北時區時間（12h、不補 0）、教練串流可讀取最近紀錄
"""
from __future__ import annotations
import os, json, hashlib, html, math
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, date, timedelta, timezone

# 依賴
import gradio as gr
import pandas as pd


# Groq
try:
    from groq import Groq
except ImportError:
    os.system('pip install groq')
    from groq import Groq

# Google Sheets
try:
    import gspread
except ImportError:
    os.system('pip install gspread google-auth google-auth-oauthlib')
    import gspread

# ---------------- 常數 ----------------
APP_TITLE = "Workout Logger"
APP_VERSION = "v1.2"  # Update: 移除時區註記，恢復簡潔
RECORDS_CSV = Path("workout_records.csv")
ITEMS_JSON = Path("known_items.json")
NUM_SETS = 5
WINDOW_MINUTES = 10
SHEET_ID = "1qWH-FQKqAMLXdN2uV4fcLIk5URRjBwY7nELznZ352og"
SHEET_TITLE_ENV = os.getenv("SHEET_TITLE", "records")

# Groq 設定
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

# 雲端狀態
CLOUD_LAST_ERROR = ""
CLOUD_WS_TITLE: Optional[str] = None

# ---------------- Google Sheets 工具 ----------------

def _gs_client() -> Optional[gspread.Client]:
    try:
        sa_json = os.getenv("gspread_service_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            return gspread.service_account_from_dict(json.loads(sa_json))
        return gspread.service_account()
    except Exception:
        return None

def _get_target_ws(sh: gspread.Spreadsheet) -> gspread.Worksheet:
    global CLOUD_WS_TITLE
    prefer = [SHEET_TITLE_ENV, "records", "record"]
    try:
        titles = [ws.title for ws in sh.worksheets()]
    except Exception:
        titles = []
    for t in prefer:
        if t in titles:
            CLOUD_WS_TITLE = t
            return sh.worksheet(t)
    if titles:
        CLOUD_WS_TITLE = titles[0]
        return sh.worksheet(titles[0])
    CLOUD_WS_TITLE = SHEET_TITLE_ENV
    return sh.add_worksheet(title=SHEET_TITLE_ENV, rows=1000, cols=30)

def ensure_records_header(ws: gspread.Worksheet):
    cols = ["date", "item"]
    for s in range(1, NUM_SETS + 1):
        cols += [f"set{s}_kg", f"set{s}_reps"]
    cols += ["note", "total_volume_kg", "created_at"]
    try:
        header = ws.row_values(1)
    except Exception:
        header = []
    if header != cols:
        ws.clear()
        ws.update(range_name="A1", values=[cols])

def _open_ws(client: gspread.Client) -> gspread.Worksheet:
    sh = client.open_by_key(SHEET_ID)
    ws = _get_target_ws(sh)
    ensure_records_header(ws)
    return ws

def read_cloud_df() -> Optional[pd.DataFrame]:
    global CLOUD_LAST_ERROR
    cli = _gs_client()
    if not cli:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return None
    try:
        ws = _open_ws(cli)
        rows = ws.get_all_values()
        if not rows:
            return None
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        df = pd.DataFrame(data, columns=header) if data else pd.DataFrame(columns=header)
        # 數值欄轉型
        for s in range(1, NUM_SETS + 1):
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

def write_cloud_df(df: pd.DataFrame) -> Tuple[bool, int]:
    global CLOUD_LAST_ERROR
    cli = _gs_client()
    if not cli:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return False, 0
    try:
        ws = _open_ws(cli)
        cols = ["date", "item"] + sum(([f"set{s}_kg", f"set{s}_reps"] for s in range(1, NUM_SETS + 1)), []) + ["note", "total_volume_kg", "created_at"]
        out = df.copy()
        for c in cols:
            if c not in out.columns:
                out[c] = ""
        out = out[cols].fillna("")
        values = [[x if isinstance(x, (int, float, str)) else ("" if x is None else str(x)) for x in row] for row in out.values.tolist()]
        ws.clear()
        ws.update(range_name="A1", values=[cols] + values)
        try:
            ws.resize(rows=max(2, len(values) + 1), cols=len(cols))
        except Exception:
            pass
        CLOUD_LAST_ERROR = ""
        return True, len(values)
    except Exception as e:
        CLOUD_LAST_ERROR = f"寫入雲端失敗：{e}"
        return False, 0

# 本地 CSV 備援
def ensure_records_csv():
    if not RECORDS_CSV.exists():
        cols = ["date", "item"]
        for s in range(1, NUM_SETS + 1):
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

# 封裝：讀寫優先雲端
def load_records_df() -> pd.DataFrame:
    df = read_cloud_df()
    return df if df is not None else load_local_df()

def save_records_df(df: pd.DataFrame) -> Tuple[bool, int]:
    ok, rows = write_cloud_df(df)
    write_local_df(df)
    return ok, rows

def cloud_status_line() -> str:
    df = read_cloud_df()
    target = CLOUD_WS_TITLE or SHEET_TITLE_ENV
    ok = df is not None
    count = 0 if df is None else len(df)
    status = "已連線至雲端試算表 ✅" if ok else f"未連線至雲端（改用本機備援）❌  {CLOUD_LAST_ERROR}"
    return f"**Cloud**：{status}，分頁：{target}，目前列數：{count}"

# ---------------- 小工具 ----------------
def load_known_items() -> List[str]:
    if ITEMS_JSON.exists():
        try:
            return json.loads(ITEMS_JSON.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

# 新增：取得台北時間 (UTC+8) 的 Helper
def get_now_tpe() -> datetime:
    return datetime.now(timezone(timedelta(hours=8)))

def save_known_items(items: List[str]):
    uniq: List[str] = []
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

def _fmt_num(n):
    if n in (None, "", "nan", "NaN", "NAN"):
        return ""
    try:
        f = float(n)
        if math.isnan(f):
            return ""
        return str(int(f)) if float(f).is_integer() else str(f)
    except Exception:
        return str(n)

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
        *[f"set{i}_kg" for i in range(1, NUM_SETS + 1)],
        *[f"set{i}_reps" for i in range(1, NUM_SETS + 1)],
        "note",
    ]}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

# 時間顯示（台北時區、上午/下午/晚上、12 小時制、不補 0）
def to_tpe_time_str(created_at: str) -> str:
    if not created_at:
        return ""
    try:
        # Update: 嘗試解析為 Naive Time 並視為 TPE，或處理舊有 Aware Time
        ts = pd.to_datetime(created_at)
        if ts.tzinfo is None:
             # 若無時區資訊，預設為台北時間 (TPE is UTC+8)
            ts = ts.tz_localize(timezone(timedelta(hours=8)))
        else:
            # 若有時區資訊，轉換至 TPE
            ts = ts.tz_convert(timezone(timedelta(hours=8)))
    except Exception:
        return ""
    
    try:
        # 因為已經是 TPE aware
        h24 = int(ts.strftime("%H"))
        m = ts.strftime("%M")
        period = "上午"
        if 12 <= h24 <= 17:
            period = "下午"
        elif 18 <= h24 <= 23:
            period = "晚上"
        h12 = ((h24 - 1) % 12) + 1
        return f"{period} {h12}:{m}"
    except Exception:
        return ""

# ---------------- HTML 呈現（五行 + Note 另起一行） ----------------
def df_to_html_compact5(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<div class='records-empty'>目前沒有紀錄</div>"
    if "note" in df.columns:
        cols = [c for c in df.columns if c != "note"] + ["note"]
        df = df[cols]

    cards: List[str] = []
    for _, row in df.iterrows():
        date_s = row.get("date", "") or ""
        item_s = row.get("item", "") or ""
        note_s = row.get("note", "") or ""
        total_s = _fmt_num(row.get("total_volume_kg", ""))
        created_s = row.get("created_at", "") or ""
        time_tpe = to_tpe_time_str(created_s)

        set_lines: List[str] = []
        for i in range(1, NUM_SETS + 1):
            kg = _fmt_num(row.get(f"set{i}_kg", ""))
            rp = _fmt_num(row.get(f"set{i}_reps", ""))
            kg_txt = (kg + "kg") if kg else ""
            rp_txt = (rp + "r") if rp else ""
            set_lines.append(
                f"<tr><td class='sidx'>{i}</td><td class='kg nowrap'>{kg_txt}</td><td class='r nowrap'>{rp_txt}</td></tr>"
            )
        lines_html = "".join(set_lines)
        note_html = (
            f"<tr class='note-row'><td class='note-cell' colspan='3'>"
            f"<b>Note：</b>{html.escape(str(note_s))}<span class='time'>（{html.escape(time_tpe)}）</span>"
            f"</td></tr>"
        )

        header_left = html.escape(str(date_s)) + " · " + html.escape(str(item_s))
        header_right = ("Σ " + html.escape(str(total_s)) + " kg") if total_s else ""

        card_html = (
            "<div class='rec-card'>"
            "<div class='rec-header'>"
            f"<div class='left nowrap'>{header_left}</div>"
            f"<div class='right nowrap'>{header_right}</div>"
            "</div>"
            "<table class='rec-sets'><tbody>"
            f"{lines_html}{note_html}"
            "</tbody></table>"
            "</div>"
        )
        cards.append(card_html)

    return "<div class='records-cards'>" + "".join(cards) + "</div>"

# ---------------- 儲存邏輯 ----------------
def save_button_clicked(date_str: str, item_name: str,
                        set1kg, set1reps, set2kg, set2reps, set3kg, set3reps, set4kg, set4reps, set5kg, set5reps,
                        note: str):
    # 日期（空白→今天，改用台北時間）
    if not date_str or not str(date_str).strip():
        dt = get_now_tpe().date()
    else:
        try:
            dt = pd.to_datetime(date_str).date()
        except Exception:
            return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), "", gr.update(), cloud_status_line()

    item_name = (item_name or "").strip()
    if not item_name:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), "", gr.update(), cloud_status_line()

    to_f = lambda x: None if x in ("", None) else float(x)
    to_i = lambda x: None if x in ("", None) else int(x)

    kg_vals = [to_f(set1kg), to_f(set2kg), to_f(set3kg), to_f(set4kg), to_f(set5kg)]
    reps_vals = [to_i(set1reps), to_i(set2reps), to_i(set3reps), to_i(set4reps), to_i(set5reps)]

    sets_kv = {}
    for idx, (kg, rp) in enumerate(zip(kg_vals, reps_vals), start=1):
        sets_kv[f"set{idx}_kg"] = kg
        sets_kv[f"set{idx}_reps"] = rp

    total_volume = compute_total_volume(kg_vals, reps_vals)
    
    # Update: 建立時間改用台北時間，但儲存時不帶時區資訊 (Naive) 以符合需求
    now_tpe = get_now_tpe()
    created_at_str = now_tpe.strftime('%Y-%m-%dT%H:%M:%S')  # ISO format without offset

    new_row = {
        "date": dt.isoformat(),
        "item": item_name,
        **sets_kv,
        "note": note or "",
        "total_volume_kg": total_volume,
        "created_at": created_at_str,
    }

    new_hash = hash_entry(new_row)
    df = load_records_df()

    # 找同日同 item 最近一筆
    idx_recent = None
    recent_row = None
    if df is not None and not df.empty:
        try:
            tmp = df.copy()
            # 讀取時，確保 tmp["created_at_dt"] 為 TPE aware
            tmp["created_at_dt"] = pd.to_datetime(tmp.get("created_at"), errors="coerce")
            
            mask_naive = tmp["created_at_dt"].apply(lambda x: x.tzinfo is None if pd.notnull(x) else False)
            if mask_naive.any():
                tmp.loc[mask_naive, "created_at_dt"] = tmp.loc[mask_naive, "created_at_dt"].dt.tz_localize(timezone(timedelta(hours=8)))
            
            mask_aware = ~mask_naive
            if mask_aware.any():
                tmp.loc[mask_aware, "created_at_dt"] = tmp.loc[mask_aware, "created_at_dt"].dt.tz_convert(timezone(timedelta(hours=8)))

            same = (tmp["date"].astype(str) == new_row["date"]) & (tmp["item"].astype(str) == new_row["item"])
            same_df = tmp[same].sort_values("created_at_dt", ascending=False)
            if not same_df.empty:
                idx_recent = same_df.index[0]
                recent_row = df.loc[idx_recent].to_dict()
        except Exception:
            pass

    if recent_row is not None and hash_entry(recent_row) == new_hash:
        merged = get_all_item_choices()
        latest = load_records_df()
        latest_html = df_to_html_compact5(latest.tail(20)) if latest is not None and not latest.empty else ""
        return ("內容未變更：未儲存。", gr.update(choices=merged), latest_html, gr.update(interactive=False), cloud_status_line())

    # Update: 10 分鐘內重複儲存邏輯 -> 移除「所有」符合條件的舊紀錄（覆寫）
    replaced = False
    if df is not None and not df.empty:
        try:
            tmp = df.copy()
            # 確保舊資料時間欄位為 TPE aware (統一基準)
            tmp["created_at_dt"] = pd.to_datetime(tmp.get("created_at"), errors="coerce")
            
            mask_naive = tmp["created_at_dt"].apply(lambda x: x.tzinfo is None if pd.notnull(x) else False)
            if mask_naive.any():
                tmp.loc[mask_naive, "created_at_dt"] = tmp.loc[mask_naive, "created_at_dt"].dt.tz_localize(timezone(timedelta(hours=8)))
            
            mask_aware = ~mask_naive
            if mask_aware.any():
                tmp.loc[mask_aware, "created_at_dt"] = tmp.loc[mask_aware, "created_at_dt"].dt.tz_convert(timezone(timedelta(hours=8)))
            
            # now_tpe 已經是 TPE aware
            mask_target = (tmp["date"].astype(str) == new_row["date"]) & (tmp["item"].astype(str) == new_row["item"])
            mask_window = (now_tpe - tmp["created_at_dt"]) <= timedelta(minutes=WINDOW_MINUTES)
            
            indices_to_drop = tmp[mask_target & mask_window].index
            
            if not indices_to_drop.empty:
                df = df.drop(index=indices_to_drop)
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
        msg += f"｜雲端同步✅｜分頁：{CLOUD_WS_TITLE or SHEET_TITLE_ENV}｜總列數：{total_rows}"
    else:
        extra = f"（{CLOUD_LAST_ERROR}）" if CLOUD_LAST_ERROR else ""
        msg += f"｜雲端同步❌ {extra}"

    known = load_known_items()
    if item_name not in known:
        known.append(item_name)
        save_known_items(known)

    merged = get_all_item_choices()
    latest = load_records_df()
    latest_html = df_to_html_compact5(latest.tail(20)) if latest is not None and not latest.empty else ""
    return (msg, gr.update(choices=merged), latest_html, gr.update(interactive=True), cloud_status_line())

# ---------------- 搜尋 ----------------
def search_records(date_from: str, date_to: str, item_filter: str) -> pd.DataFrame:
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

def search_records_html(date_from: str, date_to: str, item_filter: str) -> str:
    return df_to_html_compact5(search_records(date_from, date_to, item_filter))

# ---------------- 教練上下文 ----------------
def _truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"

def make_coach_context(days: int = 60, max_items: int = 8, max_recent: int = 10) -> str:
    df = load_records_df()
    if df is None or df.empty:
        return "（目前沒有雲端紀錄）"
    f = df.copy()
    try:
        f["date_dt"] = pd.to_datetime(f["date"], errors="coerce")
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        f = f[f["date_dt"] >= cutoff]
    except Exception:
        pass
    if f.empty:
        return f"（最近 {days} 天沒有紀錄）"

    lines = [f"期間：最近 {days} 天"]
    try:
        vol = f.groupby("item", dropna=False)["total_volume_kg"].sum(min_count=1).sort_values(ascending=False)
    except Exception:
        vol = pd.Series(dtype=float)
    try:
        cnt = f["item"].value_counts()
    except Exception:
        cnt = pd.Series(dtype=int)
    try:
        last_date = f.groupby("item", dropna=False)["date"].max()
    except Exception:
        last_date = pd.Series(dtype=str)

    items = list(cnt.index[:max_items]) if not cnt.empty else f["item"].dropna().unique().tolist()[:max_items]
    for it in items:
        c = int(cnt.get(it, 0)) if not cnt.empty else 0
        v = vol.get(it, float('nan')) if not vol.empty else float('nan')
        v_txt = _fmt_num(v)
        ld = last_date.get(it, "") if not last_date.empty else ""
        lines.append(f"- {it}: 次數 {c}，總量 {v_txt} kg，最近 {ld}")

    try:
        f["created_at_dt"] = pd.to_datetime(f["created_at"], errors="coerce")
        recent = f.sort_values("created_at_dt", ascending=False).head(max_recent)
    except Exception:
        recent = f.tail(max_recent)

    lines.append("最近幾筆：")
    for _, r in recent.iterrows():
        parts: List[str] = []
        for i in range(1, NUM_SETS + 1):
            kg = _fmt_num(r.get(f"set{i}_kg"))
            rp = _fmt_num(r.get(f"set{i}_reps"))
            if kg and rp:
                parts.append(f"{kg}x{rp}")
        sets_txt = "/".join(parts)
        note_txt = _truncate(r.get("note", ""), 40)
        total_txt = _fmt_num(r.get("total_volume_kg"))
        lines.append(f"- {r.get('date','')} {r.get('item','')}: {sets_txt}；備註：{note_txt}；total={total_txt}kg")
    return "\n".join(lines)

# ---------------- 教練（串流） ----------------
def coach_chat_stream_ctx(history, user_msg: str, use_ctx: bool, ctx_days: int):
    msg = (user_msg or "").strip()
    if not msg:
        yield history, ""
        return
    if groq_client is None:
        bot = "（尚未設定環境變數 groq_key，請設定後重試。）"
        if isinstance(history, list) and (not history or isinstance(history[0], dict)):
            ui = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": bot}]
        else:
            ui = (history or []) + [[msg, bot]]
        yield ui, ""
        return

    sys_content = SYSTEM_PROMPT
    if use_ctx:
        try:
            ctx = make_coach_context(int(ctx_days))
        except Exception:
            ctx = make_coach_context()
        sys_content += "\n\n【學員近期紀錄摘要】\n" + ctx

    api_messages = [{"role": "system", "content": sys_content}]

    if isinstance(history, list) and history and isinstance(history[0], dict):
        for m in history:
            if m.get("role") in ("user", "assistant"):
                api_messages.append({"role": m.get("role"), "content": m.get("content", "")})
        ui_hist = history.copy()
    else:
        for u, b in (history or []):
            if u:
                api_messages.append({"role": "user", "content": u})
            if b:
                api_messages.append({"role": "assistant", "content": b})
        ui_hist = []
        for u, b in (history or []):
            if u:
                ui_hist.append({"role": "user", "content": u})
            if b:
                ui_hist.append({"role": "assistant", "content": b})

    api_messages.append({"role": "user", "content": msg})

    try:
        completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=api_messages,
            temperature=0.7,
            max_completion_tokens=512,
            top_p=1,
            stream=True,
            stop=None,
        )
        ui_hist = ui_hist + [{"role": "user", "content": msg}, {"role": "assistant", "content": ""}]
        acc = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                acc += delta
                ui_hist[-1]["content"] = acc
                yield ui_hist, ""
        return
    except Exception as e:
        ui_hist = ui_hist + [{"role": "user", "content": msg}, {"role": "assistant", "content": f"抱歉，Groq 呼叫失敗：{e}"}]
        yield ui_hist, ""

# ---------------- JavaScript (Rest Timer) ----------------
def get_rest_timer_js(elem_id):
    return f"""
    (x) => {{
        // 取得按鈕元素
        const btn = document.querySelector('#{elem_id} button') || document.querySelector('#{elem_id}');
        if (!btn) return;
        
        // 防止重複點擊
        if (btn.classList.contains('counting')) return;
        btn.classList.add('counting');
        
        let seconds = 120; // 倒數 120 秒 (2 分鐘)
        const originalText = "Rest";
        
        // 播放提示音 (Web Audio API 模擬拳擊鈴聲)
        const playSound = () => {{
            try {{
                const ctx = new (window.AudioContext || window.webkitAudioContext)();
                const t = ctx.currentTime;
                
                // 模擬鈴聲：混合兩個頻率
                const osc1 = ctx.createOscillator();
                const gain1 = ctx.createGain();
                osc1.connect(gain1);
                gain1.connect(ctx.destination);
                osc1.type = 'square'; // 方波較有穿透力
                osc1.frequency.setValueAtTime(600, t);
                gain1.gain.setValueAtTime(0.3, t);
                gain1.gain.exponentialRampToValueAtTime(0.001, t + 1.2);
                
                const osc2 = ctx.createOscillator();
                const gain2 = ctx.createGain();
                osc2.connect(gain2);
                gain2.connect(ctx.destination);
                osc2.type = 'sine';
                osc2.frequency.setValueAtTime(1000, t);
                gain2.gain.setValueAtTime(0.2, t);
                gain2.gain.exponentialRampToValueAtTime(0.001, t + 1.0);

                osc1.start(t);
                osc1.stop(t + 1.5);
                osc2.start(t);
                osc2.stop(t + 1.5);
            }} catch(e) {{
                console.error("Audio play failed", e);
            }}
        }};

        btn.innerText = seconds + "s";
        
        const timer = setInterval(() => {{
            seconds--;
            if (seconds > 0) {{
                btn.innerText = seconds + "s";
            }} else {{
                clearInterval(timer);
                playSound();
                btn.innerText = "Time's up";
                // 3秒後恢復 Rest
                setTimeout(() => {{
                    btn.innerText = originalText;
                    btn.classList.remove('counting');
                }}, 3000);
            }}
        }}, 1000);
    }}
    """

# ---------------- CSS ----------------
CSS = """
.records-cards { display: grid; gap: 10px; }
.rec-card { border-bottom: 4px solid rgba(255,255,255,0.35); padding: 8px 6px; }
.rec-header { display:flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px; }
.rec-header .left { font-weight: 600; }
.rec-header .right { opacity: .8; font-size: .95em; }
.nowrap { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.rec-sets { width: 100%; border-collapse: collapse; table-layout: fixed; }
.rec-sets td { border: 1px solid rgba(255,255,255,0.15); padding: 4px; vertical-align: top; }
.rec-sets td.sidx { width: 26px; text-align: center; opacity: .8; }
.rec-sets td.kg, .rec-sets td.r { width: 56px; }
.note-row td { background: rgba(255,255,255,0.04); }
.rec-sets td.note-cell { padding: 8px 6px; }
.rec-sets td.note-cell .time { margin-left: .5em; opacity:.65; font-size:.9em; }
@media (max-width: 480px) {
  .rec-sets td.kg, .rec-sets td.r { width: 48px; }
}
"""

# ---------------- 介面 ----------------
def _today_iso() -> str:
    # Update: 預設日期改為台北時間
    return get_now_tpe().date().isoformat()

with gr.Blocks(title=f"{APP_TITLE} {APP_VERSION}", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown(f"""# 🏋️‍♂️ Workout Logger {APP_VERSION} + 🤖 你的教練
    快速記錄重量訓練與查詢歷史。""")

    cloud_md = gr.Markdown(cloud_status_line())

    with gr.Tabs():
        with gr.TabItem("Log"):
            # Update: 這裡加上時區註記
            date_in = gr.Textbox(value="", label="Date (YYYY-MM-DD) [TPE UTC+8]")

            item_dd = gr.Dropdown(choices=get_all_item_choices(), allow_custom_value=True, value=None, label="Item 名稱")

            with gr.Row():
                set1kg = gr.Number(label="Set 1 — kg", precision=2, value=None, placeholder="kg")
                set1rp = gr.Number(label="Set 1 — r", precision=0, value=None, placeholder="r")
                btn_rest1 = gr.Button("Rest", size="sm", min_width=60, elem_id="rest_btn_1", scale=0)
                btn_rest1.click(None, None, None, js=get_rest_timer_js("rest_btn_1"))

            with gr.Row():
                set2kg = gr.Number(label="Set 2 — kg", precision=2, value=None, placeholder="kg")
                set2rp = gr.Number(label="Set 2 — r", precision=0, value=None, placeholder="r")
                btn_rest2 = gr.Button("Rest", size="sm", min_width=60, elem_id="rest_btn_2", scale=0)
                btn_rest2.click(None, None, None, js=get_rest_timer_js("rest_btn_2"))

            with gr.Row():
                set3kg = gr.Number(label="Set 3 — kg", precision=2, value=None, placeholder="kg")
                set3rp = gr.Number(label="Set 3 — r", precision=0, value=None, placeholder="r")
                btn_rest3 = gr.Button("Rest", size="sm", min_width=60, elem_id="rest_btn_3", scale=0)
                btn_rest3.click(None, None, None, js=get_rest_timer_js("rest_btn_3"))

            with gr.Row():
                set4kg = gr.Number(label="Set 4 — kg", precision=2, value=None, placeholder="kg")
                set4rp = gr.Number(label="Set 4 — r", precision=0, value=None, placeholder="r")
                btn_rest4 = gr.Button("Rest", size="sm", min_width=60, elem_id="rest_btn_4", scale=0)
                btn_rest4.click(None, None, None, js=get_rest_timer_js("rest_btn_4"))

            with gr.Row():
                set5kg = gr.Number(label="Set 5 — kg", precision=2, value=None, placeholder="kg")
                set5rp = gr.Number(label="Set 5 — r", precision=0, value=None, placeholder="r")
                btn_rest5 = gr.Button("Rest", size="sm", min_width=60, elem_id="rest_btn_5", scale=0)
                btn_rest5.click(None, None, None, js=get_rest_timer_js("rest_btn_5"))

            note_in = gr.Textbox(label="Note", placeholder="RPE、感覺、下次調整…")

            save_btn = gr.Button("💾 Save", variant="primary")
            status_md = gr.Markdown("")
            cur = load_records_df()
            latest_html = gr.HTML(value=(df_to_html_compact5(cur.tail(20)) if (cur is not None and not cur.empty) else ""), label="最近 20 筆紀錄")

            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, item_dd,
                        set1kg, set1rp, set2kg, set2rp, set3kg, set3rp, set4kg, set4rp, set5kg, set5rp,
                        note_in],
                outputs=[status_md, item_dd, latest_html, save_btn, cloud_md],
            )

            demo.load(fn=_today_iso, inputs=None, outputs=date_in)

        with gr.TabItem("Records"):
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                # 改為下拉選單（可輸入），選項取自歷史紀錄
                q_item = gr.Dropdown(
                    choices=get_all_item_choices(),
                    allow_custom_value=True,
                    value=None,
                    label="Item（下拉或輸入）"
                )
            refresh_btn = gr.Button("🔄 更新選單")
            query_btn = gr.Button("🔎 Search")
            out_html = gr.HTML(value=df_to_html_compact5(load_records_df()), label="搜尋結果")

            # 刷新下拉選單內容
            refresh_btn.click(lambda: gr.update(choices=get_all_item_choices()), None, q_item)

            query_btn.click(search_records_html, inputs=[q_from, q_to, q_item], outputs=out_html)

        with gr.TabItem("你的教練"):
            chatbot = gr.Chatbot(height=420, type='messages')
            user_in = gr.Textbox(placeholder="輸入你的問題，按 Enter 或點送出…", label="訊息")
            with gr.Row():
                use_ctx = gr.Checkbox(value=True, label="把最近紀錄提供給教練")
                ctx_days = gr.Slider(7, 180, value=60, step=1, label="最近（天）")
            with gr.Row():
                send_btn = gr.Button("送出", variant="primary")
                clear_btn = gr.Button("清空")
            send_btn.click(coach_chat_stream_ctx, inputs=[chatbot, user_in, use_ctx, ctx_days], outputs=[chatbot, user_in])
            user_in.submit(coach_chat_stream_ctx, inputs=[chatbot, user_in, use_ctx, ctx_days], outputs=[chatbot, user_in])
            clear_btn.click(lambda: ([], ""), None, [chatbot, user_in], queue=False)

    gr.Markdown("""---
**Tips**
- Item 名稱可直接輸入新文字，下次會出現在下拉選單。
- 空白的數值欄會保持空白（不顯示 0）。
- Total Volume = ∑(kg × r)。
""")

if __name__ == "__main__":
    if not RECORDS_CSV.exists():
        ensure_records_csv()
    demo.launch()

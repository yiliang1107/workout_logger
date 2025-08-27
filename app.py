"""
Gradio Workout Logger + 你的教練（Groq）— app.py（行動版 Note 顯示最佳化 + 雲端）
- 直接連 Google Sheet（SHEET_ID 固定，Worksheet 自動偵測 records/record/第一個分頁）。
- 10 分鐘內同日期+同 item 覆寫；內容相同不重存並暫時停用 Save。
- 所有列表（最近 20 筆、搜尋結果）改為 **兩列一筆** 的 HTML 表格：第二列專門放 Note，滿版顯示，行動裝置不會被吃掉。
- Google Sheet 的儲存格式維持原本欄位（note 為單一欄），只是在 UI 以兩列呈現。
"""
from __future__ import annotations
import os, json, hashlib, html, math
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, date, timedelta, timezone

# ---- Groq 安裝/匯入 ----
try:
    from groq import Groq
except ImportError:
    os.system('pip install groq')
    from groq import Groq

# ---- Google Sheets 相依 ----
try:
    import gspread
except ImportError:
    os.system('pip install gspread google-auth google-auth-oauthlib')
    import gspread

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

# 雲端狀態
CLOUD_LAST_ERROR = ""
CLOUD_WS_TITLE = None

# ------------ Google Sheets 工具 ------------

def _gs_client() -> Optional[gspread.Client]:
    try:
        sa_json = os.getenv("gspread_service_json") or os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
        if sa_json:
            creds_dict = json.loads(sa_json)
            return gspread.service_account_from_dict(creds_dict)
        return gspread.service_account()
    except Exception:
        return None


def _get_target_ws(sh: gspread.Spreadsheet) -> gspread.Worksheet:
    """優先 SHEET_TITLE_ENV → 'records' → 'record' → 第一個分頁；若沒有則建立 SHEET_TITLE_ENV。"""
    global CLOUD_WS_TITLE
    preferred = [SHEET_TITLE_ENV, "records", "record"]
    titles = [ws.title for ws in sh.worksheets()]
    for name in preferred:
        if name in titles:
            CLOUD_WS_TITLE = name
            return sh.worksheet(name)
    if titles:
        CLOUD_WS_TITLE = titles[0]
        return sh.worksheet(titles[0])
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
        ws.update(range_name="A1", values=[cols])


def read_cloud_df() -> Optional[pd.DataFrame]:
    """用 get_all_values 讀取；若只有表頭回傳空 DF 但保留欄位。"""
    global CLOUD_LAST_ERROR
    client = _gs_client()
    if not client:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return None
    try:
        ws = _open_or_create_ws(client)
        rows = ws.get_all_values()
        if not rows:
            return None
        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        if not header:
            return None
        if not data:
            df = pd.DataFrame(columns=header)
        else:
            df = pd.DataFrame(data, columns=header)
        # 嘗試轉數值欄型態
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


def write_cloud_df(df: pd.DataFrame) -> Tuple[bool, int]:
    """直接 ws.update(range_name='A1', values=...)；回傳 (成功與否, 寫入列數)。"""
    global CLOUD_LAST_ERROR
    client = _gs_client()
    if not client:
        CLOUD_LAST_ERROR = "無法建立 Google 憑證（未設定 service account 或檔案路徑）。"
        return False, 0
    try:
        ws = _open_or_create_ws(client)
        cols = ["date", "item"] + sum(([f"set{s}_kg", f"set{s}_reps"] for s in range(1, NUM_SETS+1)), []) + ["note", "total_volume_kg", "created_at"]
        out_df = df.copy()
        for c in cols:
            if c not in out_df.columns:
                out_df[c] = ""
        out_df = out_df[cols].fillna("")
        values = []
        for row in out_df.values.tolist():
            values.append([x if isinstance(x, (int, float, str)) else ("" if x is None else str(x)) for x in row])
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


# ------------ 優先雲端 & 狀態行 ------------

def load_records_df() -> pd.DataFrame:
    df = read_cloud_df()
    if df is not None:
        return df
    return load_local_df()


def save_records_df(df: pd.DataFrame) -> Tuple[bool, int]:
    ok_cloud, total_rows = write_cloud_df(df)
    write_local_df(df)
    return ok_cloud, total_rows


def cloud_status_line() -> str:
    df = read_cloud_df()
    target = CLOUD_WS_TITLE or SHEET_TITLE_ENV
    cloud_status = "已連線至雲端試算表 ✅" if df is not None else f"未連線至雲端（改用本機備援）❌  {CLOUD_LAST_ERROR}"
    try:
        count = 0 if df is None else len(df)
    except Exception:
        count = 0
    return f"**Cloud**：{cloud_status}，分頁：{target}，目前列數：{count}"

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

# ------------ HTML（兩列一筆，第二列放 Note） ------------

def _pretty_name(col: str) -> str:
    mapping = {"total_volume_kg": "total", "created_at": "created_at"}
    return mapping.get(col, col)


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

def to_tpe_time_str(created_at: str) -> str:
    if not created_at:
        return ""
    try:
        ts = pd.to_datetime(created_at, utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(created_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
        except Exception:
            return ""
    try:
        tpe = ts.tz_convert('Asia/Taipei')
        hour24 = int(tpe.strftime('%H'))
        minute = tpe.strftime('%M')
        if 18 <= hour24 <= 23:
            period = '晚上'
        elif 12 <= hour24 <= 17:
            period = '下午'
        else:
            period = '上午'
        # 全部統一 12 小時制，並移除小時的前導 0
        hour12 = ((hour24 - 1) % 12) + 1
        return f"{period} {hour12}:{minute}"
    except Exception:
        return ""
    try:
        ts = pd.to_datetime(created_at, utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(created_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
        except Exception:
            return ""
    try:
        tpe = ts.tz_convert('Asia/Taipei')
        hour24 = int(tpe.strftime('%H'))
        minute = tpe.strftime('%M')  # 保留兩位數
        if 18 <= hour24 <= 23:
            period = '晚上'
        elif 12 <= hour24 <= 17:
            period = '下午'
        else:
            period = '上午'
        # 12 小時制（1..12），移除小時前導零
        hour12 = ((hour24 - 1) % 12) + 1
        return f"{period} {hour12}:{minute}"
    except Exception:
        return ""
    try:
        ts = pd.to_datetime(created_at, utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(created_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
        except Exception:
            return ""
    try:
        tpe = ts.tz_convert('Asia/Taipei')
        hour24 = int(tpe.strftime('%H'))
        minute = tpe.strftime('%M')
        if 18 <= hour24 <= 23:
            period = '晚上'
        elif 12 <= hour24 <= 17:
            period = '下午'
        else:
            period = '上午'
        if period in ('下午', '晚上'):
            hour12 = ((hour24 - 1) % 12) + 1  # 12→12, 13→1, 18→6
            return f"{period} {hour12}:{minute}"
        else:
            # 上午維持 24 小時 HH:MM 顯示
            return f"{period} {tpe.strftime('%H:%M')}"
    except Exception:
        return ""
    try:
        ts = pd.to_datetime(created_at, utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(created_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
        except Exception:
            return ""
    try:
        tpe = ts.tz_convert('Asia/Taipei')
        hour = int(tpe.strftime('%H'))
        if 18 <= hour <= 23:
            period = '晚上'
        elif 12 <= hour <= 17:
            period = '下午'
        else:
            period = '上午'
        return f"{period} {tpe.strftime('%H:%M')}"
    except Exception:
        return ""
    try:
        ts = pd.to_datetime(created_at, utc=True)
    except Exception:
        try:
            ts = pd.to_datetime(created_at)
            if ts.tzinfo is None:
                ts = ts.tz_localize('UTC')
        except Exception:
            return ""
    try:
        return ts.tz_convert('Asia/Taipei').strftime('%H:%M')
    except Exception:
        return ""

def df_to_html_compact5(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "<div class='records-empty'>目前沒有紀錄</div>"
    # 確保 note 在最後
    if "note" in df.columns:
        cols = [c for c in df.columns if c != "note"] + ["note"]
        df = df[cols]
    cards = []
    for _, row in df.iterrows():
        date_s = row.get("date", "") or ""
        item_s = row.get("item", "") or ""
        note_s = row.get("note", "") or ""
        total_s = _fmt_num(row.get("total_volume_kg", ""))
        created_s = row.get("created_at", "") or ""
        time_tpe = to_tpe_time_str(created_s)
        # 五行：set1..set5，每行兩格（kg / r）
        lines = []
        for i in range(1, NUM_SETS+1):
            kg = _fmt_num(row.get(f"set{i}_kg", ""))
            rp = _fmt_num(row.get(f"set{i}_reps", ""))
            kg_txt = (kg + "kg") if kg else ""
            rp_txt = (rp + "r") if rp else ""
            lines.append(f"<tr><td class='sidx'>{i}</td><td class='kg nowrap'>{kg_txt}</td><td class='r nowrap'>{rp_txt}</td></tr>")
        lines_html = "".join(lines)
        note_row = f"<tr class='note-row'><td class='note-cell' colspan='3'><b>Note：</b>{html.escape(str(note_s))}<span class='time'>（{html.escape(time_tpe)}）</span></td></tr>"
        card = f"""
        <div class='rec-card'>
          <div class='rec-header'>
            <div class='left nowrap'>{html.escape(str(date_s))} · {html.escape(str(item_s))}</div>
            <div class='right nowrap'>{('Σ ' + html.escape(total_s) + ' kg') if total_s else ''}</div>
          </div>
          <table class='rec-sets'>
            <tbody>
              {lines_html}
              {note_row}
            </tbody>
          </table>
        </div>
        """
        cards.append(card)
    return "<div class='records-cards'>" + "".join(cards) + "</div>"

# ------------ 儲存（覆寫與重複判斷 + 回傳 HTML） ------------

def save_button_clicked(date_str: str, item_name: str,
                        set1kg, set1reps, set2kg, set2reps, set3kg, set3reps, set4kg, set4reps, set5kg, set5reps,
                        note: str):
    # 解析日期（若空白，預設今天）
    if not date_str or not str(date_str).strip():
        dt = date.today()
    else:
        try:
            dt = pd.to_datetime(date_str).date()
        except Exception:
            return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), "", gr.update(), cloud_status_line()

    item_name = (item_name or "").strip()
    if not item_name:
        return "沒有可存的資料：請至少填一個 Item 名稱", gr.update(), "", gr.update(), cloud_status_line()

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
    now_utc = datetime.utcnow().replace(tzinfo=timezone.utc)
    new_row = {
        "date": dt.isoformat(),
        "item": item_name,
        **sets_kv,
        "note": note or "",
        "total_volume_kg": total_volume,
        "created_at": now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
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
        return ("內容未變更：未儲存。", gr.update(choices=merged_choices), df_to_html_compact5(latest.tail(20)), gr.update(interactive=False), cloud_status_line())

    replaced = False
    if recent_row is not None:
        try:
            t_recent = pd.to_datetime(recent_row.get("created_at"), errors="coerce")
            if pd.notna(t_recent) and (datetime.utcnow() - t_recent.to_pydatetime()) <= timedelta(minutes=WINDOW_MINUTES):
                df = df.drop(index=idx_recent)
                replaced = True
        except Exception:
            pass

    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # note 放最後一欄
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

    merged_choices = get_all_item_choices()
    latest = load_records_df()
    return (msg, gr.update(choices=merged_choices), df_to_html_compact5(latest.tail(20)), gr.update(interactive=True), cloud_status_line())

# ------------ 搜尋 ------------

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


def search_records_html(date_from: str, date_to: str, item_filter: str):
    return df_to_html_compact5(search_records(date_from, date_to, item_filter))

# ------------ 教練資料摘要（提供給 Groq） ------------

def _truncate(s: str, n: int) -> str:
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def make_coach_context(days: int = 60, max_items: int = 8, max_recent: int = 10) -> str:
    df = load_records_df()
    if df is None or df.empty:
        return "（目前沒有雲端紀錄）"
    f = df.copy()
    # 日期過濾
    try:
        f["date_dt"] = pd.to_datetime(f["date"], errors="coerce")
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=days)
        f = f[f["date_dt"] >= cutoff]
    except Exception:
        pass
    if f.empty:
        return f"（最近 {days} 天沒有紀錄）"
    # 每項目統計
    lines = [f"期間：最近 {days} 天"]
    try:
        vol = f.groupby("item", dropna=False)["total_volume_kg"].sum(min_count=1).sort_values(ascending=False)
    except Exception:
        vol = pd.Series(dtype=float)
    try:
        cnt = f["item"].value_counts()
    except Exception:
        cnt = pd.Series(dtype=int)
    # 最近日期
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
    # 最近幾筆（精簡）
    try:
        f["created_at_dt"] = pd.to_datetime(f["created_at"], errors="coerce")
        recent = f.sort_values("created_at_dt", ascending=False).head(max_recent)
    except Exception:
        recent = f.tail(max_recent)
    lines.append("最近幾筆：")
    for _, r in recent.iterrows():
        parts = []
        for i in range(1, NUM_SETS + 1):
            kg = _fmt_num(r.get(f"set{i}_kg"))
            rp = _fmt_num(r.get(f"set{i}_reps"))
            if kg and rp:
                parts.append(f"{kg}x{rp}")
        sets_txt = "/".join(parts)
        note_txt = _truncate(r.get("note", ""), 40)
        total_txt = _fmt_num(r.get("total_volume_kg"))
        lines.append(f"- {r.get('date','')} {r.get('item','')}: {sets_txt}；備註：{note_txt}；total={total_txt}kg")
    return "
".join(lines):
    return df_to_html_compact5(search_records(date_from, date_to, item_filter))

# ------------ 教練機器人（串流） ------------

def coach_chat_stream_ctx(history, user_msg: str, use_ctx: bool, ctx_days: int):
    msg = (user_msg or "").strip()
    if not msg:
        yield history, ""
        return
    if groq_client is None:
        bot_text = "（尚未設定環境變數 groq_key，請設定後重試。）"
        # messages 型式：list[dict]
        if isinstance(history, list) and (not history or isinstance(history[0], dict)):
            ui = history + [{"role": "user", "content": msg}, {"role": "assistant", "content": bot_text}]
        else:
            ui = (history or []) + [[msg, bot_text]]
        yield ui, ""
        return

    # 構建發給 Groq 的訊息
    sys_content = SYSTEM_PROMPT
    if use_ctx:
        try:
            ctx = make_coach_context(int(ctx_days))
        except Exception:
            ctx = make_coach_context()
        sys_content += f"

【學員近期紀錄摘要】
{ctx}"

    api_messages = [{"role": "system", "content": sys_content}]
    # 將歷史對話轉為 user/assistant 交替
    if isinstance(history, list) and history and isinstance(history[0], dict):
        for m in history:
            if m.get("role") in ("user", "assistant"):
                api_messages.append({"role": m.get("role"), "content": m.get("content", "")})
        ui_history = history.copy()
    else:
        # 舊的 (user, bot) tuples
        for u, b in (history or []):
            if u:
                api_messages.append({"role": "user", "content": u})
            if b:
                api_messages.append({"role": "assistant", "content": b})
        # 轉為 messages 風格供 UI 使用
        ui_history = []
        for u, b in (history or []):
            if u:
                ui_history.append({"role": "user", "content": u})
            if b:
                ui_history.append({"role": "assistant", "content": b})

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
        # UI 歷史：加上使用者訊息與空白助理訊息
        ui_history = ui_history + [{"role": "user", "content": msg}, {"role": "assistant", "content": ""}]
        acc = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            if delta:
                acc += delta
                ui_history[-1]["content"] = acc
                yield ui_history, ""
        return
    except Exception as e:
        ui_history = ui_history + [{"role": "user", "content": msg}, {"role": "assistant", "content": f"抱歉，Groq 呼叫失敗：{e}"}]
        yield ui_history, ""

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
lambda: ([], ""), None, [chatbot, user_in], queue=False)

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

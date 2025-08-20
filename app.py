"""
Gradio Workout Logger — 單檔可執行
需求：
1) Date 預設今天、可修改
2) item：6 個可填，輸入過的動作會記憶成下拉選項（可自訂新值）
3) 每個 item 有 5 組 set（每組 kg + reps）
4) 每個 item 有 Note 欄
5) Save 會把資料持續追加到 CSV 紀錄檔，可在 Records 分頁查找歷史

執行方式：
    pip install gradio pandas python-dateutil
    python app.py
"""
from __future__ import annotations
import gradio as gr
import pandas as pd
from datetime import datetime, date
import json
from pathlib import Path
from typing import List, Tuple

# ------------ 常數與檔案路徑 ------------
APP_TITLE = "Workout Logger"
RECORDS_CSV = Path("workout_records.csv")
ITEMS_JSON = Path("known_items.json")
NUM_ITEMS = 6
NUM_SETS = 5

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

# ------------ 商業邏輯 ------------

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


def save_button_clicked(date_str: str,
                        *flat_inputs):
    """
    flat_inputs 包含 6 個 item 區塊，展平為：
    [item_name, set1_kg, set1_reps, set2_kg, set2_reps, ..., set5_kg, set5_reps, note] * 6
    """
    # 解析日期
    try:
        # 支援 "YYYY-MM-DD" 或 "YYYY/MM/DD" 等
        dt = pd.to_datetime(date_str).date()
    except Exception:
        return "日期格式錯誤，請用 YYYY-MM-DD", gr.update(), pd.DataFrame()

    # 將展平的輸入回填為每個 item 的結構
    block_size = 1 + (NUM_SETS * 2) + 1  # item 名稱 + 10 個 set 欄 + note
    rows = []
    all_new_item_names = []

    for i in range(NUM_ITEMS):
        start = i * block_size
        end = start + block_size
        chunk = list(flat_inputs[start:end])
        item_name = (chunk[0] or "").strip()
        if not item_name:
            # 空白 item 直接跳過
            continue
        all_new_item_names.append(item_name)

        # 解析 5 組 sets
        kg_vals, reps_vals = [], []
        sets_kv = {}
        pos = 1
        for s in range(1, NUM_SETS+1):
            kg = chunk[pos]; reps = chunk[pos+1]
            pos += 2
            # 轉為數字/或 None
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

    # 追加寫入 CSV
    append_records(rows)

    # 更新已知 item 清單
    known = load_known_items()
    merged = list(dict.fromkeys([*known, *all_new_item_names]))
    save_known_items(merged)

    # 回傳訊息與最新的記錄總覽
    df = pd.read_csv(RECORDS_CSV)
    return (f"已儲存 {len(rows)} 筆（日期：{dt.isoformat()}）。",
            gr.update(choices=merged),
            df.tail(20))


# ---- Records 搜尋 ----

def search_records(date_from: str, date_to: str, item_filter: str):
    ensure_records_csv()
    if not RECORDS_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(RECORDS_CSV)

    # 日期篩選
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

    # 項目關鍵字（包含）
    if item_filter:
        df = df[df["item"].astype(str).str.contains(item_filter, case=False, na=False)]

    # 依日期與建立時間排序
    if not df.empty:
        df = df.sort_values(["date", "created_at"], ascending=[False, False])
    return df


# ------------ 建立介面 ------------
with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏋️‍♂️ Workout Logger\n用來快速記錄重量訓練每個動作 5 組的重量與次數，並可查詢歷史紀錄。")

    with gr.Tabs():
        with gr.TabItem("Log"):
            today_str = date.today().isoformat()
            date_in = gr.Textbox(value=today_str, label="Date (YYYY-MM-DD)")

            # 讀取已知 item 選項
            known_items = load_known_items()

            item_dropdowns = []  # 6 個 item 名稱元件（Dropdown）
            set_inputs = []      # 對應每個 item 的 10 個數值欄
            note_inputs = []     # 每個 item 的 Note

            for i in range(NUM_ITEMS):
                with gr.Group():
                    gr.Markdown(f"### Item {i+1}")
                    # 允許自訂輸入，會記住
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

            # 彙整所有輸入順序：
            flat_all_inputs = []
            for i in range(NUM_ITEMS):
                flat_all_inputs.append(item_dropdowns[i])
                flat_all_inputs += set_inputs[i]
                flat_all_inputs.append(note_inputs[i])

            # Save 動作
            save_btn.click(
                fn=save_button_clicked,
                inputs=[date_in, *flat_all_inputs],
                outputs=[status_md, item_dropdowns[0], latest_df],
            )

        with gr.TabItem("Records"):
            gr.Markdown("### 搜尋歷史紀錄")
            with gr.Row():
                q_from = gr.Textbox(label="From (YYYY-MM-DD)")
                q_to = gr.Textbox(label="To (YYYY-MM-DD)")
                q_item = gr.Textbox(label="Item 包含（關鍵字）")
            query_btn = gr.Button("🔎 Search")
            out_df = gr.Dataframe(headers=None, value=pd.DataFrame(), wrap=True, interactive=False, label="搜尋結果")

            query_btn.click(search_records, inputs=[q_from, q_to, q_item], outputs=out_df)

    gr.Markdown("---\n**Tips**\n- Item 名稱可直接輸入新文字，下次會出現在下拉選單。\n- 空白的 Item 不會儲存。\n- Total Volume = ∑(kg × reps)。")

if __name__ == "__main__":
    ensure_records_csv()
    demo.launch()

# ---------------- Part 1/5 ----------------
# Imports, .env, DB init, email, auth, safe helpers

import os
import io
import json
import re
import traceback
import base64
import threading
import atexit
from datetime import datetime, date, timedelta, timezone

import sqlite3
import requests
import streamlit as st
import pandas as pd
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from dotenv import load_dotenv
from email.message import EmailMessage
import smtplib
from apscheduler.schedulers.background import BackgroundScheduler

# optional fuzzy lib
try:
    import Levenshtein
    _lev_distance = lambda a,b: Levenshtein.distance(a,b)
except Exception:
    try:
        from rapidfuzz.distance import Levenshtein as RLev
        _lev_distance = lambda a,b: RLev.distance(a,b)
    except Exception:
        _lev_distance = None

# Load environment
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip() or None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.0-flash").strip()
SMTP_HOST = os.getenv("SMTP_HOST") or "smtp.gmail.com"
SMTP_PORT = int(os.getenv("SMTP_PORT") or 587)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
FROM_EMAIL = os.getenv("FROM_EMAIL") or SMTP_USER
TESSERACT_CMD = os.getenv("TESSERACT_CMD")  # optional; required if using pytesseract on Windows

# DB setup
DB = "caresync_gemini_full.db"
conn = sqlite3.connect(DB, check_same_thread=False)
c = conn.cursor()
db_lock = threading.Lock()

def init_db():
    with db_lock:
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE,
                phone TEXT,
                name TEXT,
                role TEXT CHECK(role IN ('patient','caregiver')),
                password TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS caregivers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                caregiver_id INTEGER,
                caregiver_phone TEXT,
                access_level TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                name TEXT,
                dosage TEXT,
                frequency TEXT,
                start_date TEXT,
                end_date TEXT,
                reminder_time TEXT,
                notes TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS med_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                med_id INTEGER,
                timestamp TEXT,
                status TEXT,
                actor_id INTEGER,
                note TEXT
            )
        """)
        conn.commit()

init_db()

# Email sender (HTML)
def send_email(to_email: str, subject: str, body_html: str) -> bool:
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS and FROM_EMAIL):
        print("Email not configured.")
        return False
    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = FROM_EMAIL
        msg["To"] = to_email
        msg.set_content("Your email client does not support HTML.")
        msg.add_alternative(body_html or "", subtype="html")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
            s.starttls()
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
        print("Email sent to:", to_email)
        return True
    except Exception as e:
        print("Email error:", e)
        traceback.print_exc()
        return False

# Auth helpers
def signup(name: str, email: str, phone: str, password: str, role: str) -> bool:
    try:
        with db_lock:
            c.execute("INSERT INTO users (email, phone, name, password, role) VALUES (?, ?, ?, ?, ?)",
                      (email, phone, name, password, role))
            conn.commit()
        return True
    except Exception as e:
        print("Signup error:", e)
        return False

def login(email: str, password: str):
    try:
        return c.execute("SELECT id, name, role FROM users WHERE email=? AND password=?", (email, password)).fetchone()
    except Exception as e:
        print("Login error:", e)
        return None

# stable rerun helper (works across Streamlit versions)
def safe_rerun():
    try:
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass
# ---------------- Part 2/5 ----------------
# Tesseract OCR pipeline + cleaning + Gemini text parsing + aggressive guesser

# Import pytesseract if available
try:
    import pytesseract
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception as e:
    pytesseract = None
    print("pytesseract import warning:", e)

def clean_image_for_ocr(img: Image.Image) -> Image.Image:
    try:
        img = img.convert("L")
        img = ImageOps.autocontrast(img)
        img = ImageEnhance.Contrast(img).enhance(1.6)
        img = img.filter(ImageFilter.SHARPEN)
        max_dim = 2500
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim))
        return img
    except Exception as e:
        print("clean_image_for_ocr error:", e)
        return img

def tesseract_extract_text(image_bytes: bytes):
    if pytesseract is None:
        return None, "pytesseract not installed"
    try:
        img = Image.open(io.BytesIO(image_bytes))
        cleaned = clean_image_for_ocr(img)
        text = pytesseract.image_to_string(cleaned, lang=None)
        return text, None
    except Exception as e:
        traceback.print_exc()
        return None, f"Tesseract error: {e}"

def sanitize_ocr_text(txt: str) -> str:
    if not txt:
        return ""
    txt = txt.replace("\r", "\n")
    # keep punctuation relevant for numbers, percent, hyphen and slash
    txt = re.sub(r"[^\x00-\x7F]+", " ", txt)
    # collapse suspicious long spaces
    txt = re.sub(r"\s{2,}", " ", txt)
    # keep lines
    lines = []
    for ln in txt.splitlines():
        ln2 = ln.strip()
        if not ln2:
            continue
        # preserve numbers and letters and common symbols
        ln2 = re.sub(r"[^A-Za-z0-9\s\-\.,:/()%]", " ", ln2)
        ln2 = re.sub(r"\s{2,}", " ", ln2).strip()
        if ln2:
            lines.append(ln2)
    return "\n".join(lines)

# Gemini parser (text model) with strong guessing instructions
def gemini_parse_ocr_text_to_json(ocr_text: str, model: str = None):
    if model is None:
        model = GEMINI_MODEL or "models/gemini-2.0-flash"
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not configured"
    try:
        url = f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent?key={GEMINI_API_KEY}"
        prompt = (
            "You are a robust parser for noisy OCR text from medical prescriptions and pharmacy invoices. "
            "The OCR is often corrupted; your job is to produce a clean JSON array of medication objects. "
            "You MAY GUESS missing values and fix obvious spelling errors. When you GUESS a value, add '(AI guessed)' "
            "to the notes field of that item. DO NOT output any text except valid JSON array.\n\n"
            "Each item MUST have keys: name (string), dosage (string), frequency (string), start_date (YYYY-MM-DD or empty), "
            "end_date (YYYY-MM-DD or empty), reminder_time (string), notes (string).\n\n"
            "If you are not sure about a field, leave it empty but add '(AI guessed)' in notes if you made any guess for that item.\n\n"
            "OCR TEXT:\n"
            + ocr_text
            + "\n\nReturn only a JSON array, e.g.:\n"
            '[{"name":"Paracetamol","dosage":"500 mg","frequency":"Twice a day","start_date":"","end_date":"","reminder_time":"08:00 AM","notes":"AI guessed dosage"}]\n'
        )
        body = {"contents":[{"parts":[{"text":prompt}]}], "generationConfig":{"temperature":0.0,"maxOutputTokens":1200}}
        headers = {"Content-Type":"application/json"}
        r = requests.post(url, json=body, headers=headers, timeout=90)
        r.raise_for_status()
        data = r.json()
        candidates = data.get("candidates") or []
        if not candidates:
            # Sometimes the model returns 'output' or 'output_text'
            text_blob = json.dumps(data)
        else:
            cand = candidates[0]
            # attempt to extract text from known fields
            raw_text = ""
            content = cand.get("content")
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict):
                        raw_text += p.get("text","")
            elif isinstance(content, dict):
                for p in content.get("parts", []):
                    raw_text += p.get("text","")
            if not raw_text:
                raw_text = cand.get("text") or json.dumps(cand)
            text_blob = raw_text
        # find JSON array in returned text
        jm = re.search(r"(\[[\s\S]*\])", text_blob)
        if jm:
            js = jm.group(1)
            try:
                parsed = json.loads(js)
                if isinstance(parsed, list):
                    return parsed, None
            except Exception as e:
                # return raw text for debugging
                return None, f"JSON parse error: {e}\nRaw: {js}"
        return None, f"No JSON found in Gemini output. Raw:\n{text_blob}"
    except requests.exceptions.RequestException as e:
        traceback.print_exc()
        return None, f"Gemini text error: {e}"
    except Exception as e:
        traceback.print_exc()
        return None, f"Gemini parsing error: {e}"

# Aggressive heuristic parser (fallback) - maximum guessing
def aggressive_guess_from_ocr(ocr_text: str):
    meds = []
    lines = [ln.strip() for ln in ocr_text.splitlines() if ln.strip()]
    # Candidate name tokens: words with letters and possibly hyphen
    for ln in lines:
        # skip header-like lines
        if re.search(r"(invoice|gst|phone|address|patient|dr\.|date|amount|net|signature)", ln, re.I):
            continue
        # look for patterns of medicine-like lines
        if re.search(r"(mg|ml|tablet|tab|capsule|cap|tds|bd|once|twice|daily|\d+\s*mg|\d+\s*ml)", ln, re.I) or re.search(r"[A-Z]{2,}", ln):
            # try to extract dosage
            dosage = ""
            dmatch = re.search(r"(\d+\s*(mg|ml|mcg|g))", ln, re.I)
            if dmatch:
                dosage = dmatch.group(0)
            # frequency guess
            freq = ""
            fmatch = re.search(r"(once daily|twice daily|once a day|twice a day|once|twice|daily|bd|tds|at night|morning|night|\d+\s*times\s*a\s*day)", ln, re.I)
            if fmatch:
                freq = fmatch.group(0)
            # name guess: text before dosage or first 4-6 words
            name = ln
            if dosage:
                name = ln.split(dosage)[0].strip(" -,:;")
            # clean name
            name = re.sub(r"[^A-Za-z0-9\s\-()\/]", " ", name).strip()
            if len(name) < 2:
                continue
            notes = "AI guessed"
            meds.append({
                "name": name,
                "dosage": dosage or "unknown (AI guessed)",
                "frequency": freq or "unknown (AI guessed)",
                "start_date": "",
                "end_date": "",
                "reminder_time": "",
                "notes": notes
            })
    # if still empty, attempt to greedy chunking of capitalized tokens
    if not meds:
        tokens = re.findall(r"[A-Za-z]{3,}[A-Za-z0-9\-]*", ocr_text)
        for i in range(0, len(tokens), 2):
            name = " ".join(tokens[i:i+2])
            if name:
                meds.append({
                    "name": name,
                    "dosage": "unknown (AI guessed)",
                    "frequency": "unknown (AI guessed)",
                    "start_date": "",
                    "end_date": "",
                    "reminder_time": "",
                    "notes": "AI guessed"
                })
    return meds
# ---------------- Part 3/5 ----------------
# Extraction wrapper, DB insert, reminders, weekly summary

def extract_meds_from_image_bytes(image_bytes: bytes):
    # OCR
    ocr_raw, o_err = tesseract_extract_text(image_bytes) if pytesseract else (None, "pytesseract not installed")
    ocr_clean = sanitize_ocr_text(ocr_raw or "")
    # Try Gemini parse if available
    if GEMINI_API_KEY and ocr_clean.strip():
        parsed, gem_err = gemini_parse_ocr_text_to_json(ocr_clean, model=GEMINI_MODEL)
        if parsed and isinstance(parsed, list) and parsed:
            # ensure fields exist and mark guesses if necessary
            out = []
            for item in parsed:
                it = {k: (item.get(k,"") if item.get(k) is not None else "") for k in ["name","dosage","frequency","start_date","end_date","reminder_time","notes"]}
                # if blank fields, and no notes, mark AI guessed when necessary
                guessed = False
                if not it["dosage"] or not it["frequency"] or "AI guessed" in it.get("notes",""):
                    guessed = True
                if guessed and it.get("notes","") == "":
                    it["notes"] = "AI guessed"
                out.append(it)
            return out, ocr_raw or "", None
        else:
            print("Gemini parse empty or error:", gem_err)
    # Gemini didn't return usable JSON -> fallback to aggressive heuristics
    fallback = aggressive_guess_from_ocr(ocr_clean)
    if fallback:
        return fallback, ocr_raw or "", None
    # nothing
    return [], ocr_raw or "", (o_err or "No medications detected")

def add_extracted_medications(patient_id: int, meds_list: list):
    added = 0
    with db_lock:
        for m in meds_list:
            try:
                c.execute("""
                    INSERT INTO medications (patient_id, name, dosage, frequency, start_date, end_date, reminder_time, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    patient_id,
                    m.get("name","") or "",
                    m.get("dosage","") or "",
                    m.get("frequency","") or "",
                    m.get("start_date","") or "",
                    m.get("end_date","") or "",
                    m.get("reminder_time","") or "",
                    m.get("notes","") or ""
                ))
                added += 1
            except Exception as e:
                print("DB insert error:", e)
        conn.commit()
    return added

# Reminder helpers (same dedupe as earlier)
def _normalize_time(t: str):
    if not t: return None
    s = str(t).strip()
    up = s.upper().replace(".", "")
    parts = up.split()
    if len(parts) == 2 and parts[1] in ("AM","PM"):
        time_part = parts[0]
        if ":" not in time_part: return None
        try:
            hh, mm = map(int, time_part.split(":"))
        except:
            return None
        if parts[1] == "PM" and hh != 12: hh += 12
        if parts[1] == "AM" and hh == 12: hh = 0
        return f"{hh:02d}:{mm:02d}"
    if ":" in s:
        try:
            hh, mm = map(int, s.split(":"))
            if 0 <= hh < 24 and 0 <= mm < 60:
                return f"{hh:02d}:{mm:02d}"
        except:
            return None
    return None

def parse_reminder_times(text: str):
    if not text: return [], {}
    parts = [p.strip() for p in text.split(",") if p.strip()]
    daily, weekly = [], {}
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    for p in parts:
        found = None
        for d in days:
            if p.lower().startswith(d.lower()+":"):
                found = d; break
        if found:
            rest = p.split(":",1)[1].strip()
            nt = _normalize_time(rest)
            if nt: weekly.setdefault(found, []).append(nt)
            continue
        nt = _normalize_time(p)
        if nt: daily.append(nt)
    return sorted(set(daily)), {k: sorted(set(v)) for k,v in weekly.items()}

def _parse_date_flexible(s: str):
    if not s:
        raise ValueError("Empty date")
    s2 = str(s).strip().replace("/", "-").replace(".", "-")
    return datetime.fromisoformat(s2).date()

def already_sent_today_for_schedule(med_id: int, scheduled_time: str, target_date: date) -> bool:
    pattern = f"%|{scheduled_time}|{target_date.isoformat()}"
    row = c.execute("SELECT id FROM med_logs WHERE med_id=? AND status='reminder_sent' AND note LIKE ? LIMIT 1", (med_id, pattern)).fetchone()
    return bool(row)

def notify_all(patient_id: int, title: str, message_html: str):
    try:
        row = c.execute("SELECT email FROM users WHERE id=?", (patient_id,)).fetchone()
        if row and row[0]:
            send_email(row[0], title, message_html)
        caregivers = c.execute("SELECT u.email FROM caregivers c JOIN users u ON u.id=c.caregiver_id WHERE c.patient_id=?", (patient_id,)).fetchall()
        for (cem,) in caregivers:
            if cem:
                send_email(cem, title + " (Caregiver copy)", message_html)
    except Exception as e:
        print("notify_all error:", e)

def mark_dose(med_id: int, uid: int, status: str, note: str = ""):
    try:
        ts = datetime.now(timezone.utc).isoformat()
        with db_lock:
            c.execute("INSERT INTO med_logs (med_id, timestamp, status, actor_id, note) VALUES (?, ?, ?, ?, ?)",
                      (med_id, ts, status, uid if status in ("taken","missed") else None, note))
            conn.commit()
        return True
    except Exception as e:
        print("mark_dose error:", e)
        return False

def check_and_send_reminders():
    try:
        now = datetime.now()
        today = now.date()
        now_hm = now.strftime("%H:%M")
        weekday = now.strftime("%a")
        meds = c.execute("SELECT id, patient_id, name, dosage, start_date, end_date, reminder_time FROM medications").fetchall()
        for mid, pid, name, dosage, sd_raw, ed_raw, rtext in meds:
            try:
                sd = _parse_date_flexible(sd_raw)
                ed = _parse_date_flexible(ed_raw)
            except Exception:
                continue
            if not (sd <= today <= ed):
                continue
            daily, weekly = parse_reminder_times(rtext)
            should_fire = False; scheduled_time_hit = None
            if now_hm in daily:
                should_fire = True; scheduled_time_hit = now_hm
            if weekday in weekly and now_hm in weekly.get(weekday, []):
                should_fire = True; scheduled_time_hit = now_hm
            if not should_fire:
                continue
            if scheduled_time_hit is None:
                scheduled_time_hit = now_hm
            if already_sent_today_for_schedule(mid, scheduled_time_hit, today):
                print(f"[Dedup] skipping med {mid} at {scheduled_time_hit} for {today}")
                continue
            ts = datetime.now(timezone.utc).isoformat()
            note = f"reminder_sent|{scheduled_time_hit}|{today.isoformat()}"
            with db_lock:
                c.execute("INSERT INTO med_logs (med_id, timestamp, status, actor_id, note) VALUES (?, ?, 'reminder_sent', NULL, ?)", (mid, ts, note))
                conn.commit()
            notify_all(pid, f"Medication Reminder — {name}", f"<p>Please take <b>{name}</b> ({dosage}).</p><p>— CareSync</p>")
            print(f"Reminder sent for {name} at {scheduled_time_hit}")
    except Exception as e:
        print("Reminder error:", e); traceback.print_exc()

def compose_weekly_summary(pid: int):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=7)
    u = c.execute("SELECT name,email FROM users WHERE id=?", (pid,)).fetchone()
    if not u:
        return None, None
    name, email = u
    meds = c.execute("SELECT id,name,dosage,frequency,reminder_time FROM medications WHERE patient_id=?", (pid,)).fetchall()
    rows_html = ""
    total_taken = total_missed = 0
    for mid, mname, dosage, freq, rtime in meds:
        logs = c.execute("SELECT status, COUNT(*) FROM med_logs WHERE med_id=? AND timestamp BETWEEN ? AND ? GROUP BY status", (mid, start.isoformat(), now.isoformat())).fetchall()
        taken = sum(cnt for st,cnt in logs if st=="taken")
        missed = sum(cnt for st,cnt in logs if st=="missed")
        total_taken += taken; total_missed += missed
        rows_html += f"<tr><td>{mname}</td><td>{dosage}</td><td>{freq}</td><td>{taken}</td><td>{missed}</td><td>{rtime}</td></tr>"
    total = total_taken + total_missed
    adherence = int((total_taken/total)*100) if total>0 else 0
    subject = f"Weekly Summary — {name}"
    body = f"""
<html>
<body>
  <h2>Weekly Summary for {name}</h2>
  <p>Your adherence for the last 7 days: <b>{adherence}%</b></p>
  <table border='1' cellpadding='6' style='border-collapse:collapse;'>
    <tr style='background:#eef;'>
      <th>Medication</th><th>Dosage</th><th>Frequency</th><th>Taken</th><th>Missed</th><th>Reminder</th>
    </tr>
    {rows_html}
  </table>
  <p>— CareSync</p>
</body>
</html>
"""
    return subject, body

def send_weekly_summary_for(pid:int):
    subject, html = compose_weekly_summary(pid)
    if subject:
        em = c.execute("SELECT email FROM users WHERE id=?", (pid,)).fetchone()
        if em and em[0]:
            send_email(em[0], subject, html)

def send_weekly_summary_all():
    pts = c.execute("SELECT id FROM users WHERE role='patient'").fetchall()
    for (pid,) in pts:
        send_weekly_summary_for(pid)

# Scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(check_and_send_reminders, "interval", minutes=1, id="reminder_job")
scheduler.add_job(send_weekly_summary_all, "cron", day_of_week="sun", hour=0, minute=30, id="weekly_summary_job")
scheduler.start()
print("Scheduler started.")
atexit.register(lambda: scheduler.shutdown(wait=False))
# ---------------- Part 4/5 ----------------
# UI: prescription upload, review, add, edit, delete

def extract_and_preview_ui(uid: int):
    st.subheader("Upload prescription (image)")
    uploaded = st.file_uploader("Upload image (jpg/png). For PDFs convert first page to image.", type=["jpg","jpeg","png","pdf"], key=f"upload_rx_{uid}")
    if uploaded:
        raw = uploaded.read()
        st.info("Processing — OCR + AI parsing (may take a few seconds)...")
        meds_list, raw_ocr, err = extract_meds_from_image_bytes(raw)
        st.session_state["last_rx_items"] = meds_list
        st.session_state["last_rx_raw"] = raw_ocr
        st.session_state["last_rx_err"] = err
        if err:
            st.warning("Processing note: " + str(err))
        if meds_list and len(meds_list) > 0:
            st.success(f"Extracted {len(meds_list)} medication(s). Review below.")
        else:
            st.info("No medications extracted. Check raw OCR or add manually.")

def medications_ui_for_user(uid: int, uname: str):
    st.header("Manage Medications")
    extract_and_preview_ui(uid)

    if st.session_state.get("last_rx_raw"):
        with st.expander("View raw OCR text"):
            st.code(st.session_state.get("last_rx_raw") or "")

    if st.session_state.get("last_rx_items"):
        items = st.session_state["last_rx_items"]
        st.markdown("### Review extracted medications (AI may have guessed values — check notes)")
        for i, item in enumerate(items):
            st.markdown(f"**Item {i+1}**")
            c1, c2, c3 = st.columns([3,2,2])
            name = c1.text_input(f"Name {i+1}", value=item.get("name",""), key=f"rx_name_{i}")
            dosage = c2.text_input(f"Dosage {i+1}", value=item.get("dosage",""), key=f"rx_dosage_{i}")
            freq = c3.text_input(f"Frequency {i+1}", value=item.get("frequency",""), key=f"rx_freq_{i}")
            c4, c5 = st.columns([2,3])
            try:
                sd_val = date.today() if not item.get("start_date") else datetime.fromisoformat(item.get("start_date")).date()
            except:
                sd_val = date.today()
            try:
                ed_val = date.today()+timedelta(days=30) if not item.get("end_date") else datetime.fromisoformat(item.get("end_date")).date()
            except:
                ed_val = date.today()+timedelta(days=30)
            sd = c4.date_input(f"Start date {i+1}", value=sd_val, key=f"rx_sd_{i}")
            ed = c5.date_input(f"End date {i+1}", value=ed_val, key=f"rx_ed_{i}")
            rt = st.text_input(f"Reminder times {i+1}", value=item.get("reminder_time",""), key=f"rx_rt_{i}")
            notes = st.text_area(f"Notes {i+1}", value=item.get("notes",""), key=f"rx_notes_{i}")
            items[i] = {"name": name.strip(), "dosage": dosage.strip(), "frequency": freq.strip(),
                        "start_date": str(sd), "end_date": str(ed), "reminder_time": rt.strip(), "notes": notes.strip()}
            st.markdown("---")
        if st.button("Save extracted medications to My List"):
            added = add_extracted_medications(uid, items)
            st.success(f"Added {added} medication(s). You can edit them individually on this page.")
            # clear but keep last rx raw for debugging
            st.session_state.pop("last_rx_items", None)
            safe_rerun()
        if st.button("Clear extracted items"):
            st.session_state.pop("last_rx_items", None)
            st.session_state.pop("last_rx_raw", None)
            st.session_state.pop("last_rx_err", None)
            st.success("Cleared extracted items.")

    # Manual add form
    st.subheader("Add Medication Manually")
    with st.form(f"add_med_form_{uid}"):
        name_med = st.text_input("Medicine name", key=f"m_name_{uid}")
        dosage = st.text_input("Dosage (e.g., 500 mg)", key=f"m_dosage_{uid}")
        frequency = st.text_input("Frequency (e.g., Twice a day)", key=f"m_freq_{uid}")
        start_date = st.date_input("Start date", value=date.today(), key=f"m_start_{uid}")
        end_date = st.date_input("End date", value=date.today()+timedelta(days=30), key=f"m_end_{uid}")
        reminder_time = st.text_input("Reminder Times (e.g., 08:00 AM, Mon:08:00)", key=f"m_rem_{uid}")
        notes = st.text_area("Notes (optional)", key=f"m_notes_{uid}")
        add_submitted = st.form_submit_button("Add Medication")
        if add_submitted:
            try:
                with db_lock:
                    c.execute("""
                        INSERT INTO medications (patient_id, name, dosage, frequency, start_date, end_date, reminder_time, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (uid, name_med.strip(), dosage.strip(), frequency.strip(), str(start_date), str(end_date), reminder_time.strip(), notes.strip()))
                    conn.commit()
                st.success("Medication added.")
                safe_rerun()
            except Exception as e:
                st.error("Error adding medication.")
                print("Add med error:", e)

    # Display meds with Edit/Delete/Taken/Missed
    st.subheader("Your Medications")
    meds_df = pd.read_sql("SELECT id,name,dosage,frequency,reminder_time,start_date,end_date,notes FROM medications WHERE patient_id=? ORDER BY id DESC", conn, params=(uid,))
    if meds_df.empty:
        st.info("No medications yet.")
    else:
        for _, row in meds_df.iterrows():
            mid = int(row["id"])
            st.markdown(f"### {row['name']}  —  {row['dosage']}")
            st.write(f"**Frequency:** {row['frequency']}")
            st.write(f"**Reminder:** {row['reminder_time']}")
            st.write(f"**Start:** {row['start_date']}  **End:** {row['end_date']}")
            st.write(f"**Notes:** {row.get('notes','')}")
            col1, col2, col3, col4 = st.columns([1,1,1,1])
            if col1.button("Edit", key=f"edit_{mid}"):
                st.session_state[f"editing_{mid}"] = True
                safe_rerun()
            if col2.button("Delete", key=f"del_{mid}"):
                try:
                    with db_lock:
                        c.execute("DELETE FROM medications WHERE id=?", (mid,))
                        c.execute("DELETE FROM med_logs WHERE med_id=?", (mid,))
                        conn.commit()
                    st.success("Deleted medication.")
                    safe_rerun()
                except Exception as e:
                    st.error("Delete failed.")
            if col3.button("Taken", key=f"taken_{mid}"):
                ok = mark_dose(mid, uid, "taken", "Dose taken by user")
                if ok:
                    st.success("Marked as taken.")
                else:
                    st.error("Failed to mark as taken.")
            if col4.button("Missed", key=f"missed_{mid}"):
                ok = mark_dose(mid, uid, "missed", "Dose missed by user")
                if ok:
                    notify_all(uid, "Missed Dose Alert", f"<p>{uname} missed {row['name']} ({row['dosage']}).</p><p>— CareSync</p>")
                    st.error("Marked as missed — notifications sent.")
                else:
                    st.error("Failed to mark as missed.")
            if st.session_state.get(f"editing_{mid}"):
                st.markdown("#### Edit medication")
                ename = st.text_input("Name", value=row["name"], key=f"ename_{mid}")
                edosage = st.text_input("Dosage", value=row["dosage"], key=f"edosage_{mid}")
                efreq = st.text_input("Frequency", value=row["frequency"], key=f"efreq_{mid}")
                ert = st.text_input("Reminder times", value=row["reminder_time"], key=f"ert_{mid}")
                try:
                    esd_val = datetime.fromisoformat(row["start_date"]).date() if row["start_date"] else date.today()
                except:
                    esd_val = date.today()
                try:
                    eed_val = datetime.fromisoformat(row["end_date"]).date() if row["end_date"] else date.today()+timedelta(days=30)
                except:
                    eed_val = date.today()+timedelta(days=30)
                esd = st.date_input("Start date", value=esd_val, key=f"esd_{mid}")
                eed = st.date_input("End date", value=eed_val, key=f"eed_{mid}")
                enotes = st.text_area("Notes", value=row.get("notes",""), key=f"enotes_{mid}")
                if st.button("Save changes", key=f"save_{mid}"):
                    try:
                        with db_lock:
                            c.execute("UPDATE medications SET name=?,dosage=?,frequency=?,reminder_time=?,start_date=?,end_date=?,notes=? WHERE id=?",
                                      (ename.strip(), edosage.strip(), efreq.strip(), ert.strip(), str(esd), str(eed), enotes.strip(), mid))
                            conn.commit()
                        st.success("Medication updated.")
                        st.session_state.pop(f"editing_{mid}", None)
                        safe_rerun()
                    except Exception as e:
                        st.error("Update failed.")
                        print("Update med error:", e)
# ---------------- Part 5/5 ----------------
# Main Streamlit UI (Home, Sign Up, Login, Dashboard, Medications, Caregivers, AI Assistant, General chatbot, Logout)

st.set_page_config(page_title="CareSync", layout="wide")
menu = ["Home", "Sign Up", "Login"]
choice = st.sidebar.selectbox("Menu", menu)

if choice == "Home":
    st.title("CareSync — Smart Medication Tracking")
    st.write("""
    CareSync features:
    - Prescription upload (Tesseract OCR) + Gemini 2.0 Flash text parsing
    - Aggressive AI guessing & auto-fill for noisy OCR (with notes)
    - Medication list with Edit / Delete / Taken / Missed
    - Email reminders & weekly summary
    - Caregiver linking
    - AI assistant (medical) using your medication data
    - General AI chatbot (does NOT use medication data)
    """)

elif choice == "Sign Up":
    st.header("Create Account")
    name = st.text_input("Full Name", key="su_name")
    email = st.text_input("Email", key="su_email")
    phone = st.text_input("Phone", key="su_phone")
    password = st.text_input("Password", type="password", key="su_pass")
    role = st.selectbox("Role", ["patient", "caregiver"], key="su_role")
    if st.button("Sign Up", key="su_btn"):
        ok = signup(name, email, phone, password, role)
        if ok:
            st.success("Account created. Please log in.")
        else:
            st.error("Signup failed — email may already exist or invalid input.")

elif choice == "Login":
    st.header("Login")
    email = st.text_input("Email", key="li_email")
    password = st.text_input("Password", type="password", key="li_pw")
    if st.button("Login", key="li_btn"):
        user = login(email, password)
        if user:
            uid, uname, role = user
            st.session_state["uid"] = uid
            st.session_state["uname"] = uname
            st.session_state["role"] = role
            st.success(f"Welcome {uname}")
        else:
            st.error("Invalid credentials or DB error. Check terminal logs.")

# Authenticated area
if "uid" in st.session_state:
    uid = st.session_state["uid"]
    uname = st.session_state["uname"]
    role = st.session_state["role"]
    st.sidebar.markdown(f"**Logged in:** {uname} ({role})")
    page = st.sidebar.radio("Navigate", ["Dashboard", "Medications", "Caregivers", "AI Assistant (Medical)", "General AI Chatbot", "Logout"])
    if page == "Dashboard":
        st.header("Dashboard")
        meds_df = pd.read_sql("SELECT name,dosage,frequency,reminder_time,start_date,end_date,notes FROM medications WHERE patient_id=?", conn, params=(uid,))
        st.subheader("My Medications")
        if meds_df.empty:
            st.info("No medications yet.")
        else:
            st.dataframe(meds_df)
        logs = pd.read_sql("SELECT ml.timestamp, ml.status, m.name AS medication, ml.note FROM med_logs ml JOIN medications m ON m.id=ml.med_id WHERE m.patient_id=? ORDER BY ml.timestamp DESC LIMIT 25", conn, params=(uid,))
        st.subheader("Recent Activity")
        if logs.empty:
            st.info("No activity yet.")
        else:
            st.dataframe(logs)
        if st.button("Send My Weekly Summary"):
            send_weekly_summary_for(uid)
            st.success("Weekly summary sent.")

    elif page == "Medications":
        medications_ui_for_user(uid, uname)

    elif page == "Caregivers":
        st.header("Caregiver Management")
        cg_email = st.text_input("Caregiver email", key="cg_email")
        cg_phone = st.text_input("Caregiver phone", key="cg_phone")
        cg_access = st.selectbox("Access level", ["full", "meds-only", "view-only"], key="cg_access")
        if st.button("Link Caregiver", key="link_cg"):
            cg = c.execute("SELECT id FROM users WHERE email=? AND phone=? AND role='caregiver'", (cg_email, cg_phone)).fetchone()
            if cg:
                with db_lock:
                    c.execute("INSERT INTO caregivers (patient_id, caregiver_id, caregiver_phone, access_level) VALUES (?, ?, ?, ?)", (uid, cg[0], cg_phone, cg_access))
                    conn.commit()
                st.success("Caregiver linked.")
            else:
                st.error("Caregiver not found — ensure they signed up as role='caregiver' with matching phone.")
        st.subheader("My Caregivers")
        cg_df = pd.read_sql("SELECT u.name,u.email,u.phone,c.access_level FROM caregivers c JOIN users u ON u.id=c.caregiver_id WHERE c.patient_id=?", conn, params=(uid,))
        if cg_df.empty:
            st.info("No caregivers linked.")
        else:
            st.dataframe(cg_df)

    elif page == "AI Assistant (Medical)":
        st.header("AI Medical Assistant (uses your medication data)")
        user_q = st.text_input("Ask about your medicines or adherence", key="med_q")
        if st.button("Ask", key="med_ask"):
            meds = c.execute("SELECT name,dosage,frequency,reminder_time FROM medications WHERE patient_id=?", (uid,)).fetchall()
            meds_text = "\n".join([f"- {m[0]} ({m[1]}), {m[2]}, reminders: {m[3]}" for m in meds]) or "No medications on file."
            prompt = f"You are a helpful medical assistant. Patient medications:\n{meds_text}\n\nQuestion: {user_q}\nAnswer concisely in simple language and give lifestyle tips and adherence suggestions."
            out = ""
            err = None
            if GEMINI_API_KEY:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512}}
                    r = requests.post(url, json=body, headers={"Content-Type":"application/json"}, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    candidates = data.get("candidates") or []
                    if candidates:
                        cand = candidates[0]
                        content = cand.get("content") or {}
                        out_text = ""
                        if isinstance(content, list):
                            for p in content:
                                if isinstance(p, dict):
                                    out_text += p.get("text","")
                        elif isinstance(content, dict):
                            for p in content.get("parts",[]):
                                out_text += p.get("text","")
                        if not out_text:
                            out_text = cand.get("text") or json.dumps(cand)
                        out = out_text
                except Exception as e:
                    err = f"Gemini error: {e}"
            else:
                out = "Gemini API key not configured. Set GEMINI_API_KEY in .env to enable AI assistant."
            if err:
                st.error(err)
            else:
                st.markdown("### AI Reply")
                st.write(out)

    elif page == "General AI Chatbot":
        st.header("General AI Chatbot (does NOT use your medication data)")
        q = st.text_input("Ask anything", key="g_q")
        if st.button("Ask AI", key="g_ask"):
            if GEMINI_API_KEY:
                try:
                    url = f"https://generativelanguage.googleapis.com/v1/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
                    body = {"contents": [{"parts": [{"text": q}]}], "generationConfig": {"temperature": 0.7, "maxOutputTokens": 512}}
                    r = requests.post(url, json=body, headers={"Content-Type":"application/json"}, timeout=60)
                    r.raise_for_status()
                    data = r.json()
                    candidates = data.get("candidates") or []
                    if candidates:
                        cand = candidates[0]
                        content = cand.get("content") or {}
                        out_text = ""
                        if isinstance(content, list):
                            for p in content:
                                if isinstance(p, dict):
                                    out_text += p.get("text","")
                        elif isinstance(content, dict):
                            for p in content.get("parts",[]):
                                out_text += p.get("text","")
                        if not out_text:
                            out_text = cand.get("text") or json.dumps(cand)
                        st.write(out_text)
                except Exception as e:
                    st.error("AI error: " + str(e))
            else:
                st.error("Gemini API key not configured. Set GEMINI_API_KEY in .env to use this chat.")

    elif page == "Logout":
        st.session_state.clear()
        st.success("Logged out.")
        safe_rerun()

import streamlit as st
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import requests
import PyPDF2
import re
import io
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime
from google import genai
import os
client = genai.Client(api_key="AIzaSyBMdpMoNcEt8YPTp5GsdDtaPSRsZQ2UIDU")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="GlucoBot", page_icon="🧪", layout="centered")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_KEY = "AIzaSyBp1LfOJ6Yqfasbo_aSIcDoU8ozO12z-HA"
# GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"



def ask_gemini(user_msg, history):
    try:
        prompt = f"""
You are GlucoBot, a diabetes-focused AI assistant.
Give short, practical answers (2–3 lines).
Always suggest consulting a doctor for serious issues.

User: {user_msg}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[{
                "role": "user",
                "parts": [{"text": prompt}]
            }]
        )

        return response.candidates[0].content.parts[0].text

    except Exception as e:
        return f"Error: {str(e)}"

# ── Session State ─────────────────────────────────────────────────────────────
for k, v in {"page":"home","glucose":120,"bp":70,"age":30,"insulin":80,"bmi":25.0,
              "msgs":[],"chat_history":[],"risk_result":None,"risk_inputs":None,"pdf_done":False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.main { background: #030712 !important; }
.hero { text-align:center; padding:48px 32px; border-radius:24px;
  background:linear-gradient(135deg,#0f172a,#1e1b4b);
  border:1px solid rgba(99,102,241,0.3); margin-bottom:28px;
  box-shadow:0 0 100px rgba(99,102,241,0.12); }
.hero-title { font-size:64px; font-weight:900; letter-spacing:-2px; line-height:1.1;
  background:linear-gradient(90deg,#818cf8,#22d3ee,#4ade80);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero-sub { font-size:20px; color:#cbd5e1; margin-top:10px; font-weight:500; }
.hero-desc { color:#64748b; margin-top:6px; font-size:13px; }
.fcard { border-radius:20px; padding:32px 20px;
  background:linear-gradient(135deg,#0f172a,#1a1040);
  border:1px solid rgba(99,102,241,0.2); text-align:center;
  min-height:200px; display:flex; flex-direction:column;
  align-items:center; justify-content:center; gap:10px; margin-bottom:10px; }
.fcard-icon { font-size:48px; }
.fcard-title { font-size:17px; font-weight:700; color:#e2e8f0; }
.fcard-desc { font-size:12px; color:#64748b; line-height:1.4; }
.chat-box { background:#0a0f1e; border-radius:18px; padding:20px;
  height:400px; overflow-y:auto; border:1px solid rgba(99,102,241,0.15); margin-bottom:12px; }
.bot-lbl { font-size:10px; color:#818cf8; margin-top:14px; font-weight:700;
  letter-spacing:0.08em; text-transform:uppercase; }
.bot-msg { background:linear-gradient(135deg,#3730a3,#4f46e5); color:white;
  padding:11px 15px; border-radius:16px 16px 16px 3px; margin:4px 0;
  max-width:78%; font-size:14px; line-height:1.55; display:inline-block; }
.usr-msg { background:#1e293b; color:#e2e8f0; padding:11px 15px;
  border-radius:16px 16px 3px 16px; margin:4px 0; margin-left:auto;
  max-width:78%; font-size:14px; line-height:1.55; text-align:right; display:block; }
.page-hdr { font-size:28px; font-weight:800; color:#e2e8f0;
  margin-bottom:20px; letter-spacing:-0.5px; }
.stButton>button { background:linear-gradient(90deg,#4f46e5,#0ea5e9) !important;
  color:white !important; border:none !important; border-radius:12px !important;
  font-weight:600 !important; font-size:14px !important; padding:10px !important; }
</style>""", unsafe_allow_html=True)

# ── PDF Generator ─────────────────────────────────────────────────────────────
def generate_pdf(inputs, prob, label):
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)

    rc = {'LOW RISK':('#166534','#f0fdf4','#4ade80'),
          'MODERATE RISK':('#854d0e','#fefce8','#facc15'),
          'HIGH RISK':('#991b1b','#fef2f2','#f87171')}
    dark, light, accent = rc[label]

    def ps(name, **kw): return ParagraphStyle(name, **kw)

    story = []
    story.append(Paragraph("GlucoBot Health Report",
        ps('t', fontSize=26, fontName='Helvetica-Bold', textColor=colors.HexColor('#4f46e5'), spaceAfter=6)))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y  —  %I:%M %p')}",
        ps('d', fontSize=10, textColor=colors.HexColor('#94a3b8'), spaceAfter=16)))
    story.append(Table([['']], colWidths=[495], rowHeights=[2],
        style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#4f46e5'))])))
    story.append(Spacer(1,14))

    # Risk badge
    story.append(Table(
        [[Paragraph(f"Diabetes Risk: {label}", ps('rl', fontSize=15, fontName='Helvetica-Bold', textColor=colors.HexColor(dark))),
          Paragraph(f"{prob*100:.1f}%", ps('rp', fontSize=30, fontName='Helvetica-Bold', textColor=colors.HexColor(dark), alignment=2))]],
        colWidths=[350,145],
        style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor(light)),
                          ('PADDING',(0,0),(-1,-1),14),('VALIGN',(0,0),(-1,-1),'MIDDLE')])))
    story.append(Spacer(1,18))

    # Pie chart
    fig, ax = plt.subplots(figsize=(4,4), facecolor='white')
    ax.pie([prob, 1-prob], colors=[accent,'#e5e7eb'], startangle=90,
           wedgeprops=dict(width=0.28, edgecolor='white', linewidth=2))
    ax.text(0, 0.08, f"{prob*100:.1f}%", ha='center', va='center',
            fontsize=28, fontweight='bold', color=accent)
    ax.text(0,-0.22, label, ha='center', va='center', fontsize=9, color=dark)
    ax.axis('off')
    plt.tight_layout(pad=0.3)
    cb = io.BytesIO()
    fig.savefig(cb, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    cb.seek(0); plt.close(fig)

    story.append(Table([[RLImage(cb, width=2.8*inch, height=2.8*inch)]],
        colWidths=[495], style=TableStyle([('ALIGN',(0,0),(-1,-1),'CENTER')])))
    story.append(Spacer(1,18))

    # Input table
    story.append(Paragraph("Patient Input Summary",
        ps('h', fontSize=13, fontName='Helvetica-Bold',
           textColor=colors.HexColor('#1e293b'), spaceBefore=8, spaceAfter=8)))
    rows = [['Parameter','Your Value','Normal Range'],
            ['Glucose (mg/dL)', str(inputs['glucose']), '70 – 99'],
            ['Blood Pressure (mm Hg)', str(inputs['bp']), '60 – 80'],
            ['BMI (kg/m²)', str(inputs['bmi']), '18.5 – 24.9'],
            ['Insulin (μU/mL)', str(inputs['insulin']), '16 – 166'],
            ['Age (years)', str(inputs['age']), '–'],
            ['Pregnancies', str(inputs['preg']), '–'],
            ['Skin Thickness (mm)', str(inputs['skin']), '–'],
            ['Diabetes Pedigree', str(inputs['dpf']), '–']]
    t = Table(rows, colWidths=[210,145,140])
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#4f46e5')),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),10),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.HexColor('#f8fafc'),colors.white]),
        ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#e2e8f0')),
        ('PADDING',(0,0),(-1,-1),9),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
    ]))
    story.append(t)
    story.append(Spacer(1,18))

    # Recommendations
    story.append(Paragraph("Personalised Recommendations",
        ps('h2', fontSize=13, fontName='Helvetica-Bold',
           textColor=colors.HexColor('#1e293b'), spaceBefore=8, spaceAfter=8)))
    recs = {
        'LOW RISK':["✅ Maintain your current healthy lifestyle — you're doing great!",
                    "🥗 Continue a balanced diet rich in vegetables, whole grains, and lean protein.",
                    "🏃 Keep regular physical activity — aim for 150 minutes per week.",
                    "🩺 Schedule yearly screening to stay on track.",
                    "💧 Stay well hydrated and limit sugary beverages."],
        'MODERATE RISK':["⚠️ Moderate risk detected — lifestyle changes can make a big difference.",
                         "🥗 Reduce refined carbohydrates and sugar intake significantly.",
                         "🏃 Increase physical activity — aim for 30 minutes of exercise daily.",
                         "⚖️ Losing 5–7% body weight can reduce diabetes risk by up to 58%.",
                         "🩺 Get a fasting glucose test every 6 months.",
                         "😴 Prioritise 7–8 hours of quality sleep each night."],
        'HIGH RISK':["🚨 High risk detected — please consult a doctor as soon as possible.",
                     "🏥 A doctor should review your glucose and insulin levels in person.",
                     "🥗 Follow a strict low-glycaemic diet — avoid all processed sugars.",
                     "🏃 Begin a supervised exercise program with your doctor's guidance.",
                     "💊 Ask your doctor about preventive medications like Metformin.",
                     "📊 Monitor your blood glucose daily if possible.",
                     "🧘 Manage stress — chronic stress raises blood sugar levels."],
    }[label]
    bsty = ps('b', fontSize=10, textColor=colors.HexColor('#374151'), leading=16, spaceAfter=5)
    for r in recs:
        story.append(Paragraph(r, bsty))
    story.append(Spacer(1,20))

    # Disclaimer
    story.append(Table([[Paragraph(
        "⚠️ Disclaimer: This report is generated by an AI system for informational purposes only. "
        "It is not a substitute for professional medical advice, diagnosis, or treatment. "
        "Always consult a qualified healthcare provider for medical decisions.",
        ps('dis', fontSize=8, textColor=colors.HexColor('#94a3b8'), leading=12))]],
        colWidths=[495],
        style=TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f8fafc')),
                          ('PADDING',(0,0),(-1,-1),12),
                          ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#e2e8f0'))])))
    doc.build(story)
    buf.seek(0)
    return buf

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "home":
    st.markdown("""<div class="hero">
        <div class="hero-title">GlucoBot</div>
        <div class="hero-sub">Because sugar shall stay in dessert.</div>
        <div class="hero-desc">AI-powered diabetes risk assessment & health guidance</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### 📄 Upload Medical Report *(Optional)*")
    file = st.file_uploader("Upload PDF", type=['pdf'], label_visibility="collapsed")
    if file and not st.session_state.pdf_done:
        try:
            reader = PyPDF2.PdfReader(file)
            text = " ".join(p.extract_text() or "" for p in reader.pages)
            def fv(k):
                m = re.search(k+r".{0,20}?(\d+\.?\d*)", text, re.I)
                return int(float(m.group(1))) if m else None
            st.session_state.glucose = fv("glucose") or st.session_state.glucose
            st.session_state.bp      = fv("pressure") or st.session_state.bp
            st.session_state.age     = fv("age") or st.session_state.age
            st.session_state.insulin = fv("insulin") or st.session_state.insulin
            bm = re.search(r"bmi.{0,15}?(\d+\.?\d*)", text, re.I)
            if bm: st.session_state.bmi = float(bm.group(1))
            st.session_state.pdf_done = True
            st.success("✅ Report processed! Values auto-filled in Risk Scan.")
            st.rerurn()
        except:
            st.error("Could not read PDF. Please try a different file.")

    st.markdown("---")
    st.markdown("#### Choose a Feature")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""<div class="fcard"><div class="fcard-icon">🧪</div>
            <div class="fcard-title">Risk Scan</div>
            <div class="fcard-desc">AI evaluates your diabetes probability</div></div>""", unsafe_allow_html=True)
        if st.button("Start Health Scan →", use_container_width=True, key="b1"):
            st.session_state.page = "test"; st.rerun()

    with c2:
        st.markdown("""<div class="fcard"><div class="fcard-icon">🤖</div>
            <div class="fcard-title">AI Chat</div>
            <div class="fcard-desc">Talk to GlucoBot powered by Gemini AI</div></div>""", unsafe_allow_html=True)
        if st.button("Start Conversation →", use_container_width=True, key="b2"):
            st.session_state.page = "chat"; st.rerun()

    with c3:
        st.markdown("""<div class="fcard"><div class="fcard-icon">📊</div>
            <div class="fcard-title">My Report</div>
            <div class="fcard-desc">Download your last health scan as PDF</div></div>""", unsafe_allow_html=True)
        if st.button("View Report →", use_container_width=True, key="b3"):
            st.session_state.page = "report"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# RISK SCAN
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "test":
    st.markdown('<div class="page-hdr">🧪 AI Diabetes Risk Scan</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        preg    = st.number_input("Pregnancies", 0, 15, 1, key="k_preg")
        glucose = st.number_input("Glucose (mg/dL)", 50, 250, int(st.session_state.glucose), key="k_glucose")
        bp      = st.number_input("Blood Pressure (mm Hg)", 40, 140, int(st.session_state.bp), key="k_bp")
        skin    = st.number_input("Skin Thickness (mm)", 0, 100, 20, key="k_skin")
    with c2:
        insulin = st.number_input("Insulin (μU/mL)", 0, 500, int(st.session_state.insulin), key="k_insulin")
        bmi     = st.number_input("BMI (kg/m²)", 10.0, 70.0, float(st.session_state.bmi), key="k_bmi")
        dpf     = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, step=0.01, key="k_dpf")
        age     = st.number_input("Age (years)", 10, 90, int(st.session_state.age), key="k_age")

    if st.button("🔬 Run AI Risk Analysis", use_container_width=True):
        # Feature engineering must match train.py
        bmi_age         = bmi * age
        glucose_insulin = glucose * insulin
        glucose_bmi     = glucose * bmi
        data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age,
                          bmi_age, glucose_insulin, glucose_bmi]])
        prob = model.predict_proba(scaler.transform(data))[0][1]

        placeholder = st.empty()
        target = int(prob * 100)
        for i in range(0, target + 1, max(1, target // 30)):
            v = i / 100
            c = '#4ade80' if v < 0.35 else '#facc15' if v < 0.65 else '#f87171'
            fig, ax = plt.subplots(figsize=(3, 3), facecolor='none')
            ax.pie([v, 1-v], startangle=90, colors=[c, '#1e293b'],
                   wedgeprops=dict(width=0.24, edgecolor='#030712', linewidth=2))
            ax.text(0, 0, f"{i}%", ha='center', va='center',
                    fontsize=26, color=c, fontweight='bold')
            ax.axis('off'); fig.patch.set_alpha(0)
            placeholder.pyplot(fig); plt.close(fig); time.sleep(0.015)

        if prob < 0.35:
            label = "LOW RISK"
            st.success("✅ Low risk! Keep maintaining your healthy lifestyle.")
        elif prob < 0.65:
            label = "MODERATE RISK"
            st.warning("⚠️ Moderate risk. Lifestyle changes now can significantly help.")
        else:
            label = "HIGH RISK"
            st.error("🚨 High risk. Please consult a doctor and share your report with them.")

        st.session_state.risk_result = (prob, label)
        st.session_state.risk_inputs = {'preg':preg,'glucose':glucose,'bp':bp,
                                         'skin':skin,'insulin':insulin,'bmi':bmi,'dpf':dpf,'age':age}
        st.markdown("---")
        if st.button("📄 Generate PDF Report", use_container_width=True):
            st.session_state.page = "report"; st.rerun()

    if st.button("← Back", key="back_t"):
        st.session_state.page = "home"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "report":
    st.markdown('<div class="page-hdr">📊 Your Health Report</div>', unsafe_allow_html=True)
    if st.session_state.risk_result is None:
        st.warning("No scan results yet. Please run the Risk Scan first.")
        if st.button("Go to Risk Scan"):
            st.session_state.page = "test"; st.rerun()
    else:
        prob, label = st.session_state.risk_result
        inputs = st.session_state.risk_inputs
        icons = {'LOW RISK':'🟢','MODERATE RISK':'🟡','HIGH RISK':'🔴'}
        st.markdown(f"**{icons[label]} Risk Level:** `{label}` &nbsp; **Score:** `{prob*100:.1f}%`")
        st.markdown("---")
        with st.spinner("Generating your personalised PDF report..."):
            pdf_buf = generate_pdf(inputs, prob, label)
        st.download_button("⬇️ Download Full PDF Report", data=pdf_buf,
                           file_name=f"GlucoBot_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                           mime="application/pdf", use_container_width=True)
        st.markdown("---")
        st.markdown("### Quick Summary")
        recs = {'LOW RISK':["✅ Maintain healthy lifestyle","🩺 Yearly screening","🏃 Stay active 150 min/week"],
                'MODERATE RISK':["🥗 Reduce sugar & carbs","🏃 30 min daily exercise","⚖️ Lose 5-7% body weight if overweight"],
                'HIGH RISK':["🚨 Consult a doctor immediately","📊 Monitor glucose daily","🥗 Follow low-glycaemic diet"]}[label]
        for r in recs: st.markdown(f"- {r}")
    if st.button("← Back", key="back_r"):
        st.session_state.page = "home"; st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif st.session_state.page == "chat":
    st.markdown('<div class="page-hdr">🤖 GlucoBot AI Chat</div>', unsafe_allow_html=True)
    st.markdown("*Powered by Google Gemini — Ask anything about diabetes, diet, or lifestyle!*")

    chat_html = "<div class='chat-box'>"
    if not st.session_state.msgs:
        chat_html += "<div class='bot-lbl'>GlucoBot</div>"
        chat_html += "<div class='bot-msg'>👋 Hi! I'm GlucoBot, your AI health assistant. Ask me anything about diabetes, blood sugar, diet, or lifestyle!</div>"
    for m in st.session_state.msgs:
        if m["role"] == "bot":
            chat_html += f"<div class='bot-lbl'>GlucoBot</div><div class='bot-msg'>{m['text']}</div>"
        else:
            chat_html += f"<div class='usr-msg'>{m['text']}</div>"
    chat_html += "</div>"
    st.markdown(chat_html, unsafe_allow_html=True)

    msg = st.text_input("Message", placeholder="e.g. What foods should I avoid with high blood sugar?",
                        label_visibility="collapsed")
    col1, col2, col3 = st.columns([2,2,3])
    with col1:
        send = st.button("Send 📤", use_container_width=True)
    with col2:
        if st.button("Clear 🗑️", use_container_width=True):
            st.session_state.msgs = []; st.session_state.chat_history = []; st.rerun()
    with col3:
        if st.button("← Back to Home", use_container_width=True, key="back_c"):
            st.session_state.page = "home"; st.rerun()

    if send and msg.strip():
        st.session_state.msgs.append({"role":"user","text":msg})
        st.session_state.chat_history.append({"role":"user","content":msg})
        with st.spinner("GlucoBot is thinking..."):
            reply = ask_gemini(msg, st.session_state.chat_history)
        st.session_state.msgs.append({"role":"bot","text":reply})
        st.session_state.chat_history.append({"role":"assistant","content":reply})
        st.rerun()

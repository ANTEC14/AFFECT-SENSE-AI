import gradio as gr
from google import genai
from google.genai import types
from transformers import pipeline
from datetime import datetime
import matplotlib.pyplot as plt
import os


# ==================== CONFIGURATION ====================

DEFAULT_API_KEY = "AIzaSyBmRNFscM-NoxFBPjIulWDyRU_YI7clXow"
API_KEY = os.getenv("GEMINI_API_KEY", DEFAULT_API_KEY)

SYSTEM_PROMPT = """You are a friendly empathetic mental health companion.
Use detected emotional signals to respond supportively.
Keep responses concise (3–5 sentences).
"""

RATE_LIMIT = 20
MESSAGE_DELAY = 2

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=Lora:ital,wght@0,400;0,600;1,400&display=swap');

/* ══════════════════════════════════════════
   ROOT TOKENS
══════════════════════════════════════════ */
:root {
  --ink:       #0a0e1a;
  --ink2:      #141929;
  --ink3:      #1c2438;
  --surface:   #212840;
  --border:    rgba(99,179,237,0.14);
  --border-hi: rgba(99,179,237,0.32);
  --sky:       #63b3ed;
  --sky2:      #90cdf4;
  --teal:      #4fd1c5;
  --rose:      #fc8181;
  --text:      #e8edf8;
  --muted:     #8892aa;
  --glow:      rgba(99,179,237,0.18);
  --shadow:    0 28px 72px rgba(0,0,0,0.6);
}

/* ══════════════════════════════════════════
   RESET & BASE
══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

body {
  font-family: 'Sora', sans-serif !important;
  background: var(--ink) !important;
  color: var(--text) !important;
  min-height: 100vh;
  overflow-x: hidden;
}

.gradio-container {
  background: transparent !important;
  max-width: 100% !important;
  padding: 0 !important;
  margin: 0 !important;
}

footer { display: none !important; }

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--ink2); }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 4px; }

/* ══════════════════════════════════════════
   NOISE OVERLAY (atmospheric grain)
══════════════════════════════════════════ */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E");
  opacity: 0.028;
  pointer-events: none;
  z-index: 0;
}

/* ══════════════════════════════════════════
   NAV
══════════════════════════════════════════ */
.as-nav {
  position: sticky;
  top: 0;
  z-index: 100;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 48px;
  height: 64px;
  background: rgba(10,14,26,0.82);
  backdrop-filter: blur(18px);
  border-bottom: 1px solid var(--border);
}

.as-nav-logo {
  display: flex;
  align-items: center;
  gap: 10px;
}

.as-nav-logo-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--sky), var(--teal));
  box-shadow: 0 0 10px var(--sky);
  animation: pulse-dot 3s ease-in-out infinite;
}

@keyframes pulse-dot {
  0%,100% { box-shadow: 0 0 8px var(--sky); transform: scale(1); }
  50%      { box-shadow: 0 0 18px var(--teal); transform: scale(1.2); }
}

.as-nav-logo span {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: -0.01em;
  background: linear-gradient(90deg, var(--sky2), var(--teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.as-nav-links {
  display: flex;
  gap: 34px;
}

.as-nav-links a {
  color: var(--muted);
  font-size: 13.5px;
  font-weight: 500;
  letter-spacing: 0.04em;
  text-decoration: none;
  text-transform: uppercase;
  transition: color 0.2s;
}

.as-nav-links a:hover { color: var(--sky2); }

.as-nav-cta {
  background: linear-gradient(135deg, var(--sky) 0%, var(--teal) 100%);
  border: none;
  border-radius: 50px;
  color: var(--ink) !important;
  font-family: 'Sora', sans-serif;
  font-weight: 600;
  font-size: 13px;
  letter-spacing: 0.04em;
  padding: 9px 22px;
  cursor: pointer;
  box-shadow: 0 4px 18px var(--glow);
  transition: filter 0.18s, transform 0.18s;
  text-decoration: none;
}

.as-nav-cta:hover {
  filter: brightness(1.08);
  transform: translateY(-1px);
}

/* ══════════════════════════════════════════
   HERO
══════════════════════════════════════════ */
.as-hero {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  padding: 110px 24px 90px;
  overflow: hidden;
}

.as-hero::before {
  content: '';
  position: absolute;
  top: -120px; left: 50%;
  transform: translateX(-50%);
  width: 700px; height: 700px;
  background: radial-gradient(circle, rgba(99,179,237,0.10) 0%, rgba(79,209,197,0.05) 45%, transparent 72%);
  pointer-events: none;
}

.as-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 7px;
  background: rgba(99,179,237,0.10);
  border: 1px solid var(--border-hi);
  border-radius: 50px;
  padding: 6px 16px;
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: var(--sky2);
  margin-bottom: 28px;
  animation: fadein 0.7s ease both;
}

.as-hero-badge-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--teal);
  animation: pulse-dot 2s infinite;
}

.as-hero h1 {
  font-family: 'Lora', serif;
  font-size: clamp(38px, 6vw, 72px);
  font-weight: 600;
  line-height: 1.10;
  letter-spacing: -0.02em;
  max-width: 820px;
  margin-bottom: 24px;
  animation: fadein 0.8s 0.1s ease both;
}

.as-hero h1 em {
  font-style: italic;
  background: linear-gradient(90deg, var(--sky), var(--teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.as-hero p {
  max-width: 560px;
  color: var(--muted);
  font-size: 17px;
  font-weight: 300;
  line-height: 1.7;
  margin-bottom: 44px;
  animation: fadein 0.8s 0.2s ease both;
}

.as-hero-btns {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
  animation: fadein 0.8s 0.3s ease both;
}

.as-btn-primary {
  background: linear-gradient(135deg, var(--sky) 0%, var(--teal) 100%);
  border: none;
  border-radius: 50px;
  color: var(--ink) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  padding: 15px 38px !important;
  cursor: pointer;
  box-shadow: 0 8px 30px rgba(99,179,237,0.25);
  transition: filter 0.18s, transform 0.18s, box-shadow 0.18s;
  text-decoration: none;
  display: inline-block;
}

.as-btn-primary:hover {
  filter: brightness(1.07);
  transform: translateY(-2px);
  box-shadow: 0 14px 44px rgba(99,179,237,0.38);
}

.as-btn-ghost {
  background: transparent;
  border: 1px solid var(--border-hi);
  border-radius: 50px;
  color: var(--text) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 500 !important;
  font-size: 15px !important;
  padding: 14px 36px !important;
  cursor: pointer;
  transition: background 0.18s, border-color 0.18s;
  text-decoration: none;
  display: inline-block;
}

.as-btn-ghost:hover {
  background: rgba(99,179,237,0.08);
  border-color: var(--sky);
}

/* Stats row */
.as-hero-stats {
  display: flex;
  gap: 48px;
  margin-top: 64px;
  animation: fadein 0.9s 0.4s ease both;
}

.as-stat {
  text-align: center;
}

.as-stat-num {
  font-family: 'Lora', serif;
  font-size: 32px;
  font-weight: 600;
  background: linear-gradient(90deg, var(--sky2), var(--teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  display: block;
  line-height: 1;
  margin-bottom: 6px;
}

.as-stat-label {
  font-size: 12px;
  color: var(--muted);
  letter-spacing: 0.05em;
  text-transform: uppercase;
  font-weight: 500;
}

/* ══════════════════════════════════════════
   SECTION WRAPPER
══════════════════════════════════════════ */
.as-section {
  padding: 96px 24px;
  max-width: 1100px;
  margin: 0 auto;
}

.as-section-label {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--teal);
  margin-bottom: 16px;
}

.as-section-title {
  font-family: 'Lora', serif;
  font-size: clamp(28px, 4vw, 44px);
  font-weight: 600;
  line-height: 1.18;
  letter-spacing: -0.015em;
  margin-bottom: 18px;
}

.as-section-sub {
  color: var(--muted);
  font-size: 16px;
  font-weight: 300;
  line-height: 1.7;
  max-width: 600px;
}

/* ══════════════════════════════════════════
   DIVIDER
══════════════════════════════════════════ */
.as-divider {
  width: 100%;
  height: 1px;
  background: linear-gradient(90deg, transparent 0%, var(--border-hi) 30%, var(--border-hi) 70%, transparent 100%);
  margin: 0 auto;
}

/* ══════════════════════════════════════════
   ABOUT SECTION
══════════════════════════════════════════ */
.as-about-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 64px;
  align-items: center;
  margin-top: 56px;
}

.as-about-text p {
  color: var(--muted);
  font-size: 15.5px;
  line-height: 1.8;
  margin-bottom: 20px;
  font-weight: 300;
}

.as-about-text p strong {
  color: var(--text);
  font-weight: 600;
}

.as-about-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 32px;
}

.as-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(99,179,237,0.08);
  border: 1px solid var(--border-hi);
  border-radius: 50px;
  padding: 7px 16px;
  font-size: 13px;
  font-weight: 500;
  color: var(--sky2);
}

.as-pill-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--teal);
}

/* About visual card */
.as-about-visual {
  background: linear-gradient(160deg, var(--ink3) 0%, var(--ink2) 100%);
  border: 1px solid var(--border-hi);
  border-radius: 20px;
  padding: 32px;
  box-shadow: var(--shadow);
  position: relative;
  overflow: hidden;
}

.as-about-visual::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: radial-gradient(circle, rgba(79,209,197,0.12), transparent 70%);
  pointer-events: none;
}

.as-visual-row {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 14px 0;
  border-bottom: 1px solid var(--border);
}

.as-visual-row:last-child { border-bottom: none; }

.as-visual-icon {
  width: 40px; height: 40px;
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
  flex-shrink: 0;
}

.as-visual-icon.sky  { background: rgba(99,179,237,0.12); }
.as-visual-icon.teal { background: rgba(79,209,197,0.12); }
.as-visual-icon.rose { background: rgba(252,129,129,0.12); }
.as-visual-icon.gold { background: rgba(236,201,75,0.12); }

.as-visual-row-text strong {
  display: block;
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 2px;
}

.as-visual-row-text span {
  font-size: 12.5px;
  color: var(--muted);
  font-weight: 300;
}

/* ══════════════════════════════════════════
   FEATURES GRID
══════════════════════════════════════════ */
.as-features-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 22px;
  margin-top: 56px;
}

.as-feature-card {
  background: linear-gradient(160deg, var(--ink3) 0%, var(--ink2) 100%);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 28px 26px;
  transition: border-color 0.22s, transform 0.22s, box-shadow 0.22s;
  position: relative;
  overflow: hidden;
}

.as-feature-card::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(135deg, var(--glow), transparent 60%);
  opacity: 0;
  transition: opacity 0.22s;
  border-radius: 18px;
}

.as-feature-card:hover {
  border-color: var(--border-hi);
  transform: translateY(-4px);
  box-shadow: 0 18px 48px rgba(0,0,0,0.4);
}

.as-feature-card:hover::before { opacity: 1; }

.as-feature-icon {
  width: 48px; height: 48px;
  border-radius: 14px;
  background: rgba(99,179,237,0.10);
  border: 1px solid var(--border-hi);
  display: flex; align-items: center; justify-content: center;
  font-size: 22px;
  margin-bottom: 18px;
}

.as-feature-card h3 {
  font-size: 16px;
  font-weight: 600;
  margin-bottom: 10px;
  color: var(--text);
}

.as-feature-card p {
  font-size: 14px;
  color: var(--muted);
  line-height: 1.65;
  font-weight: 300;
}

/* ══════════════════════════════════════════
   CHAT SECTION
══════════════════════════════════════════ */
.as-chat-section {
  padding: 80px 24px 100px;
  max-width: 860px;
  margin: 0 auto;
}

.as-chat-header {
  text-align: center;
  margin-bottom: 42px;
}

.as-chat-wrap {
  background: linear-gradient(180deg, var(--ink3) 0%, var(--ink2) 100%);
  border: 1px solid var(--border-hi);
  border-radius: 24px;
  overflow: hidden;
  box-shadow: var(--shadow), inset 0 1px 0 rgba(99,179,237,0.10);
}

.as-chat-titlebar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 16px 22px;
  background: rgba(255,255,255,0.02);
  border-bottom: 1px solid var(--border);
}

.as-titlebar-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  background: rgba(99,179,237,0.18);
  border: 1px solid var(--border-hi);
}

.as-titlebar-dot.live {
  background: linear-gradient(135deg, var(--sky), var(--teal));
  border: none;
  animation: pulse-dot 2s infinite;
}

.as-titlebar-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--muted);
  letter-spacing: 0.05em;
  margin-left: 6px;
}

/* Override Gradio chatbot inside our wrapper */
#chat_card {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}

.gr-chatbot, #chat_card > div {
  background: transparent !important;
}

#chat_card .message {
  border-radius: 12px !important;
  padding: 12px 15px !important;
  font-size: 14.5px !important;
  line-height: 1.65 !important;
  font-family: 'Sora', sans-serif !important;
}

#chat_card .message.user {
  background: rgba(99,179,237,0.10) !important;
  border: 1px solid rgba(99,179,237,0.22) !important;
  color: var(--text) !important;
}

#chat_card .message.bot,
#chat_card .message.assistant {
  background: rgba(79,209,197,0.07) !important;
  border: 1px solid rgba(79,209,197,0.18) !important;
  color: var(--text) !important;
}

.as-chat-input-row {
  display: flex;
  gap: 12px;
  padding: 16px 20px;
  background: rgba(255,255,255,0.02);
  border-top: 1px solid var(--border);
  align-items: flex-end;
}

#msg_box {
  flex: 1;
}

#msg_box textarea,
#msg_box input {
  background: var(--ink2) !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 12px !important;
  padding: 12px 16px !important;
  color: var(--text) !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 14.5px !important;
  font-weight: 300 !important;
  caret-color: var(--sky) !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
  resize: none !important;
}

#msg_box textarea::placeholder,
#msg_box input::placeholder {
  color: var(--muted) !important;
  font-style: italic;
}

#msg_box textarea:focus,
#msg_box input:focus {
  outline: none !important;
  border-color: var(--sky) !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,0.14) !important;
}

#send_btn {
  background: linear-gradient(135deg, var(--sky) 0%, var(--teal) 100%) !important;
  border: none !important;
  border-radius: 12px !important;
  color: var(--ink) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 700 !important;
  font-size: 14px !important;
  letter-spacing: 0.03em !important;
  padding: 12px 24px !important;
  box-shadow: 0 4px 18px rgba(99,179,237,0.22) !important;
  transition: filter 0.15s ease, transform 0.15s ease !important;
  cursor: pointer !important;
  white-space: nowrap;
}

#send_btn:hover {
  filter: brightness(1.08) !important;
  transform: translateY(-1px) !important;
}

/* ══════════════════════════════════════════
   SUMMARY / ANALYTICS PANEL
══════════════════════════════════════════ */
.as-analytics-wrap {
  margin-top: 20px;
  display: flex;
  gap: 14px;
  align-items: center;
}

#summary_btn {
  background: transparent !important;
  border: 1px solid var(--border-hi) !important;
  border-radius: 12px !important;
  color: var(--sky2) !important;
  font-family: 'Sora', sans-serif !important;
  font-weight: 500 !important;
  font-size: 13.5px !important;
  letter-spacing: 0.03em !important;
  padding: 10px 22px !important;
  transition: background 0.15s ease, border-color 0.15s ease !important;
  cursor: pointer !important;
}

#summary_btn:hover {
  background: rgba(99,179,237,0.08) !important;
  border-color: var(--sky) !important;
}

#summary_output {
  background: var(--ink3) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 18px 20px !important;
  color: var(--text) !important;
  font-family: 'Sora', sans-serif !important;
  font-size: 14px !important;
  line-height: 1.75 !important;
  margin-top: 14px;
}

#summary_output h3 {
  font-family: 'Lora', serif !important;
  color: var(--sky2) !important;
  font-size: 18px !important;
  margin-bottom: 12px !important;
}

/* ══════════════════════════════════════════
   FOOTER
══════════════════════════════════════════ */
.as-footer {
  border-top: 1px solid var(--border);
  padding: 36px 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 16px;
}

.as-footer-logo {
  font-family: 'Lora', serif;
  font-size: 16px;
  font-weight: 600;
  background: linear-gradient(90deg, var(--sky2), var(--teal));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.as-footer p {
  color: var(--muted);
  font-size: 13px;
  font-weight: 300;
}

/* ══════════════════════════════════════════
   ANIMATIONS
══════════════════════════════════════════ */
@keyframes fadein {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ══════════════════════════════════════════
   RESPONSIVE
══════════════════════════════════════════ */
@media (max-width: 768px) {
  .as-nav { padding: 0 20px; }
  .as-nav-links { display: none; }
  .as-about-grid { grid-template-columns: 1fr; gap: 36px; }
  .as-features-grid { grid-template-columns: 1fr; }
  .as-hero-stats { gap: 28px; flex-wrap: wrap; justify-content: center; }
}
"""


# ==================== DATASET MODELS ====================

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

suicide_classifier = pipeline(
    "text-classification",
    model="sentinetyd/suicidality"
)


# ==================== INPUT VALIDATION ====================

def validate_input(text):
    if text is None:
        return False, "Please enter a message."
    text = str(text)
    if not text.strip():
        return False, "Please enter a message."
    if len(text) > 2000:
        return False, "Message too long."
    if len(set(text)) < 5 and len(text) > 20:
        return False, "Invalid message detected."
    return True, ""


# ==================== SUICIDE RISK DETECTION ====================

def detect_crisis(text):
    try:
        out = suicide_classifier(text)
        if not out:
            return False
        result = out[0]
        if not isinstance(result, dict):
            return False
        return result.get("label") == "suicide" and float(result.get("score", 0)) > 0.75
    except Exception:
        return False


def crisis_response():
    return """🚨 **I'm concerned about your safety.**

India: AASRA 91-22-27546669  
Vandrevala Foundation: 1860-2662-345  

Professional help is available 24/7.
"""


# ==================== EMOTION CLASSIFIER ====================

def analyze_sentiment(text):
    try:
        out = emotion_classifier(text)
        if not out:
            return "neutral", "😐", 0.0

        # The HF pipeline shape can vary:
        # - Flat list: [{'label': 'joy', 'score': 0.9}, ...]
        # - Nested list: [[{'label': 'joy', 'score': 0.9}, ...]]
        # We always select the max score across the returned candidates.
        top = None
        if isinstance(out, list):
            if out and isinstance(out[0], dict):
                # Flat list of candidates
                top = max(out, key=lambda x: x.get("score", 0))
            elif out and isinstance(out[0], list):
                # Nested list of candidates for each input
                top = max(out[0], key=lambda x: x.get("score", 0))
        elif isinstance(out, dict):
            top = out

        if not isinstance(top, dict):
            return "neutral", "😐", 0.0
        label = top.get("label", "neutral")
        confidence = round(float(top.get("score", 0)), 3)
    except Exception:
        return "neutral", "😐", 0.0

    emoji_map = {
        "joy": "😊", "sadness": "😔", "anger": "😠",
        "fear": "😰", "surprise": "😮", "love": "❤️", "neutral": "😐"
    }
    return label, emoji_map.get(label, "😐"), confidence


# ==================== GEMINI RESPONSE ====================

def ai_reply(user_input, emotion):
    try:
        if not API_KEY:
            return (
                "Gemini API key is missing. "
                "Set `GEMINI_API_KEY` in your environment and restart."
            )
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{SYSTEM_PROMPT}\nDetected emotion:{emotion}\nUser:{user_input}",
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)
            )
        )
        return response.text
    except Exception as e:
        # Don't crash the chat; surface the real reason for easier debugging.
        print("Gemini request failed:", repr(e), flush=True)
        err = str(e).strip() or type(e).__name__

        # Most common case: quota exhausted / billing issue.
        if ("RESOURCE_EXHAUSTED" in err) or (" 429" in err) or ("code:429" in err) or ("429" in err):
            return (
                "Quota exceeded (429 RESOURCE_EXHAUSTED).\n"
                "Check your Gemini billing/plan or wait for quota to reset."
            )

        return f"Request failed: {type(e).__name__}. {err[:160]}"


# ==================== ANALYTICS ====================

def emotion_statistics(messages):
    stats = {}
    for msg in messages:
        if msg["role"] == "user":
            emotion = msg["sentiment"][0]
            stats[emotion] = stats.get(emotion, 0) + 1
    return stats


def plot_emotions(messages):
    stats = emotion_statistics(messages)
    if not stats:
        return
    plt.bar(stats.keys(), stats.values())
    plt.title("Emotion Distribution")
    plt.savefig("emotion_chart.png")


# ==================== CHAT FORMATTER ====================

def format_messages_for_chatbot(msgs):
    chatbot_msgs = []
    for msg in msgs:
        timestamp = msg.get("timestamp", "")
        time_str = f"<span style='font-size:0.75em;color:#888'>{timestamp}</span><br>"
        role = msg.get("role")
        if role == "user":
            sentiment = msg.get("sentiment") or ("neutral", "😐", 0.0)
            emotion = sentiment[0] if len(sentiment) > 0 else "neutral"
            emoji   = sentiment[1] if len(sentiment) > 1 else "😐"
            confidence = sentiment[2] if len(sentiment) > 2 else 0.0
            content = f"{time_str}{emoji} {emotion} ({confidence})<br>{msg.get('content', '')}"
            chatbot_msgs.append({"role": "user", "content": content})
        elif role == "assistant":
            content = f"🧠 Affect Sense AI<br>{time_str}{msg.get('content', '')}"
            chatbot_msgs.append({"role": "assistant", "content": content})
        elif role == "system":
            chatbot_msgs.append({"role": "system", "content": msg.get("content", "")})
    return chatbot_msgs


# ==================== SUMMARY ENGINE ====================

def generate_summary(state):
    msgs = state["messages"]
    if len(msgs) < 4:
        return "Not enough data yet."
    user_msgs = [m for m in msgs if m["role"] == "user"]
    stats = emotion_statistics(msgs)
    dominant = max(stats, key=stats.get)
    avg_confidence = sum(m["sentiment"][2] for m in user_msgs) / len(user_msgs)
    variability = len(stats)
    plot_emotions(msgs)
    return f"""
### Conversation Summary

**Total messages:** {len(msgs)}

**Dominant emotion:** {dominant}

**Emotion variability score:** {variability}

**Average classifier confidence:** {avg_confidence:.2f}
"""


# ==================== MAIN SEND HANDLER ====================

def send(msg, state, dark_mode):
    try:
        valid, error = validate_input(msg)
        if not valid:
            yield format_messages_for_chatbot(state["messages"]), state, error, dark_mode, msg
            return

        timestamp = datetime.now().strftime("%I:%M:%S %p")
        emotion, emoji, confidence = analyze_sentiment(msg)
        sentiment = (emotion, emoji, confidence)

        state["messages"].append({
            "role": "user",
            "content": msg,
            "sentiment": sentiment,
            "timestamp": timestamp
        })

        if detect_crisis(msg):
            state["messages"].append({
                "role": "assistant",
                "content": crisis_response(),
                "timestamp": timestamp
            })
            yield format_messages_for_chatbot(state["messages"]), state, "", dark_mode, gr.update(value="")
            return

        yield format_messages_for_chatbot(state["messages"]), state, "Typing...", dark_mode, gr.update(value="")

        reply = ai_reply(msg, emotion)
        state["messages"].append({
            "role": "assistant",
            "content": reply,
            "timestamp": timestamp
        })

        yield format_messages_for_chatbot(state["messages"]), state, "", dark_mode, gr.update(value="")
    except Exception as e:
        print("Send callback error:", repr(e), flush=True)
        yield (
            format_messages_for_chatbot(state["messages"]),
            state,
            "Something went wrong. Please try again.",
            dark_mode,
            msg,
        )


# ==================== GRADIO UI ====================

NAVBAR_HTML = """
<nav class="as-nav">
  <div class="as-nav-logo">
    <div class="as-nav-logo-dot"></div>
    <span>Affect Sense AI</span>
  </div>
  <div class="as-nav-links">
    <a href="#about">About</a>
    <a href="#features">Features</a>
    <a href="#chat">Try It</a>
  </div>
  <a href="#chat" class="as-nav-cta">Start Session →</a>
</nav>
"""

HERO_HTML = """
<section class="as-hero" id="home">
  <div class="as-hero-badge">
    <div class="as-hero-badge-dot"></div>
    AI-Powered Emotional Intelligence
  </div>
  <h1>Understand Your Emotions,<br><em>One Conversation at a Time</em></h1>
  <p>Affect Sense AI listens deeply, detects how you truly feel, and responds with the empathy and care you deserve — in real time.</p>
  <div class="as-hero-btns">
    <a href="#chat" class="as-btn-primary">Start Talking →</a>
    <a href="#about" class="as-btn-ghost">Learn More</a>
  </div>
  <div class="as-hero-stats">
    <div class="as-stat">
      <span class="as-stat-num">7+</span>
      <span class="as-stat-label">Emotions Detected</span>
    </div>
    <div class="as-stat">
      <span class="as-stat-num">Real‑time</span>
      <span class="as-stat-label">Analysis</span>
    </div>
    <div class="as-stat">
      <span class="as-stat-num">24 / 7</span>
      <span class="as-stat-label">Always Available</span>
    </div>
    <div class="as-stat">
      <span class="as-stat-num">Safe</span>
      <span class="as-stat-label">Crisis Detection</span>
    </div>
  </div>
</section>
<div class="as-divider"></div>
"""

ABOUT_HTML = """
<div id="about" style="scroll-margin-top:80px">
<section class="as-section">
  <span class="as-section-label">About the Project</span>
  <h2 class="as-section-title">What is Affect Sense AI?</h2>
  <p class="as-section-sub">A research-backed mental wellness companion that combines state-of-the-art NLP with empathetic AI.</p>

  <div class="as-about-grid">
    <div class="as-about-text">
      <p>
        <strong>Affect Sense AI</strong> is an intelligent mental health companion that goes beyond simple chatbots.
        It uses a fine-tuned emotion classification model to detect nuanced emotional states — joy, sadness, anger,
        fear, surprise, love, and neutrality — directly from the words you write.
      </p>
      <p>
        Every message you send is analysed in real time. The detected emotion is then used to <strong>shape and
        personalise</strong> the AI's response, ensuring you always receive a reply that genuinely matches how you feel
        — not a generic template.
      </p>
      <p>
        The project also incorporates a dedicated <strong>crisis detection layer</strong>: if signs of suicidal ideation
        are detected, Affect Sense AI immediately pauses the conversation and surfaces verified emergency helplines,
        prioritising your safety above everything else.
      </p>
      <p>
        At the end of each session, the <strong>Analytics Summary</strong> gives you a clear picture of your emotional
        journey — dominant feelings, variability, and confidence scores — helping you build self-awareness over time.
      </p>
      <div class="as-about-pills">
        <span class="as-pill"><span class="as-pill-dot"></span>DistilRoBERTa Emotion Model</span>
        <span class="as-pill"><span class="as-pill-dot"></span>Gemini 2.5 Flash</span>
        <span class="as-pill"><span class="as-pill-dot"></span>Suicidality Classifier</span>
        <span class="as-pill"><span class="as-pill-dot"></span>Real-time NLP</span>
        <span class="as-pill"><span class="as-pill-dot"></span>Session Analytics</span>
      </div>
    </div>

    <div class="as-about-visual">
      <div class="as-visual-row">
        <div class="as-visual-icon sky">🎯</div>
        <div class="as-visual-row-text">
          <strong>Emotion Detection</strong>
          <span>7-class classification with confidence scores</span>
        </div>
      </div>
      <div class="as-visual-row">
        <div class="as-visual-icon teal">🧠</div>
        <div class="as-visual-row-text">
          <strong>Empathetic AI Responses</strong>
          <span>Gemini-powered, emotion-conditioned replies</span>
        </div>
      </div>
      <div class="as-visual-row">
        <div class="as-visual-icon rose">🚨</div>
        <div class="as-visual-row-text">
          <strong>Crisis Safety Layer</strong>
          <span>Automatic detection + emergency resources</span>
        </div>
      </div>
      <div class="as-visual-row">
        <div class="as-visual-icon gold">📊</div>
        <div class="as-visual-row-text">
          <strong>Session Analytics</strong>
          <span>Emotion trends, variability, and summaries</span>
        </div>
      </div>
    </div>
  </div>
</section>
</div>
<div class="as-divider"></div>
"""

FEATURES_HTML = """
<div id="features" style="scroll-margin-top:80px">
<section class="as-section">
  <span class="as-section-label">Features</span>
  <h2 class="as-section-title">Built for Your Wellbeing</h2>
  <p class="as-section-sub">Every feature is designed to help you feel heard, understood, and supported.</p>

  <div class="as-features-grid">
    <div class="as-feature-card">
      <div class="as-feature-icon">🎭</div>
      <h3>Real-Time Emotion Detection</h3>
      <p>Each message is classified across 7 emotional dimensions using a fine-tuned DistilRoBERTa model trained on diverse real-world conversations.</p>
    </div>
    <div class="as-feature-card">
      <div class="as-feature-icon">💬</div>
      <h3>Emotion-Adaptive Responses</h3>
      <p>Your detected emotion is passed directly to Gemini, ensuring every response is tonally appropriate and uniquely calibrated to your current state.</p>
    </div>
    <div class="as-feature-card">
      <div class="as-feature-icon">🛡️</div>
      <h3>Crisis Detection & Safety</h3>
      <p>A dedicated suicidality classifier monitors conversations. If risk is detected, the AI immediately provides verified Indian emergency helplines.</p>
    </div>
    <div class="as-feature-card">
      <div class="as-feature-icon">📈</div>
      <h3>Session Analytics</h3>
      <p>After your session, get a concise summary of your emotional journey — dominant feelings, variability score, and classifier confidence metrics.</p>
    </div>
    <div class="as-feature-card">
      <div class="as-feature-icon">⚡</div>
      <h3>Zero-Latency Thinking</h3>
      <p>Powered by Gemini 2.5 Flash with thinking budget set to zero — blazing fast responses without sacrificing empathy or relevance.</p>
    </div>
    <div class="as-feature-card">
      <div class="as-feature-icon">🔒</div>
      <h3>Input Validation & Safety</h3>
      <p>All messages pass through a robust validation pipeline before processing, protecting the system and ensuring a stable, consistent experience.</p>
    </div>
  </div>
</section>
</div>
<div class="as-divider"></div>
"""

CHAT_HEADER_HTML = """
<div id="chat" style="scroll-margin-top:80px">
<div class="as-chat-section">
  <div class="as-chat-header">
    <span class="as-section-label">Live Session</span>
    <h2 class="as-section-title" style="text-align:center">Talk to Affect Sense AI</h2>
    <p class="as-section-sub" style="margin:0 auto;text-align:center">
      Share what's on your mind. The AI will detect your emotion and respond with genuine empathy.
    </p>
  </div>

  <div class="as-chat-wrap">
    <div class="as-chat-titlebar">
      <div class="as-titlebar-dot live"></div>
      <div class="as-titlebar-dot"></div>
      <div class="as-titlebar-dot"></div>
      <span class="as-titlebar-label">Affect Sense AI — Active Session</span>
    </div>
"""

CHAT_CLOSE_HTML = """
  </div>
</div>
</div>
"""

FOOTER_HTML = """
<div class="as-divider"></div>
<footer class="as-footer">
  <span class="as-footer-logo">Affect Sense AI</span>
  <p>Built with ❤️ for mental wellness · Powered by Gemini &amp; HuggingFace Transformers</p>
  <p style="font-size:12px;color:#555">This is not a substitute for professional mental health care.</p>
</footer>
"""

with gr.Blocks(css=CUSTOM_CSS, title="Affect Sense AI") as demo:

    state = gr.State({"messages": []})
    dark_mode_state = gr.State(False)

    gr.HTML(NAVBAR_HTML)
    gr.HTML(HERO_HTML)
    gr.HTML(ABOUT_HTML)
    gr.HTML(FEATURES_HTML)

    # ── Chat section ──
    gr.HTML(CHAT_HEADER_HTML)

    chat = gr.Chatbot(height=480, elem_id="chat_card")

    with gr.Row(elem_classes=["as-chat-input-row"]):
        msg_box = gr.Textbox(
            placeholder="Share what's on your mind...",
            elem_id="msg_box",
            show_label=False,
            scale=8,
        )
        send_btn = gr.Button("Send ↑", elem_id="send_btn", scale=1)

    gr.HTML(CHAT_CLOSE_HTML)

    # ── Analytics row ──
    with gr.Row(elem_classes=["as-analytics-wrap"]):
        summary_btn = gr.Button("📊 Generate Summary", elem_id="summary_btn")

    summary_output = gr.Markdown(elem_id="summary_output")

    gr.HTML(FOOTER_HTML)

    # ── Events ──
    send_btn.click(
        send,
        inputs=[msg_box, state, dark_mode_state],
        outputs=[chat, state, summary_output, dark_mode_state, msg_box]
    )

    msg_box.submit(
        send,
        inputs=[msg_box, state, dark_mode_state],
        outputs=[chat, state, summary_output, dark_mode_state, msg_box],
    )

    summary_btn.click(
        generate_summary,
        inputs=[state],
        outputs=[summary_output]
    )


if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=True,
        height=900,
        width="100%",
        prevent_thread_lock=False,
    )
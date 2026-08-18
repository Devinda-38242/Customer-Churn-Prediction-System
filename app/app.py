import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make project root available to Python
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT MODEL PIPELINE
# ============================================================

from src.models.predict import (
    analyze_customer,
    load_model,
    load_preprocessor,
    prepare_customer_data,
    determine_risk_level,
    THRESHOLD
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PREMIUM DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 15% 10%,
                rgba(99, 102, 241, 0.08),
                transparent 28%
            ),
            radial-gradient(
                circle at 85% 15%,
                rgba(14, 165, 233, 0.06),
                transparent 25%
            ),
            #0b0f19;
    }

    .main {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* ========================================================
       TYPOGRAPHY
       ======================================================== */

    h1 {
        font-size: 2.8rem !important;
        font-weight: 750 !important;
        letter-spacing: -1.5px;
        line-height: 1.15;
    }

    h2 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.8px;
    }

    h3 {
        font-weight: 650 !important;
        letter-spacing: -0.3px;
    }

    p {
        line-height: 1.65;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #171b27 0%,
                #121620 100%
            );

        border-right: 1px solid rgba(255, 255, 255, 0.07);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.7rem !important;
        letter-spacing: -0.8px;
    }

    section[data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.08);
        margin: 1.5rem 0;
    }

    section[data-testid="stSidebar"] .stCaption {
        color: rgba(255, 255, 255, 0.55);
    }


    /* ========================================================
       NAVIGATION
       ======================================================== */

    div[role="radiogroup"] {
        gap: 0.35rem;
    }

    div[role="radiogroup"] label {
        border-radius: 10px;
        padding: 0.55rem 0.65rem;
        transition: all 0.2s ease;
    }

    div[role="radiogroup"] label:hover {
        background: rgba(255, 255, 255, 0.05);
    }


    /* ========================================================
       KPI / METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.065),
                rgba(255, 255, 255, 0.025)
            );

        border: 1px solid rgba(255, 255, 255, 0.09);

        border-radius: 18px;

        padding: 1.25rem 1.3rem;

        min-height: 135px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.22),
            inset 0 1px 0 rgba(255, 255, 255, 0.035);

        backdrop-filter: blur(12px);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-testid="stMetric"]:hover {

        transform: translateY(-3px);

        border-color: rgba(99, 102, 241, 0.35);

        box-shadow:
            0 16px 40px rgba(0, 0, 0, 0.28),
            0 0 25px rgba(99, 102, 241, 0.08);
    }

    div[data-testid="stMetricLabel"] {

        font-size: 0.85rem !important;

        font-weight: 600 !important;

        color: rgba(255, 255, 255, 0.65) !important;

        letter-spacing: 0.2px;
    }

    div[data-testid="stMetricValue"] {

        font-size: 1.85rem !important;

        font-weight: 700 !important;

        letter-spacing: -0.8px;

        white-space: nowrap;

        overflow: hidden;

        text-overflow: ellipsis;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {

        border: 1px solid rgba(255, 255, 255, 0.10);

        border-radius: 10px;

        min-height: 2.8rem;

        font-weight: 650;

        letter-spacing: 0.1px;

        background:
            linear-gradient(
                135deg,
                rgba(99, 102, 241, 0.95),
                rgba(59, 130, 246, 0.95)
            );

        color: white;

        box-shadow:
            0 8px 20px rgba(37, 99, 235, 0.20);

        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            filter 0.18s ease;
    }

    .stButton > button:hover {

        transform: translateY(-2px);

        filter: brightness(1.08);

        box-shadow:
            0 12px 28px rgba(37, 99, 235, 0.30);
    }

    .stButton > button:active {
        transform: translateY(0);
    }


    /* ========================================================
       INPUT FIELDS
       ======================================================== */

    div[data-baseweb="select"] > div {

        background: rgba(255, 255, 255, 0.035);

        border: 1px solid rgba(255, 255, 255, 0.09);

        border-radius: 10px;

        transition:
            border-color 0.2s ease,
            box-shadow 0.2s ease;
    }

    div[data-baseweb="select"] > div:hover {

        border-color: rgba(99, 102, 241, 0.40);
    }

    input {

        border-radius: 10px !important;

        background: rgba(255, 255, 255, 0.035) !important;

        border: 1px solid rgba(255, 255, 255, 0.09) !important;
    }

    input:focus {

        border-color: rgba(99, 102, 241, 0.65) !important;

        box-shadow:
            0 0 0 2px rgba(99, 102, 241, 0.12) !important;
    }


    /* ========================================================
       FILE UPLOADER
       ======================================================== */

    section[data-testid="stFileUploaderDropzone"] {

        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.045),
                rgba(255, 255, 255, 0.018)
            );

        border: 1px dashed rgba(255, 255, 255, 0.18);

        border-radius: 16px;

        padding: 1.2rem;

        transition:
            border-color 0.2s ease,
            background 0.2s ease;
    }

    section[data-testid="stFileUploaderDropzone"]:hover {

        border-color: rgba(99, 102, 241, 0.55);

        background:
            rgba(99, 102, 241, 0.035);
    }


    /* ========================================================
       ALERT / MESSAGE BOXES
       ======================================================== */

    div[data-testid="stAlert"] {

        border-radius: 12px;

        border: 1px solid rgba(255, 255, 255, 0.08);

        box-shadow:
            0 6px 18px rgba(0, 0, 0, 0.12);
    }


    /* ========================================================
       CONTAINERS / CARDS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        background:
            linear-gradient(
                145deg,
                rgba(255, 255, 255, 0.045),
                rgba(255, 255, 255, 0.018)
            );

        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 16px;

        box-shadow:
            0 10px 30px rgba(0, 0, 0, 0.18);
    }


    /* ========================================================
       DATAFRAME
       ======================================================== */

    div[data-testid="stDataFrame"] {

        border-radius: 14px;

        border: 1px solid rgba(255, 255, 255, 0.08);

        overflow: hidden;

        box-shadow:
            0 8px 25px rgba(0, 0, 0, 0.16);
    }


    /* ========================================================
       PROGRESS BAR
       ======================================================== */

    div[data-testid="stProgress"] {

        margin-top: 0.5rem;

        margin-bottom: 0.8rem;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    details {

        background:
            rgba(255, 255, 255, 0.025);

        border:
            1px solid rgba(255, 255, 255, 0.08);

        border-radius: 12px;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {

        border: none;

        border-top:
            1px solid rgba(255, 255, 255, 0.08);

        margin: 2rem 0;
    }


    /* ========================================================
       CAPTIONS
       ======================================================== */

    .stCaption {

        color:
            rgba(255, 255, 255, 0.52) !important;
    }


    /* ========================================================
       DOWNLOAD BUTTON
       ======================================================== */

    div[data-testid="stDownloadButton"] button {

        border-radius: 10px;

        font-weight: 650;

        background:
            rgba(255, 255, 255, 0.045);

        border:
            1px solid rgba(255, 255, 255, 0.12);

        transition:
            all 0.2s ease;
    }

    div[data-testid="stDownloadButton"] button:hover {

        background:
            rgba(99, 102, 241, 0.12);

        border-color:
            rgba(99, 102, 241, 0.40);
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: #0b0f19;
    }

    ::-webkit-scrollbar-thumb {

        background:
            rgba(255, 255, 255, 0.15);

        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {

        background:
            rgba(255, 255, 255, 0.25);
    }


    /* ========================================================
       MOBILE / SMALL SCREEN
       ======================================================== */

    @media (max-width: 900px) {

        h1 {
            font-size: 2.2rem !important;
        }

        h2 {
            font-size: 1.6rem !important;
        }

        div[data-testid="stMetric"] {
            min-height: 115px;
        }

    }


    /* ========================================================
       CHURN INTELLIGENCE PREMIUM SYSTEM
       ======================================================== */
    :root {
        --ci-bg: #080b12;
        --ci-panel: rgba(17, 23, 35, 0.78);
        --ci-border: rgba(148, 163, 184, 0.14);
        --ci-text: #f8fafc;
        --ci-muted: #8d99ab;
        --ci-accent: #6d7cff;
    }

    .stApp { background-color: var(--ci-bg); }
    .block-container { max-width: 1480px; padding-left: 2.2rem; padding-right: 2.2rem; }

    .brand-block {
        padding: 0.4rem 0 1.25rem 0;
        border-bottom: 1px solid rgba(255,255,255,.08);
        margin-bottom: 1rem;
    }
    .brand-kicker { font-size: .64rem; letter-spacing: .18em; color: #7d8aa0; font-weight: 700; margin-bottom: .55rem; }
    .brand-title { font-size: 1.55rem; line-height: .98; font-weight: 800; letter-spacing: -.04em; color: #fff; }
    .brand-subtitle { margin-top: .7rem; font-size: .76rem; line-height: 1.45; color: #8995a8; max-width: 190px; }
    .system-status { display:flex; align-items:center; gap:.48rem; padding:.62rem .7rem; margin:.8rem 0 1.15rem; background:rgba(16,185,129,.055); border:1px solid rgba(16,185,129,.15); border-radius:10px; font-size:.61rem; letter-spacing:.08em; color:#7f8b9d; }
    .system-status strong { margin-left:auto; color:#6ee7b7; font-size:.61rem; }
    .status-dot { width:7px; height:7px; border-radius:50%; background:#34d399; box-shadow:0 0 10px rgba(52,211,153,.7); }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0d121c 0%,#090d15 100%); }
    section[data-testid="stSidebar"] .stRadio > label { font-size:.66rem !important; text-transform:uppercase; letter-spacing:.12em; color:#657185 !important; font-weight:700; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label { border:1px solid transparent !important; padding:.68rem .75rem; margin:.16rem 0; background:transparent !important; color: #aeb8c9 !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label:hover { background:rgba(34,211,238,.08) !important; border-color:rgba(34,211,238,.18) !important; color: #fff !important; }
    section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] { background:rgba(34,211,238,.15) !important; border-color:rgba(34,211,238,.35) !important; color:#22d3ee !important; box-shadow: inset 2px 0 0 #22d3ee !important; }

    .page-kicker { font-size:.68rem; text-transform:uppercase; letter-spacing:.16em; color:#778399; font-weight:750; margin-bottom:.5rem; }
    .hero-panel { background:linear-gradient(135deg,rgba(109,124,255,.11),rgba(17,23,35,.68) 55%,rgba(17,23,35,.4)); border:1px solid var(--ci-border); border-radius:20px; padding:1.45rem 1.55rem; margin-bottom:1.3rem; box-shadow:0 18px 50px rgba(0,0,0,.18); }
    .hero-title { font-size:2.35rem; font-weight:800; letter-spacing:-.045em; margin:0; color:#f8fafc; }
    .hero-copy { color:#8e9aac; font-size:.92rem; max-width:760px; margin-top:.55rem; line-height:1.65; }

    div[data-testid="stMetric"] { background:linear-gradient(145deg,rgba(22,29,43,.88),rgba(12,17,27,.92)); border:1px solid var(--ci-border); border-radius:16px; min-height:125px; padding:1.1rem 1.15rem; }
    div[data-testid="stMetricLabel"] { color:#7f8a9d !important; text-transform:uppercase; letter-spacing:.08em; font-size:.66rem !important; font-weight:750 !important; }
    div[data-testid="stMetricValue"] { color:#f5f7fb !important; font-size:1.8rem !important; font-weight:780 !important; }

    .section-label { font-size:.68rem; text-transform:uppercase; letter-spacing:.15em; color:#69758a; font-weight:750; margin:1.4rem 0 .7rem; }
    .stButton > button { border-radius:10px; min-height:2.75rem; font-weight:700; background:linear-gradient(135deg,#6977ff,#4d62db); border:1px solid rgba(255,255,255,.1); box-shadow:0 8px 25px rgba(65,80,190,.2); }
    .stButton > button:hover { box-shadow:0 12px 32px rgba(65,80,190,.3); }
    input, div[data-baseweb="select"] > div { background:#0f1520 !important; border-color:rgba(148,163,184,.14) !important; }
    details { background:rgba(15,21,32,.72); border-color:rgba(148,163,184,.13); border-radius:12px; }
    div[data-testid="stAlert"] { background:rgba(18,24,36,.72); border-radius:12px; }
    div[data-testid="stDataFrame"] { border:1px solid rgba(148,163,184,.12); border-radius:12px; }

    @media (max-width: 900px) {
        .block-container { padding-left:1rem; padding-right:1rem; }
        .hero-title { font-size:1.75rem; }
        div[data-testid="stMetric"] { min-height:105px; }
    }
    @media (max-width: 600px) {
        .block-container { padding-top:1rem; }
        h1 { font-size:1.75rem !important; }
        h2 { font-size:1.35rem !important; }
    }

        /* ========================================================
           PHASE 09.5 — FINAL PRODUCT POLISH
           ======================================================== */

        :root {
            --ci-bg: #090d16;
            --ci-panel: rgba(18, 24, 38, 0.82);
            --ci-panel-strong: #111827;
            --ci-border: rgba(148, 163, 184, 0.14);
            --ci-border-strong: rgba(99, 102, 241, 0.38);
            --ci-text: #f8fafc;
            --ci-muted: #94a3b8;
            --ci-accent: #6366f1;
            --ci-cyan: #22d3ee;
            --ci-success: #34d399;
        }

        /* Premium app shell */
        .stApp {
            color: var(--ci-text);
            background:
                radial-gradient(circle at 8% 0%, rgba(99,102,241,.13), transparent 26%),
                radial-gradient(circle at 92% 4%, rgba(34,211,238,.08), transparent 24%),
                linear-gradient(180deg, #090d16 0%, #0b101b 55%, #090d16 100%);
        }

        .block-container {
            max-width: 1460px;
            padding-left: clamp(1rem, 3vw, 3rem);
            padding-right: clamp(1rem, 3vw, 3rem);
        }

        /* Premium sidebar identity */
        section[data-testid="stSidebar"] {
            background:
                radial-gradient(circle at 50% -10%, rgba(99,102,241,.18), transparent 30%),
                linear-gradient(180deg, #0e1421 0%, #0a0f18 100%);
            box-shadow: 14px 0 40px rgba(0,0,0,.18);
        }

        section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 1.15rem .9rem 1.5rem;
        }

        .ci-brand {
            position: relative;
            padding: 1.1rem 1rem 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(148,163,184,.13);
            border-radius: 18px;
            background: linear-gradient(145deg, rgba(99,102,241,.12), rgba(15,23,42,.62));
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 18px 40px rgba(0,0,0,.18);
            overflow: hidden;
        }

        .ci-brand::after {
            content: "";
            position: absolute;
            width: 110px;
            height: 110px;
            right: -55px;
            top: -55px;
            border-radius: 50%;
            background: rgba(34,211,238,.08);
        }

        .ci-brand-mark {
            width: 42px;
            height: 42px;
            display: grid;
            place-items: center;
            border-radius: 13px;
            margin-bottom: .8rem;
            color: #fff;
            font-size: 1.15rem;
            font-weight: 800;
            background: linear-gradient(135deg, #6366f1, #22d3ee);
            box-shadow: 0 10px 25px rgba(99,102,241,.28);
        }

        .ci-brand-title {
            font-size: 1.12rem;
            font-weight: 800;
            letter-spacing: .08em;
            color: #fff;
        }

        .ci-brand-subtitle {
            margin-top: .25rem;
            color: #94a3b8;
            font-size: .78rem;
            line-height: 1.45;
        }

        .ci-status {
            display: flex;
            align-items: center;
            gap: .5rem;
            margin-top: .95rem;
            padding: .55rem .7rem;
            border-radius: 10px;
            background: rgba(52,211,153,.06);
            border: 1px solid rgba(52,211,153,.14);
            color: #a7f3d0;
            font-size: .72rem;
            font-weight: 700;
            letter-spacing: .05em;
            text-transform: uppercase;
        }

        .ci-status-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: #34d399;
            box-shadow: 0 0 12px rgba(52,211,153,.75);
        }

        /* Disable sidebar collapse completely */
        button[data-testid="stSidebarCollapsedControl"] {
            display: none !important;
            visibility: hidden !important;
            pointer-events: none !important;
        }

        /* Prevent sidebar from collapsing */
        section[data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
            max-width: 260px !important;
        }

        /* Adjust main content to make room for sidebar */
        .main {
            margin-left: 0 !important;
            padding-left: 0 !important;
        }

        .stApp {
            display: flex !important;
            flex-direction: row !important;
            width: 100% !important;
        }

        section[data-testid="stSidebar"] {
            position: relative !important;
            flex-shrink: 0 !important;
        }

        /* Content area adjusts to sidebar */
        .stMain {
            flex: 1 !important;
            width: calc(100% - 260px) !important;
            overflow-x: auto !important;
        }

        .block-container {
            max-width: 100% !important;
            width: 100% !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }

        /* Navigation */
        section[data-testid="stSidebar"] div[role="radiogroup"] {
            gap: .45rem;
        }

        /* Override Streamlit radio button defaults */
        section[data-testid="stSidebar"] label span {
            color: #aeb8c9 !important;
        }

        section[data-testid="stSidebar"] label[data-checked="true"] span,
        section[data-testid="stSidebar"] label[aria-selected="true"] span {
            color: #22d3ee !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            min-height: 44px;
            padding: .7rem .8rem;
            border: 1px solid transparent !important;
            border-radius: 12px;
            color: #aeb8c9 !important;
            background: transparent !important;
            transition: all .18s ease;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            color: #fff !important;
            background: rgba(34,211,238,.08) !important;
            border-color: rgba(34,211,238,.18) !important;
        }

        section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"],
        section[data-testid="stSidebar"] div[role="radiogroup"] label[aria-selected="true"] {
            color: #22d3ee !important;
            background: rgba(34,211,238,.15) !important;
            border-color: rgba(34,211,238,.35) !important;
            box-shadow: inset 2px 0 0 #22d3ee !important;
        }

        /* Executive hero */
        .ci-hero {
            position: relative;
            overflow: hidden;
            margin: .2rem 0 1.5rem;
            padding: clamp(1.5rem, 3vw, 2.5rem);
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 24px;
            background:
                radial-gradient(circle at 85% 20%, rgba(34,211,238,.12), transparent 25%),
                radial-gradient(circle at 70% 80%, rgba(99,102,241,.14), transparent 30%),
                linear-gradient(135deg, rgba(20,28,45,.95), rgba(11,16,27,.92));
            box-shadow: 0 25px 70px rgba(0,0,0,.28), inset 0 1px 0 rgba(255,255,255,.045);
        }

        .ci-hero::before {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -90px;
            top: -120px;
            border: 1px solid rgba(99,102,241,.18);
            border-radius: 50%;
        }

        .ci-eyebrow {
            color: #818cf8;
            font-size: .72rem;
            font-weight: 800;
            letter-spacing: .14em;
            text-transform: uppercase;
            margin-bottom: .65rem;
        }

        .ci-hero-title {
            position: relative;
            margin: 0;
            color: #f8fafc;
            font-size: clamp(1.8rem, 4vw, 3.2rem);
            line-height: 1.12;
            letter-spacing: -.055em;
            font-weight: 850;
        }

        .ci-hero-copy {
            position: relative;
            max-width: 760px;
            margin-top: .9rem;
            color: #aab5c7;
            font-size: 1rem;
            line-height: 1.7;
        }

        .ci-pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin-top: 1.25rem;
        }

        .ci-pill {
            padding: .45rem .7rem;
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 999px;
            background: rgba(255,255,255,.035);
            color: #cbd5e1;
            font-size: .72rem;
            font-weight: 650;
        }

        /* Section headers */
        .ci-section {
            margin: 1.8rem 0 .85rem;
            padding-bottom: .65rem;
            border-bottom: 1px solid rgba(148,163,184,.10);
        }

        .ci-section-kicker {
            color: #64748b;
            font-size: .68rem;
            font-weight: 800;
            letter-spacing: .12em;
            text-transform: uppercase;
        }

        .ci-section-title {
            margin-top: .18rem;
            color: #f1f5f9;
            font-size: 1.28rem;
            font-weight: 760;
        }

        .ci-card {
            height: 100%;
            padding: 1.15rem;
            border: 1px solid rgba(148,163,184,.12);
            border-radius: 16px;
            background: linear-gradient(145deg, rgba(255,255,255,.045), rgba(255,255,255,.018));
            box-shadow: 0 12px 32px rgba(0,0,0,.15);
        }

        .ci-card-number {
            color: #818cf8;
            font-size: .7rem;
            font-weight: 800;
            letter-spacing: .1em;
        }

        .ci-card-title {
            margin-top: .55rem;
            color: #f8fafc;
            font-weight: 730;
            font-size: 1rem;
        }

        .ci-card-copy {
            margin-top: .4rem;
            color: #8fa0b5;
            font-size: .82rem;
            line-height: 1.55;
        }

        /* Inputs */
        div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input {
            min-height: 44px;
        }

        /* Expander polish */
        details {
            background: rgba(255,255,255,.025) !important;
            border: 1px solid rgba(148,163,184,.11) !important;
            border-radius: 14px !important;
            overflow: hidden;
        }

        details summary {
            padding: .8rem 1rem !important;
            font-weight: 650 !important;
        }

        /* Data tables */
        div[data-testid="stDataFrame"] {
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(148,163,184,.12);
            background: rgba(15,23,42,.42);
        }

        /* Responsive */
        @media (max-width: 1024px) {
            section[data-testid="stSidebar"] {
                width: 240px !important;
                min-width: 240px !important;
                max-width: 240px !important;
            }
            .stMain {
                width: calc(100% - 240px) !important;
            }
        }

        @media (max-width: 900px) {
            .block-container { 
                padding-top: 1.2rem;
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
            }
            .ci-hero { border-radius: 18px; padding: 1.35rem; }
            .ci-hero-title { font-size: 2rem; }
            .ci-pill-row { gap: .4rem; }
            section[data-testid="stSidebar"] {
                width: 200px !important;
                min-width: 200px !important;
                max-width: 200px !important;
            }
            .stMain {
                width: calc(100% - 200px) !important;
            }
        }

        @media (max-width: 600px) {
            .ci-hero-copy { font-size: .9rem; }
            .ci-card { padding: 1rem; }
            section[data-testid="stSidebar"] {
                width: 160px !important;
                min-width: 160px !important;
                max-width: 160px !important;
            }
            .stMain {
                width: calc(100% - 160px) !important;
            }
            .block-container {
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "prediction_result" not in st.session_state:
    st.session_state["prediction_result"] = None

if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="ci-brand">
            <div class="ci-brand-mark">CI</div>
            <div class="ci-brand-title">CHURN INTELLIGENCE</div>
            <div class="ci-brand-subtitle">Customer retention analytics and predictive risk intelligence.</div>
            <div class="ci-status"><span class="ci-status-dot"></span> Model system online</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.caption("WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Single Customer Analysis",
            "Portfolio Analysis",
            "System Information"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption("DECISION ENGINE")
    st.markdown(
        "**Gradient Boosting**  ")
    st.caption("SHAP explainability • 30% decision threshold")

    st.markdown("---")
    st.caption("Churn Intelligence Platform • v09.5")


# ============================================================
# HOME / DASHBOARD
# ============================================================

if page == "Dashboard":

    st.markdown(
        """
        <div class="ci-hero">
            <div class="ci-eyebrow">Predictive retention intelligence</div>
            <div class="ci-hero-title">CHURN INTELLIGENCE<br>AI-Powered Customer Churn Prediction & Business Intelligence for Telecommunications</div>
            <div class="ci-hero-copy">
                Transform customer behavior into clear churn risk signals, explainable drivers,
                and prioritized retention decisions through one comprehensive executive analytics platform.
            </div>
            <div class="ci-pill-row">
                <span class="ci-pill">Predictive Risk Scoring</span>
                <span class="ci-pill">SHAP Explainability</span>
                <span class="ci-pill">Portfolio Analytics</span>
                <span class="ci-pill">Retention Prioritization</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="ci-section"><div class="ci-section-kicker">System overview</div><div class="ci-section-title">Decision engine at a glance</div></div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Engine", "Gradient Boosting")
    with col2:
        st.metric("Decision Threshold", f"{THRESHOLD:.0%}")
    with col3:
        st.metric("Explainability", "SHAP")
    with col4:
        st.metric("Decision Output", "Risk Classification")

    st.markdown(
        '<div class="ci-section"><div class="ci-section-kicker">Core capabilities</div><div class="ci-section-title">From prediction to action</div></div>',
        unsafe_allow_html=True
    )

    capabilities = [
        ("01", "Customer Risk Assessment", "Evaluate an individual customer profile and quantify predicted churn probability."),
        ("02", "Explainable Risk Drivers", "Surface the customer-specific factors contributing positively or negatively to risk."),
        ("03", "Portfolio Intelligence", "Analyze batch predictions, risk distribution, customer segments, and value exposure."),
        ("04", "Retention Prioritization", "Focus intervention on high-risk customers using probability and business-value signals."),
    ]
    cols = st.columns(4)
    for col, (num, title, copy) in zip(cols, capabilities):
        with col:
            st.markdown(
                f'<div class="ci-card"><div class="ci-card-number">{num}</div><div class="ci-card-title">{title}</div><div class="ci-card-copy">{copy}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="ci-section"><div class="ci-section-kicker">Decision workflow</div><div class="ci-section-title">How the intelligence layer operates</div></div>',
        unsafe_allow_html=True
    )

    workflow = [
        ("01", "Customer Data", "Capture profile, services, tenure, billing, and contract attributes."),
        ("02", "Model Inference", "Generate a churn probability using the trained Gradient Boosting pipeline."),
        ("03", "Risk & Explainability", "Classify risk and identify the strongest SHAP-supported drivers."),
        ("04", "Retention Action", "Translate risk signals into focused customer retention priorities."),
    ]
    cols = st.columns(4)
    for col, (num, title, copy) in zip(cols, workflow):
        with col:
            st.markdown(
                f'<div class="ci-card"><div class="ci-card-number">{num}</div><div class="ci-card-title">{title}</div><div class="ci-card-copy">{copy}</div></div>',
                unsafe_allow_html=True
            )

    st.info("Select a workspace from the sidebar to begin an assessment or analyze a customer portfolio.")


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

elif page == "Single Customer Analysis":

    st.title(
        "Single Customer Analysis"
    )

    st.markdown(
        """
        Enter the customer's information below to calculate
        their churn probability, risk level, and model
        explanation.
        """
    )

    st.markdown("---")

    # ========================================================
    # CUSTOMER INFORMATION
    # ========================================================

    st.subheader(
        "Customer Profile"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        gender = st.selectbox(
            "Gender",
            [
                "Male",
                "Female"
            ]
        )

    with col2:

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [
                "No",
                "Yes"
            ]
        )

    with col3:

        partner = st.selectbox(
            "Partner",
            [
                "No",
                "Yes"
            ]
        )

    with col4:

        dependents = st.selectbox(
            "Dependents",
            [
                "No",
                "Yes"
            ]
        )

    # ========================================================
    # TENURE
    # ========================================================

    st.subheader(
        "Customer Tenure"
    )

    tenure_months = st.number_input(
        "Tenure Months",
        min_value=0,
        max_value=100,
        value=12,
        step=1
    )

    # ========================================================
    # PHONE SERVICES
    # ========================================================

    st.subheader(
        "Telephony Services"
    )

    col1, col2 = st.columns(2)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            [
                "Yes",
                "No"
            ]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "No",
                "Yes",
                "No phone service"
            ]
        )

    # ========================================================
    # INTERNET SERVICES
    # ========================================================

    st.subheader(
        "Internet Services"
    )

    col1, col2 = st.columns(2)

    with col1:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

    with col2:

        online_security = st.selectbox(
            "Online Security",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        online_backup = st.selectbox(
            "Online Backup",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col2:

        device_protection = st.selectbox(
            "Device Protection",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        tech_support = st.selectbox(
            "Tech Support",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    with col2:

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    col1, col2 = st.columns(2)

    with col1:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "No",
                "Yes",
                "No internet service"
            ]
        )

    # ========================================================
    # CONTRACT AND BILLING
    # ========================================================

    st.subheader(
        "Contract & Billing"
    )

    contract = st.selectbox(
        "Contract",
        [
            "Month-to-month",
            "One year",
            "Two year"
        ]
    )

    col1, col2 = st.columns(2)

    with col1:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            [
                "Yes",
                "No"
            ]
        )

    with col2:

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    # ========================================================
    # CHARGES
    # ========================================================

    st.subheader(
        "Customer Charges"
    )

    col1, col2 = st.columns(2)

    with col1:

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=0.50
        )

    with col2:

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )

    st.markdown("---")

    # ========================================================
    # PREDICTION BUTTON
    # ========================================================

    predict_button = st.button(
        "Run Churn Assessment",
        type="primary",
        use_container_width=True
    )

    # ========================================================
    # RUN PREDICTION
    # ========================================================

    if predict_button:

        customer = {
            "Gender": gender,
            "Senior Citizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "Tenure Months": tenure_months,
            "Phone Service": phone_service,
            "Multiple Lines": multiple_lines,
            "Internet Service": internet_service,
            "Online Security": online_security,
            "Online Backup": online_backup,
            "Device Protection": device_protection,
            "Tech Support": tech_support,
            "Streaming TV": streaming_tv,
            "Streaming Movies": streaming_movies,
            "Contract": contract,
            "Paperless Billing": paperless_billing,
            "Payment Method": payment_method,
            "Monthly Charges": monthly_charges,
            "Total Charges": total_charges
        }

        with st.spinner(
            "Analyzing customer profile..."
        ):

            try:

                result = analyze_customer(
                    customer
                )

                st.session_state[
                    "prediction_result"
                ] = result

                st.success(
                    "Customer analysis completed successfully."
                )

            except Exception as error:

                st.session_state[
                    "prediction_result"
                ] = None

                st.error(
                    f"Prediction failed: {error}"
                )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if st.session_state["prediction_result"] is not None:

        result = st.session_state[
            "prediction_result"
        ]

        st.markdown("---")

        # ====================================================
        # CUSTOMER RISK ANALYSIS
        # ====================================================

        st.subheader(
            "Churn Risk Assessment"
        )

        churn_probability = float(
            result["churn_probability"]
        )

        prediction_label = result[
            "prediction_label"
        ]

        risk_level = result[
            "risk_level"
        ]

        threshold = float(
            result["threshold"]
        )

        # ----------------------------------------------------
        # Main Metrics
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                label="Churn Probability",
                value=f"{churn_probability:.2%}"
            )

        with col2:

            if risk_level == "HIGH":

                st.error(
                    "HIGH RISK"
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    "MEDIUM RISK"
                )

            else:

                st.success(
                    "LOW RISK"
                )

        with col3:

            st.metric(
                label="Prediction",
                value=prediction_label
            )

        # ----------------------------------------------------
        # Probability Bar
        # ----------------------------------------------------

        st.markdown(
            "### Churn Probability"
        )

        st.progress(
            min(
                max(
                    churn_probability,
                    0.0
                ),
                1.0
            )
        )

        st.caption(
            f"Decision threshold: {threshold:.0%} "
            "• Probability above this threshold "
            "is classified as churn."
        )

        # ----------------------------------------------------
        # Prediction Message
        # ----------------------------------------------------

        if prediction_label == "Churn":

            st.error(
                "Customer is classified as churn risk."
            )

        else:

            st.success(
                "Customer is classified as retained."
            )


        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Churn Risk Drivers"
        )

        st.caption(
            "SHAP values show how individual customer "
            "features influence the model's churn prediction."
        )

        positive_factors = result.get(
            "positive_factors",
            pd.DataFrame()
        )

        negative_factors = result.get(
            "negative_factors",
            pd.DataFrame()
        )


        # ====================================================
        # SHAP FACTOR COLUMNS
        # ====================================================

        col1, col2 = st.columns(2)


        # ----------------------------------------------------
        # Factors Increasing Churn
        # ----------------------------------------------------

        with col1:

            st.markdown(
                "### Factors Increasing Churn"
            )

            if positive_factors.empty:

                st.info(
                    "No significant factors increasing "
                    "churn were identified."
                )

            else:

                for _, row in positive_factors.iterrows():

                    feature = str(
                        row["Feature"]
                    )

                    shap_value = float(
                        row["SHAP Value"]
                    )

                    # Make feature names easier to read
                    display_feature = feature

                    if display_feature.startswith(
                        "num__"
                    ):

                        display_feature = (
                            display_feature
                            .replace(
                                "num__",
                                ""
                            )
                        )

                    elif display_feature.startswith(
                        "cat__"
                    ):

                        display_feature = (
                            display_feature
                            .replace(
                                "cat__",
                                ""
                            )
                        )

                    st.markdown(
                        f"**{display_feature}**"
                    )

                    st.caption(
                        f"SHAP Impact: "
                        f"+{shap_value:.4f}"
                    )

                    st.progress(
                        min(
                            abs(shap_value) / 0.7,
                            1.0
                        )
                    )


        # ----------------------------------------------------
        # Factors Reducing Churn
        # ----------------------------------------------------

        with col2:

            st.markdown(
                "### Factors Reducing Churn"
            )

            if negative_factors.empty:

                st.info(
                    "No significant factors reducing "
                    "churn were identified."
                )

            else:

                for _, row in negative_factors.iterrows():

                    feature = str(
                        row["Feature"]
                    )

                    shap_value = float(
                        row["SHAP Value"]
                    )

                    # Make feature names easier to read
                    display_feature = feature

                    if display_feature.startswith(
                        "num__"
                    ):

                        display_feature = (
                            display_feature
                            .replace(
                                "num__",
                                ""
                            )
                        )

                    elif display_feature.startswith(
                        "cat__"
                    ):

                        display_feature = (
                            display_feature
                            .replace(
                                "cat__",
                                ""
                            )
                        )

                    st.markdown(
                        f"**{display_feature}**"
                    )

                    st.caption(
                        f"SHAP Impact: "
                        f"{shap_value:.4f}"
                    )

                    st.progress(
                        min(
                            abs(shap_value) / 0.7,
                            1.0
                        )
                    )


        # ====================================================
        # SHAP SUMMARY TABLE
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Explainability Impact Summary"
        )

        shap_explanation = result.get(
            "shap_explanation",
            pd.DataFrame()
        )

        if (
            isinstance(
                shap_explanation,
                pd.DataFrame
            )
            and not shap_explanation.empty
        ):

            shap_display = (
                shap_explanation[
                    [
                        "Feature",
                        "Feature Value",
                        "SHAP Value"
                    ]
                ]
                .copy()
            )

            shap_display["Feature"] = (
                shap_display["Feature"]
                .str.replace(
                    "num__",
                    "",
                    regex=False
                )
                .str.replace(
                    "cat__",
                    "",
                    regex=False
                )
            )

            shap_display = (
                shap_display
                .sort_values(
                    "SHAP Value",
                    key=lambda x: x.abs(),
                    ascending=False
                )
                .head(10)
            )

            st.dataframe(
                shap_display,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "SHAP explanation data is not available."
            )


                # ====================================================
        # PERSONALIZED RETENTION RECOMMENDATIONS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Retention Recommendations"
        )

        st.caption(
            "These recommendations are generated from the "
            "customer's profile and SHAP-confirmed factors "
            "contributing to churn risk."
        )

        recommendations = result.get(
            "recommendations",
            []
        )

        recommendation_count = result.get(
            "recommendation_count",
            0
        )

        # ----------------------------------------------------
        # No Recommendations
        # ----------------------------------------------------

        if recommendation_count == 0:

            if risk_level == "LOW":

                st.success(
                    "🟢 No immediate retention action is "
                    "required for this customer."
                )

            else:

                st.info(
                    "No SHAP-confirmed retention "
                    "recommendations were generated."
                )

        # ----------------------------------------------------
        # Recommendations Available
        # ----------------------------------------------------

        else:

            st.markdown(
                f"### {recommendation_count} "
                f"Priority Retention Action"
                f"{'s' if recommendation_count != 1 else ''}"
            )

            for recommendation in recommendations:

                rank = recommendation.get(
                    "rank",
                    0
                )

                trigger = recommendation.get(
                    "trigger",
                    "Retention Opportunity"
                )

                action_priority = recommendation.get(
                    "action_priority",
                    "LOW"
                )

                model_impact = recommendation.get(
                    "model_impact",
                    "LOW"
                )

                shap_feature = recommendation.get(
                    "shap_feature",
                    "N/A"
                )

                shap_impact = float(
                    recommendation.get(
                        "shap_impact",
                        0
                    )
                )

                reason = recommendation.get(
                    "reason",
                    ""
                )

                action = recommendation.get(
                    "recommendation",
                    ""
                )

                # ------------------------------------------------
                # Recommendation Card
                # ------------------------------------------------

                with st.container(border=True):

                    st.markdown(
                        f"### {rank}. {trigger}"
                    )

                    col1, col2, col3 = st.columns(3)

                    # --------------------------------------------
                    # Action Priority
                    # --------------------------------------------

                    with col1:

                        if action_priority == "HIGH":

                            st.error(
                                "HIGH PRIORITY"
                            )

                        elif action_priority == "MEDIUM":

                            st.warning(
                                "MEDIUM PRIORITY"
                            )

                        else:

                            st.info(
                                "LOW PRIORITY"
                            )

                    # --------------------------------------------
                    # Model Impact
                    # --------------------------------------------

                    with col2:

                        st.metric(
                            "Model Impact",
                            model_impact
                        )

                    # --------------------------------------------
                    # SHAP Impact
                    # --------------------------------------------

                    with col3:

                        st.metric(
                            "SHAP Impact",
                            f"+{shap_impact:.4f}"
                        )

                    st.markdown(
                        "**Why this matters**"
                    )

                    st.write(
                        reason
                    )

                    st.markdown(
                        "**Recommended Action**"
                    )

                    st.success(
                        action
                    )

                    st.caption(
                        f"SHAP-confirmed factor: "
                        f"{shap_feature}"
                    )


        # ====================================================
        # MODEL INTERPRETATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Interpreting Model Contributions"
        )

        col1, col2 = st.columns(2)

        with col1:

            st.info(
                """
                **Positive SHAP values**

                Push the model toward a higher
                probability of customer churn.
                """
            )

        with col2:

            st.success(
                """
                **Negative SHAP values**

                Push the model toward a lower
                probability of customer churn.
                """
            )



# ============================================================
# BATCH PREDICTION
# ============================================================

elif page == "Portfolio Analysis":

    st.title("Portfolio Churn Analysis")

    st.markdown(
        """
        Analyze an entire customer base at once. Upload customer
        data to identify churn probability, segment customers by
        risk, and prioritize the customers who need attention first.
        """
    )

    st.markdown("---")

    # ========================================================
    # TELCO DATASET REQUIREMENT
    # ========================================================

    st.markdown(
        """
        <div style="background: linear-gradient(135deg, rgba(34,211,238,.12), rgba(34,211,238,.06)); border: 1px solid rgba(34,211,238,.35); border-radius: 16px; padding: 1.2rem 1.5rem; margin-bottom: 1.5rem;">
            <div style="color: #22d3ee; font-weight: 750; font-size: 0.85rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.5rem;">📊 Telco Dataset Required</div>
            <div style="color: #f8fafc; font-size: 1rem; font-weight: 600; margin-bottom: 0.6rem;">Upload a Telecommunications Customer Dataset</div>
            <div style="color: #aab5c7; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0.8rem;">
                This system requires a telecommunications customer dataset containing customer profiles, service details, billing information, and contract attributes. The prediction model is trained specifically on Telco customer data to accurately assess churn risk within the telecommunications industry.
            </div>
            <div style="color: #cbd5e1; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.5rem;">Required Data Elements:</div>
            <div style="color: #94a3b8; font-size: 0.9rem; line-height: 1.5;">
                • <strong>Customer Information:</strong> Demographics, tenure, contract type<br>
                • <strong>Service Details:</strong> Phone, internet, and add-on services<br>
                • <strong>Billing & Payment:</strong> Monthly charges, total charges, payment method<br>
                • <strong>Contract Terms:</strong> Month-to-month, annual, or multi-year agreements
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # REQUIRED COLUMNS
    # ========================================================

    st.subheader("Required Data Schema")

    required_columns = [
        "Gender",
        "Senior Citizen",
        "Partner",
        "Dependents",
        "Tenure Months",
        "Phone Service",
        "Multiple Lines",
        "Internet Service",
        "Online Security",
        "Online Backup",
        "Device Protection",
        "Tech Support",
        "Streaming TV",
        "Streaming Movies",
        "Contract",
        "Paperless Billing",
        "Payment Method",
        "Monthly Charges",
        "Total Charges"
    ]

    st.caption(
        "Upload a CSV containing these customer attributes. "
        "Additional columns such as Customer ID are preserved."
    )

    with st.expander("View required columns", expanded=False):

        st.dataframe(
            pd.DataFrame({
                "Required Column": required_columns
            }),
            use_container_width=True,
            hide_index=True
        )

    # ========================================================
    # FILE UPLOAD
    # ========================================================

    st.markdown("---")

    st.subheader("Upload Customer Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Upload a CSV containing customer information."
    )

    if uploaded_file is not None:

        try:

            batch_df = pd.read_csv(uploaded_file)

            st.success(
                f" Dataset loaded successfully — "
                f"{len(batch_df):,} customers found."
            )

            # =================================================
            # DATASET OVERVIEW
            # =================================================

            preview_col1, preview_col2, preview_col3 = st.columns(3)

            with preview_col1:
                st.metric(
                    "Rows",
                    f"{len(batch_df):,}"
                )

            with preview_col2:
                st.metric(
                    "Columns",
                    f"{len(batch_df.columns):,}"
                )

            with preview_col3:
                st.metric(
                    "File Size",
                    f"{uploaded_file.size / 1024:.1f} KB"
                )

            with st.expander("Preview Dataset", expanded=False):

                st.dataframe(
                    batch_df.head(10),
                    use_container_width=True,
                    hide_index=True
                )

            # =================================================
            # VALIDATE COLUMNS
            # =================================================

            missing_columns = [
                column
                for column in required_columns
                if column not in batch_df.columns
            ]

            if missing_columns:

                st.error(
                    "Uploaded file is missing required columns."
                )

                st.markdown("**Missing columns:**")

                for column in missing_columns:
                    st.write(f"- `{column}`")

                st.info(
                    "Please correct the CSV and upload it again."
                )

            else:

                st.success(
                    " All required customer columns are present."
                )

                st.markdown("---")

                # =================================================
                # RUN BATCH PREDICTION
                # =================================================

                run_batch = st.button(
                    "Run Portfolio Assessment",
                    type="primary",
                    use_container_width=True
                )

                if run_batch:

                    try:

                        with st.spinner(
                            " Processing customer base and generating predictions..."
                        ):

                            # -------------------------------------
                            # Create working copy
                            # -------------------------------------

                            working_df = batch_df.copy()

                            # -------------------------------------
                            # Handle Total Charges
                            # -------------------------------------

                            working_df["Total Charges"] = pd.to_numeric(
                                working_df["Total Charges"],
                                errors="coerce"
                            )

                            # -------------------------------------
                            # Identify invalid rows
                            # -------------------------------------

                            invalid_rows = working_df[
                                working_df["Total Charges"].isnull()
                            ].copy()

                            # -------------------------------------
                            # Keep only valid customers
                            # -------------------------------------

                            valid_df = working_df[
                                working_df["Total Charges"].notnull()
                            ].copy()

                            if valid_df.empty:

                                raise ValueError(
                                    "No valid customers remain after "
                                    "checking Total Charges."
                                )

                            # -------------------------------------
                            # Load model and preprocessor
                            # -------------------------------------

                            model = load_model()

                            preprocessor = load_preprocessor()

                            # -------------------------------------
                            # Prepare customer data
                            # -------------------------------------

                            prepared_df = prepare_customer_data(
                                valid_df
                            )

                            # -------------------------------------
                            # Preprocess
                            # -------------------------------------

                            processed_df = preprocessor.transform(
                                prepared_df
                            )

                            # -------------------------------------
                            # Predict churn probability
                            # -------------------------------------

                            probabilities = model.predict_proba(
                                processed_df
                            )[:, 1]

                            # -------------------------------------
                            # Apply decision threshold
                            # -------------------------------------

                            predictions = (
                                probabilities >= THRESHOLD
                            ).astype(int)

                            # -------------------------------------
                            # Determine risk levels
                            # -------------------------------------

                            risk_levels = [
                                determine_risk_level(
                                    probability
                                )
                                for probability in probabilities
                            ]

                            # -------------------------------------
                            # Create results
                            # -------------------------------------

                            results_df = valid_df.copy()

                            results_df[
                                "Churn Probability"
                            ] = probabilities.round(4)

                            results_df[
                                "Prediction"
                            ] = predictions

                            results_df[
                                "Prediction Label"
                            ] = [
                                "Churn"
                                if prediction == 1
                                else "No Churn"
                                for prediction in predictions
                            ]

                            results_df[
                                "Risk Level"
                            ] = risk_levels

                            results_df[
                                "Decision Threshold"
                            ] = THRESHOLD

                            # -------------------------------------
                            # Create business priority
                            # -------------------------------------

                            results_df[
                                "Priority Score"
                            ] = (
                                results_df["Churn Probability"]
                                * 100
                            ).round(1)

                            # -------------------------------------
                            # Store results
                            # -------------------------------------

                            st.session_state[
                                "batch_results"
                            ] = results_df

                        st.success(
                            " Batch analysis completed successfully!"
                        )

                        if len(invalid_rows) > 0:

                            st.warning(
                                f" {len(invalid_rows):,} customer(s) were "
                                f"excluded because Total Charges was missing "
                                f"or invalid."
                            )

                            st.caption(
                                "These rows were excluded to maintain consistency "
                                "with the preprocessing used during model training."
                            )

                    except Exception as error:

                        st.error(
                            f"Portfolio assessment failed: {error}"
                        )

        except Exception as error:

            st.error(
                f"Unable to read the uploaded CSV: {error}"
            )

    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    if st.session_state["batch_results"] is not None:

        results_df = st.session_state[
            "batch_results"
        ].copy()

        st.markdown("---")

        st.subheader("Portfolio Intelligence")

        # ====================================================
        # SUMMARY METRICS
        # ====================================================

        total_customers = len(results_df)

        churn_count = int(
            (
                results_df["Prediction"] == 1
            ).sum()
        )

        churn_rate = (
            churn_count / total_customers
            if total_customers > 0
            else 0
        )

        high_risk_count = int(
            (
                results_df["Risk Level"] == "HIGH"
            ).sum()
        )

        medium_risk_count = int(
            (
                results_df["Risk Level"] == "MEDIUM"
            ).sum()
        )

        low_risk_count = int(
            (
                results_df["Risk Level"] == "LOW"
            ).sum()
        )

        # ----------------------------------------------------
        # KPI ROW 1
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Customers",
                f"{total_customers:,}"
            )

        with col2:

            st.metric(
                "Predicted Churn",
                f"{churn_count:,}",
                f"{churn_rate:.1%} of base"
            )

        with col3:

            st.metric(
                "High Risk Customers",
                f"{high_risk_count:,}"
            )

        with col4:

            st.metric(
                "Medium Risk",
                f"{medium_risk_count:,}"
            )

        # ----------------------------------------------------
        # KPI ROW 2
        # ----------------------------------------------------

        st.markdown("")

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Low Risk",
                f"{low_risk_count:,}"
            )

        with col2:

            average_probability = float(
                results_df["Churn Probability"].mean()
            )

            st.metric(
                "Average Churn Probability",
                f"{average_probability:.1%}"
            )

        with col3:

            high_risk_rate = (
                high_risk_count / total_customers
                if total_customers > 0
                else 0
            )

            st.metric(
                "High-Risk Share",
                f"{high_risk_rate:.1%}"
            )

        with col4:

            st.metric(
                " Threshold",
                f"{THRESHOLD:.0%}"
            )

        # ====================================================
        # RISK DISTRIBUTION
        # ====================================================

        st.markdown("---")

        st.subheader("Risk Distribution")

        risk_counts = (
            results_df["Risk Level"]
            .value_counts()
            .reindex(
                ["HIGH", "MEDIUM", "LOW"],
                fill_value=0
            )
        )

        risk_percentages = (
            risk_counts / total_customers * 100
            if total_customers > 0
            else risk_counts
        )

        chart_col, insight_col = st.columns(
            [2, 1]
        )

        with chart_col:

            st.bar_chart(
                risk_counts,
                horizontal=True
            )

        with insight_col:

            st.markdown("### Risk Breakdown")

            st.markdown(
                f"""
                **High Risk**  
                {high_risk_count:,} customers •
                {risk_percentages["HIGH"]:.1f}%

                **Medium Risk**  
                {medium_risk_count:,} customers •
                {risk_percentages["MEDIUM"]:.1f}%

                **Low Risk**  
                {low_risk_count:,} customers •
                {risk_percentages["LOW"]:.1f}%
                """
            )

            st.caption(
                "High-risk customers should receive the earliest "
                "retention attention."
            )

        # ====================================================
        # CHURN PROBABILITY DISTRIBUTION
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Churn Probability Distribution"
        )

        probability_bins = [
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0
        ]

        probability_labels = [
            "0–10%",
            "10–20%",
            "20–30%",
            "30–40%",
            "40–50%",
            "50–60%",
            "60–70%",
            "70–80%",
            "80–90%",
            "90–100%"
        ]

        probability_distribution = pd.cut(
            results_df["Churn Probability"],
            bins=probability_bins,
            labels=probability_labels,
            include_lowest=True,
            right=True
        ).value_counts().reindex(
            probability_labels,
            fill_value=0
        )

        st.bar_chart(
            probability_distribution
        )

        st.caption(
            "The distribution shows how churn probability is spread "
            "across the entire uploaded customer base."
        )

        # ====================================================
        # BUSINESS INSIGHTS
        # ====================================================

        st.markdown("---")
        st.subheader("Business Insights")

        st.markdown(
            "Turn the prediction results into business-focused "
            "priorities using the uploaded customer population."
        )

        high_risk_business_df = results_df[
            results_df["Risk Level"] == "HIGH"
        ].copy()

        # Revenue exposure
        total_monthly_charges = float(results_df["Monthly Charges"].sum())
        high_risk_monthly_charges = (
            float(high_risk_business_df["Monthly Charges"].sum())
            if not high_risk_business_df.empty else 0.0
        )
        high_risk_avg_monthly = (
            float(high_risk_business_df["Monthly Charges"].mean())
            if not high_risk_business_df.empty else 0.0
        )
        revenue_share = (
            high_risk_monthly_charges / total_monthly_charges
            if total_monthly_charges > 0 else 0.0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric(
                "High-Risk Monthly Charges",
                f"${high_risk_monthly_charges:,.0f}"
            )
        with c2:
            st.metric(
                "Average High-Risk Charge",
                f"${high_risk_avg_monthly:,.2f}"
            )
        with c3:
            st.metric(
                "High-Risk Revenue Share",
                f"{revenue_share:.1%}"
            )

        # Contract analysis
        st.markdown("### Contract Risk Analysis")

        contract_summary = None
        highest_contract = None

        if "Contract" in results_df.columns:
            contract_summary = (
                results_df.groupby("Contract")
                .agg(
                    Customers=("Contract", "size"),
                    High_Risk=("Risk Level", lambda x: (x == "HIGH").sum()),
                    Predicted_Churn=("Prediction", "sum"),
                    Avg_Probability=("Churn Probability", "mean")
                )
                .reset_index()
            )

            contract_summary["High Risk %"] = (
                contract_summary["High_Risk"]
                / contract_summary["Customers"] * 100
            ).round(1)

            contract_summary["Churn %"] = (
                contract_summary["Predicted_Churn"]
                / contract_summary["Customers"] * 100
            ).round(1)

            contract_summary["Avg Probability"] = (
                contract_summary["Avg_Probability"]
                .map(lambda x: f"{x:.1%}")
            )

            st.dataframe(
                contract_summary[
                    [
                        "Contract", "Customers", "High_Risk",
                        "High Risk %", "Predicted_Churn",
                        "Churn %", "Avg Probability"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            highest_contract = contract_summary.sort_values(
                "High Risk %", ascending=False
            ).iloc[0]

            st.info(
                f"**Highest-risk contract:** "
                f"{highest_contract['Contract']} — "
                f"{highest_contract['High Risk %']:.1f}% high risk."
            )

        # Tenure analysis
        st.markdown("### Tenure Risk Analysis")

        valid_tenure = pd.DataFrame()
        highest_tenure = None

        if "Tenure Months" in results_df.columns:
            tenure_analysis = results_df.copy()

            tenure_analysis["Tenure Group"] = pd.cut(
                tenure_analysis["Tenure Months"],
                bins=[-1, 6, 12, 24, 48, float("inf")],
                labels=[
                    "0–6 months", "7–12 months", "13–24 months",
                    "25–48 months", "49+ months"
                ]
            )

            tenure_summary = (
                tenure_analysis.groupby(
                    "Tenure Group", observed=False
                )
                .agg(
                    Customers=("Tenure Months", "size"),
                    High_Risk=("Risk Level", lambda x: (x == "HIGH").sum()),
                    Predicted_Churn=("Prediction", "sum"),
                    Avg_Probability=("Churn Probability", "mean")
                )
                .reset_index()
            )

            tenure_summary["High Risk %"] = (
                tenure_summary["High_Risk"]
                / tenure_summary["Customers"] * 100
            ).round(1)

            tenure_summary["Churn %"] = (
                tenure_summary["Predicted_Churn"]
                / tenure_summary["Customers"] * 100
            ).round(1)

            tenure_summary["Avg Probability"] = (
                tenure_summary["Avg_Probability"]
                .map(lambda x: f"{x:.1%}")
            )

            st.dataframe(
                tenure_summary[
                    [
                        "Tenure Group", "Customers", "High_Risk",
                        "High Risk %", "Predicted_Churn",
                        "Churn %", "Avg Probability"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            valid_tenure = tenure_summary[
                tenure_summary["Customers"] > 0
            ]

            if not valid_tenure.empty:
                highest_tenure = valid_tenure.sort_values(
                    "High Risk %", ascending=False
                ).iloc[0]

                st.info(
                    f"**Highest-risk tenure group:** "
                    f"{highest_tenure['Tenure Group']} — "
                    f"{highest_tenure['High Risk %']:.1f}% high risk."
                )

        # Payment analysis
        st.markdown("### Payment Method Risk Analysis")

        if "Payment Method" in results_df.columns:
            payment_summary = (
                results_df.groupby("Payment Method")
                .agg(
                    Customers=("Payment Method", "size"),
                    High_Risk=("Risk Level", lambda x: (x == "HIGH").sum()),
                    Predicted_Churn=("Prediction", "sum"),
                    Avg_Probability=("Churn Probability", "mean")
                )
                .reset_index()
            )

            payment_summary["High Risk %"] = (
                payment_summary["High_Risk"]
                / payment_summary["Customers"] * 100
            ).round(1)

            payment_summary["Churn %"] = (
                payment_summary["Predicted_Churn"]
                / payment_summary["Customers"] * 100
            ).round(1)

            payment_summary["Avg Probability"] = (
                payment_summary["Avg_Probability"]
                .map(lambda x: f"{x:.1%}")
            )

            st.dataframe(
                payment_summary[
                    [
                        "Payment Method", "Customers", "High_Risk",
                        "High Risk %", "Predicted_Churn",
                        "Churn %", "Avg Probability"
                    ]
                ].sort_values("High Risk %", ascending=False),
                use_container_width=True,
                hide_index=True
            )

        # Retention priority
        st.markdown("### Retention Priority Opportunities")

        priority_df = results_df.copy()
        priority_df["Retention Priority Score"] = (
            priority_df["Churn Probability"]
            * priority_df["Monthly Charges"]
        )

        priority_segments = priority_df[
            priority_df["Risk Level"] == "HIGH"
        ].sort_values(
            "Retention Priority Score", ascending=False
        ).head(10)

        if not priority_segments.empty:
            priority_columns = [
                c for c in [
                    "Customer ID", "CustomerID", "customerID",
                    "Churn Probability", "Risk Level", "Contract",
                    "Tenure Months", "Monthly Charges",
                    "Retention Priority Score"
                ]
                if c in priority_segments.columns
            ]

            priority_display = priority_segments[priority_columns].copy()

            if "Churn Probability" in priority_display:
                priority_display["Churn Probability"] = (
                    priority_display["Churn Probability"]
                    .map(lambda x: f"{x:.1%}")
                )

            if "Monthly Charges" in priority_display:
                priority_display["Monthly Charges"] = (
                    priority_display["Monthly Charges"]
                    .map(lambda x: f"${x:,.2f}")
                )

            if "Retention Priority Score" in priority_display:
                priority_display["Retention Priority Score"] = (
                    priority_display["Retention Priority Score"]
                    .map(lambda x: f"{x:,.2f}")
                )

            st.dataframe(
                priority_display,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Priority Score = churn probability × monthly charges. "
                "This is a retention prioritization heuristic, not a "
                "forecast of actual financial loss."
            )
        else:
            st.success("No high-risk customers currently require priority action.")

        # Automated recommendations
        st.markdown("### Recommended Business Focus")

        recommendations = []

        if high_risk_count > 0:
            recommendations.append(
                f"Prioritize **{high_risk_count:,} high-risk customers** "
                "for proactive retention outreach."
            )

        if high_risk_monthly_charges > 0:
            recommendations.append(
                f"High-risk customers represent approximately "
                f"**${high_risk_monthly_charges:,.0f} in combined monthly "
                "charges**, making them an important value-protection segment."
            )

        if highest_contract is not None:
            recommendations.append(
                f"Review **{highest_contract['Contract']}** customers because "
                f"this contract group has the highest high-risk rate "
                f"(**{highest_contract['High Risk %']:.1f}%**)."
            )

        if highest_tenure is not None:
            recommendations.append(
                f"Investigate the **{highest_tenure['Tenure Group']}** segment "
                "because it has the highest high-risk rate among tenure groups."
            )

        for i, recommendation in enumerate(recommendations, 1):
            st.markdown(f"** Priority {i}** — {recommendation}")

        st.caption(
            "These insights are descriptive analytics derived from the "
            "uploaded prediction results. Combine them with operational "
            "and financial context before taking action."
        )

        # ====================================================
        # HIGH-RISK PRIORITIZATION
        # ====================================================

        st.markdown("---")

        st.subheader(
            " High-Risk Customers"
        )

        st.markdown(
            """
            These customers have been identified as the highest-priority
            segment based on predicted churn probability. Use this view
            to focus retention efforts where they are most needed.
            """
        )

        high_risk_df = (
            results_df[
                results_df["Risk Level"] == "HIGH"
            ]
            .copy()
            .sort_values(
                "Churn Probability",
                ascending=False
            )
        )

        if high_risk_df.empty:

            st.success(
                "No customers are currently classified as high risk."
            )

        else:

            high_risk_display = high_risk_df.copy()

            # -----------------------------------------------
            # Identify a customer identifier when available
            # -----------------------------------------------

            identifier_candidates = [
                "Customer ID",
                "CustomerID",
                "customerID"
            ]

            identifier_column = next(
                (
                    column
                    for column in identifier_candidates
                    if column in high_risk_display.columns
                ),
                None
            )

            if identifier_column is None:

                high_risk_display.insert(
                    0,
                    "Customer",
                    high_risk_display.index.astype(str)
                )

                identifier_column = "Customer"

            # -----------------------------------------------
            # Create focused business view
            # -----------------------------------------------

            high_risk_columns = [
                identifier_column,
                "Churn Probability",
                "Prediction Label",
                "Risk Level",
                "Contract",
                "Tenure Months",
                "Monthly Charges",
                "Payment Method"
            ]

            high_risk_columns = [
                column
                for column in high_risk_columns
                if column in high_risk_display.columns
            ]

            high_risk_display = high_risk_display[
                high_risk_columns
            ].copy()

            high_risk_display[
                "Churn Probability"
            ] = (
                high_risk_display["Churn Probability"]
                .map(
                    lambda value: f"{value:.1%}"
                )
            )

            high_risk_display[
                "Monthly Charges"
            ] = (
                high_risk_display["Monthly Charges"]
                .map(
                    lambda value: f"${value:,.2f}"
                )
            )

            st.dataframe(
                high_risk_display,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                f"Showing {len(high_risk_display):,} high-risk "
                "customers, ordered by predicted churn probability."
            )

        # ====================================================
        # PRIORITIZATION FILTER
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Explore Customer Risk Segments"
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            selected_risk = st.selectbox(
                "Risk Segment",
                [
                    "All Customers",
                    "HIGH",
                    "MEDIUM",
                    "LOW"
                ]
            )

        with filter_col2:

            sort_option = st.selectbox(
                "Sort Customers By",
                [
                    "Churn Probability",
                    "Monthly Charges",
                    "Tenure Months"
                ]
            )

        if selected_risk == "All Customers":

            filtered_df = results_df.copy()

        else:

            filtered_df = results_df[
                results_df["Risk Level"] == selected_risk
            ].copy()

        if sort_option in filtered_df.columns:

            filtered_df = filtered_df.sort_values(
                sort_option,
                ascending=False
            )

        display_columns = [
            column
            for column in [
                "Customer ID",
                "CustomerID",
                "customerID",
                "Churn Probability",
                "Prediction Label",
                "Risk Level",
                "Contract",
                "Tenure Months",
                "Monthly Charges",
                "Payment Method"
            ]
            if column in filtered_df.columns
        ]

        if display_columns:

            filtered_display = filtered_df[
                display_columns
            ].copy()

        else:

            filtered_display = filtered_df.copy()

        st.dataframe(
            filtered_display,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            f"{len(filtered_df):,} customer(s) shown."
        )

        # ====================================================
        # FULL RESULTS
        # ====================================================

        st.markdown("---")

        with st.expander(
            " View complete prediction dataset",
            expanded=False
        ):

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

        # ====================================================
        # DOWNLOAD RESULTS
        # ====================================================

        st.markdown("---")

        st.subheader(
            "Export Results"
        )

        st.markdown(
            """
            Download the complete prediction dataset for further
            analysis, reporting, or use by a customer retention team.
            """
        )

        csv_data = results_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Batch Prediction Results",
            data=csv_data,
            file_name="customer_churn_batch_analytics.csv",
            mime="text/csv",
            use_container_width=True
        )

# ============================================================
# ABOUT
# ============================================================

elif page == "System Information":

    st.title(
        "System Information"
    )

    st.markdown(
        """
        ## CHURN INTELLIGENCE
        ### AI-Powered Customer Churn Prediction & Business Intelligence for Telecommunications

        This application is designed to identify customers who are at risk of leaving a telecommunications service and
        provide actionable insights to support customer retention.

        The system combines machine learning, explainable AI, and personalized retention recommendations into a
        single executive analytics dashboard built for telecom business decision-makers.
        """
    )

    st.markdown("---")

    # ========================================================
    # ABOUT THIS SYSTEM
    # ========================================================

    st.subheader(
        "About This System"
    )

    about_col1, about_col2 = st.columns(2)

    with about_col1:
        st.markdown(
            """
            **Industry**  
            Telecommunications

            **Prediction Target**  
            Customer Churn

            **Model Foundation**  
            Trained on Telco customer data with feature engineering and threshold optimization
            """
        )

    with about_col2:
        st.markdown(
            """
            **Analytics Capabilities**  
            Customer risk assessment, churn driver identification, business-value segmentation

            **Supported Analysis**  
            Individual customer predictions, batch portfolio analysis, risk distribution insights

            **Decision Support**  
            Retention prioritization, revenue-focused intervention planning
            """
        )

    st.info(
        "🎯 **System Scope**: This system analyzes telecommunications customer datasets to predict churn risk and "
        "identify retention opportunities. Predictions are powered by a Gradient Boosting model with SHAP-based explainability."
    )

    st.markdown("---")

    # ========================================================
    # PROJECT OBJECTIVE
    # ========================================================

    st.subheader(
        " Project Objective"
    )

    st.markdown(
        """
        Customer churn can have a significant impact on business
        revenue and long-term customer relationships.

        The objective of this project is to:

        - Predict the probability that a customer will churn.
        - Identify customers who require retention attention.
        - Explain the factors influencing each prediction.
        - Generate personalized retention recommendations.
        - Support data-driven customer retention strategies.
        """
    )

    # ========================================================
    # HOW IT WORKS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Decision Intelligence Workflow"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 01  |  Customer Data

            Customer information such as tenure,
            contract type, services, payment method,
            and charges is provided to the system.
            """
        )

    with col2:

        st.markdown(
            """
            ###  2. Churn Prediction

            A tuned Gradient Boosting model
            calculates the probability that the
            customer will churn.
            """
        )

    with col3:

        st.markdown(
            """
            ###  3. Explainability

            SHAP identifies the customer-specific
            factors that increase or decrease
            the predicted churn risk.
            """
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            ### 04  |  Risk Classification

            The predicted probability is converted
            into a business-friendly risk level:

            **LOW → MEDIUM → HIGH**
            """
        )

    with col2:

        st.markdown(
            """
            ###  5. Recommendations

            SHAP-confirmed risk factors are used
            to generate personalized retention
            actions.
            """
        )

    with col3:

        st.markdown(
            """
            ###  6. Analytics

            The application provides an interactive
            interface for analyzing individual and
            multiple customers.
            """
        )

    # ========================================================
    # MACHINE LEARNING
    # ========================================================

    st.markdown("---")

    st.subheader(
        " Machine Learning"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            **Model**

            Tuned Gradient Boosting Classifier

            **Decision Threshold**

            30%

            The threshold was selected to make the
            system more sensitive to potential
            churn customers.
            """
        )

    with col2:

        st.markdown(
            """
            **Explainability**

            SHAP (SHapley Additive exPlanations)

            SHAP helps identify which features
            contribute positively or negatively
            to an individual churn prediction.
            """
        )

    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Feature Engineering"
    )

    st.markdown(
        """
        The system uses business-focused engineered features
        in addition to the original customer attributes.

        **Engineered features include:**

        - Additional Services Count
        - Tenure Group
        - New Customer + Month-to-Month

        These features help the model capture customer
        behavior and retention patterns more effectively.
        """
    )

    # ========================================================
    # TECHNOLOGY STACK
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Technology Stack"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.markdown(
            """
            ### Python

            Core programming language
            """
        )

    with col2:

        st.markdown(
            """
            ###  Scikit-learn

            Machine learning pipeline
            """
        )

    with col3:

        st.markdown(
            """
            ###  SHAP

            Explainable AI
            """
        )

    with col4:

        st.markdown(
            """
            ### Streamlit

            Interactive web application
            """
        )

    # ========================================================
    # SYSTEM PIPELINE
    # ========================================================

    st.markdown("---")

    st.subheader(
        " System Pipeline"
    )

    st.code(
        """
Raw Customer Data
        ↓
Data Validation
        ↓
Feature Engineering
        ↓
Preprocessing
        ↓
Tuned Gradient Boosting Model
        ↓
Churn Probability
        ↓
30% Decision Threshold
        ↓
Risk Classification
        ↓
SHAP Explanation
        ↓
Personalized Retention Recommendations
        """,
        language="text"
    )

    # ========================================================
    # PROJECT HIGHLIGHTS
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Project Highlights"
    )

    highlights = [
        "End-to-end machine learning pipeline",
        "Tuned Gradient Boosting model",
        "Business-focused feature engineering",
        "Custom churn decision threshold",
        "Customer-level SHAP explainability",
        "SHAP-driven retention recommendations",
        "Interactive Streamlit dashboard",
        "Individual customer prediction",
        "Batch prediction support"
    ]

    for highlight in highlights:

        st.markdown(
            f" {highlight}"
        )

    # ========================================================
    # DEVELOPER
    # ========================================================

    st.markdown("---")

    st.subheader(
        "Project Overview"
    )

    st.markdown(
        """
        **Customer Churn Prediction System**

        An end-to-end machine learning project focused on
        customer churn prediction, explainable AI, and
        data-driven retention strategies.

        Built as a portfolio project to demonstrate the
        practical application of machine learning in a
        real-world business problem.
        """
    )

    st.info(
        "This system is intended as a decision-support tool. "
        "Predictions should be evaluated together with business "
        "context before taking customer retention actions."
    )  


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Churn Prediction System • "
    "Gradient Boosting + SHAP"
)
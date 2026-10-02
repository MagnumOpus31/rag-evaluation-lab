from pathlib import Path
import base64
import hashlib
import html
import time

import streamlit as st

from src.rag_pipeline import RAGPipeline

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Evaluation Lab",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
RAG_IMAGE = BASE_DIR / "assets" / "rag_pic.png"

# ============================================================
# IMAGE LOADER
# ============================================================


def get_image_data_uri(image_path):
    if not image_path.exists():
        return None

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    mime_type = mime_types.get(image_path.suffix.lower(), "image/png")
    return f"data:{mime_type};base64,{encoded}"


rag_image_uri = get_image_data_uri(RAG_IMAGE)

# A short hash changes the iframe HTML whenever the image file changes,
# so the browser never reuses an older hero image.
if RAG_IMAGE.exists():
    image_version = hashlib.sha256(RAG_IMAGE.read_bytes()).hexdigest()[:12]
else:
    image_version = "missing"

# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
<style>
html, body, [class*="css"] {
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.stApp {
    background:
        radial-gradient(circle at 50% -10%, rgba(105,105,255,0.055), transparent 35%),
        #ffffff;
    color: #111111;
}
.block-container { max-width: 1450px; padding-top: 1.2rem; padding-bottom: 3rem; }
#MainMenu, footer, header { visibility: hidden; }

/* STATUS */
.status-wrapper { display: flex; justify-content: center; margin-bottom: 2px; }
.status-pill {
    display: inline-flex; align-items: center; gap: 9px;
    padding: 7px 15px; border-radius: 999px;
    background: rgba(248,248,250,0.96); border: 1px solid #e7e7eb;
    color: #55555f; font-size: 12px; font-weight: 600; letter-spacing: 0.04em;
    box-shadow: 0 4px 15px rgba(0,0,0,0.035);
}
.status-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #32c46a;
    box-shadow: 0 0 0 4px rgba(50,196,106,0.11);
    animation: statusPulse 2s ease-in-out infinite;
}
@keyframes statusPulse {
    0%, 100% { transform: scale(1); opacity: 0.75; }
    50% { transform: scale(1.25); opacity: 1; }
}

/* SECTION HEADERS */
.section-kicker {
    margin-top: 20px; margin-bottom: 5px; color: #8a8a94;
    font-size: 11px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
}
.section-title {
    margin-bottom: 15px; color: #15151a;
    font-size: 27px; font-weight: 750; letter-spacing: -0.035em;
}

/* METRIC CARDS */
.metric-card {
    position: relative; padding: 19px 20px; min-height: 105px;
    border: 1px solid #ececf0; border-radius: 18px;
    background: rgba(255,255,255,0.92);
    box-shadow: 0 8px 30px rgba(0,0,0,0.035); overflow: hidden;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.metric-card:hover {
    transform: translateY(-3px); border-color: #dedee6;
    box-shadow: 0 14px 36px rgba(0,0,0,0.07);
}
.metric-label { color: #8a8a94; font-size: 11px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; }
.metric-value { margin-top: 9px; color: #18181d; font-size: 27px; font-weight: 750; letter-spacing: -0.04em; }
.metric-description { margin-top: 3px; color: #9999a2; font-size: 11px; }

/* QUERY */
.query-heading { margin-top: 34px; margin-bottom: 10px; color: #17171b; font-size: 22px; font-weight: 750; letter-spacing: -0.025em; }
.query-description { margin-bottom: 14px; color: #888891; font-size: 13px; }

div[data-testid="stTextInput"] input {
    height: 54px !important; border-radius: 14px !important;
    border: 1px solid #dedee5 !important; background: #ffffff !important;
    color: #15151a !important; font-size: 15px !important; padding-left: 17px !important;
    box-shadow: 0 5px 18px rgba(0,0,0,0.035) !important;
    transition: border-color .25s ease, box-shadow .25s ease !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color: #9a9aff !important;
    box-shadow: 0 0 0 4px rgba(104,104,255,0.08), 0 8px 25px rgba(0,0,0,0.05) !important;
}
div[data-testid="stTextInput"] input::placeholder { color: #a4a4ad !important; }

/* BUTTON */
div.stButton > button {
    width: 100%; height: 54px; border: none; border-radius: 14px;
    background: #17171c; color: white; font-size: 14px; font-weight: 700;
    box-shadow: 0 8px 20px rgba(0,0,0,0.09);
    transition: transform .2s ease, box-shadow .2s ease, background .2s ease;
}
div.stButton > button:hover { background: #292930; transform: translateY(-2px); box-shadow: 0 12px 28px rgba(0,0,0,0.13); }
div.stButton > button:active { transform: translateY(0); }

/* ANSWER */
.answer-panel {
    margin-top: 24px; padding: 25px 27px; border: 1px solid #e8e8ed; border-radius: 20px;
    background: #ffffff; box-shadow: 0 10px 35px rgba(0,0,0,0.045);
    animation: panelReveal .65s cubic-bezier(.22,1,.36,1) both;
}
@keyframes panelReveal {
    from { opacity: 0; transform: translateY(16px); }
    to { opacity: 1; transform: translateY(0); }
}
.answer-label { color: #8d8d97; font-size: 10px; font-weight: 800; letter-spacing: .15em; text-transform: uppercase; margin-bottom: 11px; }
.answer-text { color: #202027; font-size: 15px; line-height: 1.75; overflow-wrap: anywhere; word-break: break-word; white-space: normal; }

/* EVIDENCE */
.evidence-card {
    margin-top: 12px; padding: 17px 19px; border: 1px solid #ededf1; border-radius: 15px;
    background: #fafafd; transition: transform .2s ease, background .2s ease;
}
.evidence-card:hover { transform: translateX(3px); background: #f7f7fb; }
.evidence-score { margin-bottom: 8px; color: #777782; font-size: 11px; font-weight: 700; }
.evidence-text { color: #3b3b43; font-size: 13px; line-height: 1.65; overflow-wrap: anywhere; word-break: break-word; }

/* FOOTER */
.footer {
    margin-top: 50px; padding-top: 20px; border-top: 1px solid #eeeef2;
    color: #a0a0a8; font-size: 11px; text-align: center;
}
</style>
"""
)

# ============================================================
# STATUS PILL
# ============================================================

st.html(
    """
<div class="status-wrapper">
    <div class="status-pill">
        <span class="status-dot"></span>
        RAG SYSTEM ONLINE
    </div>
</div>
"""
)

# ============================================================
# HERO
#
# Layout fix: the heading block now sits at the very top of the
# scene and the image sits BELOW it (its own zone), so the text
# never overlaps the picture. Decorative layers (rings, dots,
# labels, background word) are all kept behind the image.
# ============================================================

HERO_HEIGHT = 800

if rag_image_uri:

    visual_html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<!-- RAG image version: {image_version} -->
<style>
* {{ box-sizing: border-box; }}

html, body {{
    margin: 0; padding: 0; width: 100%; height: 100%;
    overflow: hidden; background: transparent;
    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}

/* SCENE */
.scene {{
    position: relative; width: 100%; height: {HERO_HEIGHT - 10}px; overflow: hidden;
    background: radial-gradient(circle at 50% 58%, rgba(115,115,255,0.055), transparent 38%);
}}

/* HERO TEXT (top zone, normal flow, never overlaps the image) */
.hero-copy {{ position: relative; z-index: 30; padding: 6px 20px 0; text-align: center; }}

/* Eyebrow: "Evaluating [rotating word]" - words slide + blur through one slot */
.eyebrow {{
    display: inline-flex; align-items: baseline; gap: .7em; margin-bottom: 10px;
    color: #888893; font-size: 11px; font-weight: 800; letter-spacing: .2em; text-transform: uppercase;
    opacity: 0; animation: copyFade .8s cubic-bezier(.22,1,.36,1) .25s forwards;
}}
.rotator {{ display: inline-grid; text-align: left; color: #5555d8; }}
.rotator span {{
    grid-area: 1 / 1; opacity: 0;
    animation: wordCycle 9.6s cubic-bezier(.22,1,.36,1) infinite backwards;
}}
.rotator span:nth-child(1) {{ animation-delay: 2.2s; }}
.rotator span:nth-child(2) {{ animation-delay: 4.6s; }}
.rotator span:nth-child(3) {{ animation-delay: 7.0s; }}
.rotator span:nth-child(4) {{ animation-delay: 9.4s; }}
@keyframes wordCycle {{
    0% {{ opacity: 0; transform: translateY(80%); filter: blur(4px); }}
    5%, 20% {{ opacity: 1; transform: translateY(0); filter: blur(0); }}
    25%, 100% {{ opacity: 0; transform: translateY(-80%); filter: blur(4px); }}
}}

/* Title: decode effect. Each letter scrambles, slows down, then locks in
   with a pop + glow. Hover the title to decode it again. */
.js .main-title {{ visibility: hidden; }}
.js .main-title.ready {{ visibility: visible; }}
.main-title {{
    margin: 0; color: #141419; cursor: default;
    font-size: clamp(40px, 5.2vw, 66px); line-height: 1.05; font-weight: 900; letter-spacing: -.06em;
}}
.word {{ display: inline-block; white-space: nowrap; }}
.ch {{
    display: inline-block; text-align: center; color: var(--c, #141419);
    transition: transform .35s cubic-bezier(.34,1.56,.64,1), color .2s ease;
}}
.ch.hid {{ opacity: 0; }}
.ch.scr {{ color: #8c8cf5; filter: blur(.6px); }}
.ch.lock {{ animation: lockPop .6s cubic-bezier(.34,1.56,.64,1), lockGlow .9s ease-out; }}
.ch:hover {{ transform: translateY(-7px) skewX(-8deg); color: #5555d8; }}
@keyframes lockPop {{
    0% {{ transform: translateY(-10px) scale(1.28); }}
    100% {{ transform: none; }}
}}
@keyframes lockGlow {{
    0% {{ text-shadow: 0 0 22px rgba(104,104,233,.95); }}
    100% {{ text-shadow: 0 0 0 rgba(104,104,233,0); }}
}}

/* Subtitle: words rise out of a blur, one after another */
.hero-subtitle {{ width: min(560px, 85%); margin: 10px auto 0; color: #6a6a74; font-size: 14px; line-height: 1.6; }}
.hero-subtitle .w {{
    display: inline-block; opacity: 0;
    animation: wordIn .7s cubic-bezier(.22,1,.36,1) calc(1.9s + var(--i) * 55ms) forwards;
}}
@keyframes wordIn {{
    from {{ opacity: 0; transform: translateY(14px); filter: blur(6px); }}
    to {{ opacity: 1; transform: none; filter: blur(0); }}
}}
@keyframes copyFade {{
    from {{ opacity: 0; transform: translateY(22px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

/* Reduced motion: show the finished state, no movement */
.rm .hero-subtitle .w {{ opacity: 1; animation: none; }}
.rm .rotator span {{ animation: none; }}
.rm .rotator span:first-child {{ opacity: 1; }}
.rm .ch:hover {{ transform: none; }}

/* VISUAL ZONE: starts below the text */
.visual-center {{
    position: absolute; left: 50%; top: 190px;
    width: 540px; height: 540px;
    transform: translateX(-50%); transform-origin: top center;
    display: flex; align-items: center; justify-content: center;
    z-index: 8;
}}
.glow {{
    position: absolute; width: 540px; height: 540px; border-radius: 50%;
    background: radial-gradient(circle, rgba(106,106,255,.12), rgba(106,106,255,.03) 42%, transparent 72%);
    filter: blur(12px); z-index: 1;
    animation: glowPulse 4.5s ease-in-out infinite;
}}
@keyframes glowPulse {{
    0%, 100% {{ transform: scale(.96); opacity: .7; }}
    50% {{ transform: scale(1.06); opacity: 1; }}
}}

/* RINGS (centered on the image, kept behind it) */
.ring {{
    position: absolute; left: 50%; top: 50%; border-radius: 50%;
    border: 1px solid rgba(95,95,120,.11); z-index: 2;
    transform: translate(-50%, -50%);
}}
.ring.one {{ width: 580px; height: 580px; animation: rotateClockwise 24s linear infinite; }}
.ring.two {{ width: 640px; height: 640px; border-style: dashed; opacity: .55; animation: rotateCounter 34s linear infinite; }}
.ring.three {{ width: 700px; height: 700px; opacity: .32; animation: rotateClockwise 45s linear infinite; }}
@keyframes rotateClockwise {{
    from {{ transform: translate(-50%, -50%) rotate(0deg); }}
    to {{ transform: translate(-50%, -50%) rotate(360deg); }}
}}
@keyframes rotateCounter {{
    from {{ transform: translate(-50%, -50%) rotate(360deg); }}
    to {{ transform: translate(-50%, -50%) rotate(0deg); }}
}}

/* IMAGE */
.image-container {{
    position: relative; width: 520px; height: 520px;
    display: flex; align-items: center; justify-content: center;
    z-index: 12; opacity: 0;
    transform: translateY(40px) scale(.8);
    animation: imageEnter 1.2s cubic-bezier(.16,1,.3,1) .22s forwards;
}}
@keyframes imageEnter {{
    from {{ opacity: 0; transform: translateY(40px) scale(.8); filter: blur(10px); }}
    to {{ opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }}
}}
.image-float {{
    width: 100%; height: 100%;
    display: flex; align-items: center; justify-content: center;
    animation: floatingImage 5.5s ease-in-out 1.5s infinite;
}}
@keyframes floatingImage {{
    0%, 100% {{ transform: translateY(0); }}
    50% {{ transform: translateY(-8px); }}
}}
.rag-image {{
    width: 100%; height: 100%; object-fit: contain; display: block;
    background: transparent;
    filter: drop-shadow(0 18px 24px rgba(25,25,45,.14));
}}

/* DOTS (behind the image) */
.dot {{
    position: absolute; width: 8px; height: 8px; border-radius: 50%;
    background: #6868e9; z-index: 5; opacity: 0;
    box-shadow: 0 0 0 5px rgba(104,104,233,.09), 0 0 18px rgba(104,104,233,.32);
    animation: dotAppear .7s cubic-bezier(.22,1,.36,1) forwards;
}}
.dot-a {{ top: 330px; left: 24%; animation-delay: 1.15s; }}
.dot-b {{ top: 420px; right: 21%; animation-delay: 1.3s; }}
.dot-c {{ top: 640px; left: 29%; animation-delay: 1.45s; }}
.dot-d {{ top: 600px; right: 27%; animation-delay: 1.6s; }}
@keyframes dotAppear {{
    from {{ opacity: 0; transform: scale(0); }}
    to {{ opacity: 1; transform: scale(1); }}
}}

/* LABELS (behind the image) */
.system-label {{
    position: absolute; z-index: 5; display: flex; align-items: center; gap: 10px;
    color: #777780; font-size: 10px; font-weight: 800; letter-spacing: .13em;
    text-transform: uppercase; opacity: 0;
}}
.label-line {{ width: 42px; height: 1px; background: linear-gradient(90deg, transparent, #aaaab5); }}
.label-left {{ left: 7%; top: 440px; animation: labelLeft .9s cubic-bezier(.22,1,.36,1) 1.05s forwards; }}
.label-right {{ right: 7%; top: 400px; flex-direction: row-reverse; animation: labelRight .9s cubic-bezier(.22,1,.36,1) 1.25s forwards; }}
.label-bottom-left {{ left: 14%; top: 650px; animation: labelLeft .9s cubic-bezier(.22,1,.36,1) 1.45s forwards; }}
.label-bottom-right {{ right: 14%; top: 650px; flex-direction: row-reverse; animation: labelRight .9s cubic-bezier(.22,1,.36,1) 1.65s forwards; }}
@keyframes labelLeft {{
    from {{ opacity: 0; transform: translateX(-35px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}
@keyframes labelRight {{
    from {{ opacity: 0; transform: translateX(35px); }}
    to {{ opacity: 1; transform: translateX(0); }}
}}

/* CAPTION */
.hero-caption {{
    position: absolute; bottom: 10px; left: 0; right: 0; text-align: center; z-index: 20;
    color: #a0a0aa; font-size: 10px; font-weight: 700; letter-spacing: .16em; text-transform: uppercase;
    opacity: 0; animation: copyFade .8s cubic-bezier(.22,1,.36,1) 1.8s forwards;
}}

/* RESPONSIVE */
@media (max-width: 950px) {{
    .visual-center {{ transform: translateX(-50%) scale(.85); }}
    .system-label {{ display: none; }}
}}
@media (max-width: 700px) {{
    .main-title {{ font-size: 40px; }}
    .hero-subtitle {{ font-size: 12px; }}
    .visual-center {{ top: 175px; transform: translateX(-50%) scale(.62); }}
    .dot {{ display: none; }}
}}
@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{
        animation-duration: .01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: .01ms !important;
    }}
}}
</style>
</head>

<body>
<div class="scene">

    <div class="hero-copy">
        <div class="eyebrow">
            <span>Evaluating</span>
            <span class="rotator"><span>retrieval</span><span>evidence</span><span>generation</span><span>latency</span></span>
        </div>
        <h1 class="main-title" id="title">RAG Evaluation Lab</h1>
        <div class="hero-subtitle" id="sub">
            Retrieval, evidence, generation and evaluation
            brought together in one intelligent pipeline.
        </div>
    </div>

    <div class="visual-center">
        <div class="glow"></div>
        <div class="ring one"></div>
        <div class="ring two"></div>
        <div class="ring three"></div>
        <div class="image-container">
            <div class="image-float">
                <img class="rag-image" src="{rag_image_uri}" alt="RAG Evaluation Lab pipeline">
            </div>
        </div>
    </div>

    <div class="dot dot-a"></div>
    <div class="dot dot-b"></div>
    <div class="dot dot-c"></div>
    <div class="dot dot-d"></div>

    <div class="system-label label-left"><span>Documents</span><span class="label-line"></span></div>
    <div class="system-label label-right"><span>Embeddings</span><span class="label-line"></span></div>
    <div class="system-label label-bottom-left"><span>Retrieval</span><span class="label-line"></span></div>
    <div class="system-label label-bottom-right"><span>Evaluation</span><span class="label-line"></span></div>

    <div class="hero-caption">Retrieval-Augmented Generation · Evaluation · Evidence</div>

</div>
<script>
(function () {{
    var TITLE = "RAG Evaluation Lab";
    var GLYPHS = "01<>/{{}}[]#$%&*+=?ABCDEFGHJKLMNPQRSTUVWXYZ";
    var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var h1 = document.getElementById("title");
    var sub = document.getElementById("sub");
    var letters = [], raf = 0, running = false;

    document.documentElement.classList.add("js");
    if (reduce) document.body.classList.add("rm");

    // Colour ramps from ink -> indigo (middle of the title) -> ink
    function mix(a, b, t) {{ return Math.round(a + (b - a) * t); }}
    function colorAt(t) {{
        var m = Math.sin(Math.PI * t);
        return "rgb(" + mix(20, 85, m) + "," + mix(20, 85, m) + "," + mix(25, 216, m) + ")";
    }}

    // Split the title into per-letter spans (screen readers keep the real text)
    var words = TITLE.split(" ");
    var total = TITLE.replace(/ /g, "").length, k = 0;
    h1.setAttribute("aria-label", TITLE);
    h1.textContent = "";
    words.forEach(function (word, wi) {{
        var w = document.createElement("span");
        w.className = "word";
        w.setAttribute("aria-hidden", "true");
        for (var i = 0; i < word.length; i++) {{
            var s = document.createElement("span");
            s.className = "ch";
            s.textContent = word.charAt(i);
            s.setAttribute("data-final", word.charAt(i));
            s.style.setProperty("--c", colorAt(k / (total - 1)));
            w.appendChild(s);
            letters.push(s);
            k++;
        }}
        h1.appendChild(w);
        if (wi < words.length - 1) h1.appendChild(document.createTextNode(" "));
    }});

    // Split the subtitle into words for the staggered reveal
    var parts = sub.textContent.trim().split(/\\s+/);
    sub.setAttribute("aria-label", parts.join(" "));
    sub.textContent = "";
    parts.forEach(function (t, i) {{
        var s = document.createElement("span");
        s.className = "w";
        s.setAttribute("aria-hidden", "true");
        s.style.setProperty("--i", i);
        s.textContent = t;
        sub.appendChild(s);
        sub.appendChild(document.createTextNode(" "));
    }});

    // Freeze each letter's width (in em) so scrambling never shifts the layout
    function lockWidths() {{
        var fs = parseFloat(getComputedStyle(h1).fontSize);
        letters.forEach(function (s) {{ s.style.width = ""; }});
        letters.forEach(function (s) {{
            s.style.width = (s.getBoundingClientRect().width / fs) + "em";
        }});
    }}

    // Decode animation: letters cycle random glyphs, decelerate, then lock
    function decode(base, step, settle) {{
        cancelAnimationFrame(raf);
        running = true;
        var t0 = performance.now();
        var last = letters.map(function () {{ return 0; }});
        var done = letters.map(function () {{ return false; }});
        letters.forEach(function (s) {{
            s.classList.remove("lock", "scr");
            s.classList.add("hid");
            s.textContent = "";
        }});
        function frame(now) {{
            var el = now - t0, pending = 0;
            letters.forEach(function (s, i) {{
                if (done[i]) return;
                var t = el - (base + i * step);
                if (t < 0) {{ pending++; return; }}
                s.classList.remove("hid");
                if (t >= settle) {{
                    done[i] = true;
                    s.textContent = s.getAttribute("data-final");
                    s.classList.remove("scr");
                    void s.offsetWidth;
                    s.classList.add("lock");
                    return;
                }}
                pending++;
                var p = t / settle;
                if (now - last[i] > 35 + 150 * p * p) {{
                    last[i] = now;
                    s.textContent = GLYPHS.charAt(Math.floor(Math.random() * GLYPHS.length));
                    s.classList.add("scr");
                }}
            }});
            if (pending) {{ raf = requestAnimationFrame(frame); }} else {{ running = false; }}
        }}
        raf = requestAnimationFrame(frame);
    }}

    var ready = (document.fonts && document.fonts.ready) ? document.fonts.ready : Promise.resolve();
    ready.then(function () {{
        lockWidths();
        if (!reduce) decode(450, 65, 650);
        h1.classList.add("ready");
    }});

    if (!reduce) {{
        h1.addEventListener("mouseenter", function () {{
            if (!running) decode(0, 35, 380);
        }});
    }}
}})();
</script>
</body>
</html>
"""

    st.iframe(visual_html, height=HERO_HEIGHT)

else:
    st.error("RAG image not found.")
    st.code(str(RAG_IMAGE))

# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.html('<div class="section-kicker">System Overview</div>')
st.html('<div class="section-title">Evaluation Console</div>')

metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)


def metric_card(label, value, description):
    return f"""
<div class="metric-card">
    <div class="metric-label">{label}</div>
    <div class="metric-value">{value}</div>
    <div class="metric-description">{description}</div>
</div>
"""


with metric_col1:
    st.html(metric_card("Documents", "7", "Knowledge sources"))

with metric_col2:
    st.html(metric_card("Embeddings", "384D", "MiniLM vector space"))

with metric_col3:
    st.html(metric_card("Retrieval", "Top-K", "Semantic similarity"))

with metric_col4:
    st.html(metric_card("Evaluation", "0.3", "OOD score threshold"))

# ============================================================
# QUERY CONSOLE
# ============================================================

st.html('<div class="query-heading">Ask the RAG system</div>')

st.html(
    """
<div class="query-description">
    Enter a question and inspect the generated answer,
    retrieved evidence and system latency.
</div>
"""
)

query_col, button_col = st.columns([5, 1], gap="medium")

with query_col:
    query = st.text_input(
        "Query",
        placeholder="e.g. What are embeddings?",
        label_visibility="collapsed",
    )

with button_col:
    run_query = st.button("Run Query", use_container_width=True)

# ============================================================
# RAG PIPELINE
# ============================================================

if run_query and query.strip():

    with st.spinner("Running retrieval and generation..."):
        start_time = time.perf_counter()

        pipeline = RAGPipeline(
            chunk_size=50,
            overlap=10,
            score_threshold=0.3,
        )

        result = pipeline.answer(query.strip(), top_k=3)

        total_ui_time = time.perf_counter() - start_time

    # GENERATED ANSWER
    safe_answer = html.escape(result["answer"])

    st.html(
        f"""
<div class="answer-panel">
    <div class="answer-label">Generated Answer</div>
    <div class="answer-text">{safe_answer}</div>
</div>
"""
    )

    # RETRIEVED EVIDENCE
    st.html('<div class="section-kicker">Evidence</div>')
    st.html('<div class="section-title">Retrieved Context</div>')

    retrieved = result.get("retrieved", [])

    if retrieved:
        for index, item in enumerate(retrieved, start=1):
            score = item.get("score", 0.0)
            text = html.escape(item.get("text", ""))

            st.html(
                f"""
<div class="evidence-card">
    <div class="evidence-score">
        EVIDENCE {index} &nbsp;&nbsp;·&nbsp;&nbsp; SIMILARITY {score:.3f}
    </div>
    <div class="evidence-text">{text}</div>
</div>
"""
            )
    else:
        st.info("No relevant evidence was retrieved.")

    # LATENCY
    st.html('<div class="section-kicker">Performance</div>')
    st.html('<div class="section-title">Latency</div>')

    latency = result.get("latency", {})

    latency_col1, latency_col2, latency_col3 = st.columns(3)

    with latency_col1:
        st.html(
            metric_card(
                "Retrieval",
                f"{latency.get('retrieval_seconds', 0):.3f}s",
                "Vector search",
            )
        )

    with latency_col2:
        st.html(
            metric_card(
                "Generation",
                f"{latency.get('generation_seconds', 0):.2f}s",
                "Local LLM",
            )
        )

    with latency_col3:
        st.html(
            metric_card(
                "Total",
                f"{latency.get('total_seconds', total_ui_time):.2f}s",
                "End-to-end pipeline",
            )
        )

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
<div class="footer">
    RAG Evaluation Lab · Retrieval-Augmented Generation ·
    Semantic Search · Local LLM Evaluation
</div>
"""
)
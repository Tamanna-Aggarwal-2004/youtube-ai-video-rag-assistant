"""
YouTube AI Video Assistant — Premium Streamlit Frontend
=========================================================
IMPORTANT: All backend logic below (extract_video_id, get_transcript,
build_retriever, prompt, format_docs, the RAG chain composition, and the
summary prompt) is copied EXACTLY from the original script. Nothing about
how transcripts are fetched, how the vector store is built, or how the LLM
is prompted has been changed. Only the presentation layer (Streamlit UI,
CSS, layout, state handling) is new.
"""

import re
import time
import json
import urllib.request
import urllib.parse

import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# BACKEND — UNCHANGED. Same models, same prompts, same chain composition as
# the original script. Only wrapped in small functions so the UI can call
# them per-session instead of running once top-to-bottom.
# =============================================================================

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)

    if match:
        return match.group(1)

    raise ValueError("Invalid YouTube URL")


def get_transcript(video_id):
    api = YouTubeTranscriptApi()
    transcript = api.fetch(video_id, languages=["en"])
    text = " ".join(chunk.text for chunk in transcript)
    return text


def build_retriever(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = splitter.create_documents([text])

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embedding)

    return vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )


prompt = PromptTemplate(
    template="""
You are an AI assistant.

Answer ONLY from the transcript.

If the answer cannot be found in the transcript,
reply exactly:

"I couldn't find that information in this video. Please ask questions related to the uploaded video."

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


parser = StrOutputParser()


def build_rag_chain(retriever):
    """Exact same LCEL composition as the original script, just returned
    from a function so it can be built once per uploaded video."""
    return (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough()
            }
        )
        | prompt
        | llm
        | parser
    )


def generate_summary(transcript):
    """Exact same summary prompt as the original script."""
    summary_prompt = f"""
Analyze the following transcript and generate:

1. Executive Summary

2. Key Points

3. Objectives

4. Important Concepts

5. Important Facts

6. Conclusion

Transcript:

{transcript}
"""
    return llm.invoke(summary_prompt).content


# =============================================================================
# UI-ONLY HELPERS (no backend/RAG behaviour here — display formatting only)
# =============================================================================

SECTION_ICONS = {
    "executive summary": "🧭",
    "key points": "🔑",
    "objectives": "🎯",
    "important concepts": "🧠",
    "important facts": "📌",
    "conclusion": "✅",
}

SUGGESTED_QUESTIONS = [
    "Summarize this video",
    "What are the key concepts?",
    "Explain like I'm a beginner",
    "What are the important takeaways?",
    "What examples were discussed?",
]


def fetch_video_meta(video_id):
    """Best-effort, UI-only metadata lookup (title/channel/thumbnail) via
    YouTube's public oEmbed endpoint. Purely cosmetic — does not touch the
    transcript/RAG pipeline. Falls back gracefully if unavailable."""
    fallback = {
        "title": "YouTube Video",
        "author": "Unknown Channel",
        "thumbnail": f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }
    try:
        oembed_url = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "format": "json",
        })
        with urllib.request.urlopen(oembed_url, timeout=6) as resp:
            data = json.loads(resp.read().decode())
        return {
            "title": data.get("title", fallback["title"]),
            "author": data.get("author_name", fallback["author"]),
            "thumbnail": data.get("thumbnail_url", fallback["thumbnail"]),
        }
    except Exception:
        return fallback


def parse_summary_sections(raw_text):
    """Splits the numbered summary output into (title, body) sections for
    card rendering. Purely presentational parsing of the LLM's own output."""
    pattern = r"(?:^|\n)\s*#{0,3}\s*\d+[\.\):]\s*([A-Za-z][A-Za-z /]+?)\s*\**\s*\n"
    matches = list(re.finditer(pattern, raw_text))
    sections = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        body = raw_text[start:end].strip()
        if body:
            sections.append((title, body))
    if not sections:
        sections = [("Executive Summary", raw_text.strip())]
    return sections


def icon_for(title):
    key = title.strip().lower()
    for k, v in SECTION_ICONS.items():
        if k in key:
            return v
    return "📄"


def render_error_card(kind, detail=""):
    presets = {
        "invalid_url": ("🚫", "That doesn't look like a YouTube link",
                         "Please paste a valid YouTube video URL, e.g. https://www.youtube.com/watch?v=VIDEO_ID"),
        "no_transcript": ("🗒️", "No transcript available for this video",
                           "This video doesn't have an English transcript/captions. Try a different video."),
        "too_long": ("⏱️", "This video may be too long to process",
                      "Try a shorter video, or give it a bit more time — processing can take longer for lengthy transcripts."),
        "network": ("📡", "Network hiccup",
                     "We couldn't reach YouTube or the AI service. Check your connection and try again."),
        "no_answer_context": ("🤔", "Outside this video's content",
                               "I couldn't find that in the transcript — try asking something covered in the video."),
        "generic": ("⚠️", "Something went wrong",
                     "We hit an unexpected issue while processing this request. Please try again."),
    }
    icon, title, msg = presets.get(kind, presets["generic"])
    st.markdown(f"""
    <div class="error-card">
        <div class="error-icon">{icon}</div>
        <div>
            <div class="error-title">{title}</div>
            <div class="error-msg">{msg}{f"<br><span class='error-detail'>{detail}</span>" if detail else ""}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="YouTube AI Video Assistant",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# GLOBAL STYLE — design tokens
# Palette: deep signal-blue background, violet→cyan "transcript signal"
# gradient accent, warm-neutral text. Space Grotesk for display type,
# Inter for body, JetBrains Mono for URLs/ids/durations.
# =============================================================================

st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root{
    --bg:#0A0E17;
    --bg-soft:#0D1220;
    --surface:#131A2B;
    --surface-2:#1A2338;
    --border:rgba(255,255,255,0.07);
    --border-strong:rgba(255,255,255,0.14);
    --text:#EDEFF7;
    --text-muted:#8B93A7;
    --text-faint:#5D6478;
    --accent-1:#7C5CFC;
    --accent-2:#22D3EE;
    --accent-grad:linear-gradient(120deg,#7C5CFC 0%,#5B8DF7 50%,#22D3EE 100%);
    --success:#34D399;
    --warning:#FBBF24;
    --danger:#F87171;
    --radius-lg:20px;
    --radius-md:14px;
    --radius-sm:10px;
    --shadow-soft:0 8px 30px rgba(0,0,0,0.35);
    --shadow-glow:0 0 0 1px rgba(124,92,252,0.15), 0 8px 24px rgba(124,92,252,0.10);
}

html, body, [class*="css"]{
    font-family:'Inter', sans-serif;
}

.stApp{
    background:
        radial-gradient(1100px 600px at 15% -10%, rgba(124,92,252,0.16), transparent 60%),
        radial-gradient(900px 500px at 100% 0%, rgba(34,211,238,0.10), transparent 55%),
        var(--bg);
}

#MainMenu, footer, header[data-testid="stHeader"]{
    background:transparent;
}
header[data-testid="stHeader"]{
    background:transparent !important;
    box-shadow:none;
}

.block-container{
    padding-top:1.6rem;
    padding-bottom:6rem;
    max-width:1180px;
}

h1,h2,h3,h4{
    font-family:'Space Grotesk', sans-serif !important;
    color:var(--text) !important;
    letter-spacing:-0.01em;
}
p, span, div, label{
    color:var(--text);
}
code, .mono{
    font-family:'JetBrains Mono', monospace !important;
}

/* ---------- Hero header ---------- */
.hero-wrap{
    display:flex; align-items:center; gap:18px;
    margin-bottom:6px;
}
.hero-badge{
    width:56px;height:56px;border-radius:16px;
    background:var(--accent-grad);
    display:flex;align-items:center;justify-content:center;
    font-size:28px;
    box-shadow:var(--shadow-glow);
    flex-shrink:0;
}
.hero-title{
    font-size:2.05rem; font-weight:700; margin:0; line-height:1.15;
}
.hero-subtitle{
    color:var(--text-muted); font-size:1.0rem; margin-top:4px;
}
.hero-divider{
    height:1px;
    background:linear-gradient(90deg, rgba(124,92,252,0.55), rgba(34,211,238,0.35), transparent);
    margin:22px 0 28px 0;
    border:none;
}

/* ---------- Waveform signature divider ---------- */
.wave-divider{
    display:flex; align-items:center; gap:3px; height:22px; margin:26px 0 18px 0; opacity:0.85;
}
.wave-bar{
    width:3px; border-radius:3px; background:var(--accent-grad);
    animation:wavepulse 1.6s ease-in-out infinite;
}
@keyframes wavepulse{
    0%,100%{ height:6px; opacity:0.5;}
    50%{ height:20px; opacity:1;}
}

/* ---------- Cards ---------- */
.card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    padding:22px 24px;
    box-shadow:var(--shadow-soft);
    margin-bottom:18px;
}
.card:hover{
    border-color:var(--border-strong);
    transition:border-color 0.25s ease;
}
.insight-card{
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    padding:24px 28px;
    margin:0 auto 18px auto;
    box-shadow:var(--shadow-soft);
    position:relative;
    overflow:hidden;
    width:100%;
}
.insight-card::before{
    content:"";
    position:absolute; left:0; top:0; bottom:0; width:3px;
    background:var(--accent-grad);
}
.insight-header{
    display:flex; align-items:center; gap:10px; margin-bottom:10px;
}
.insight-icon{
    font-size:20px;
}
.insight-title{
    font-family:'Space Grotesk', sans-serif;
    font-weight:600; font-size:1.05rem; color:var(--text);
}
.insight-body{
    color:var(--text-muted); font-size:0.94rem; line-height:1.65; white-space:pre-wrap;
}

/* ---------- Video info card ---------- */
.video-card{
    display:flex; gap:20px;
    background:var(--surface);
    border:1px solid var(--border);
    border-radius:var(--radius-lg);
    padding:18px;
    box-shadow:var(--shadow-soft);
    margin-bottom:8px;
}
.video-thumb{
    width:220px; min-width:220px; border-radius:var(--radius-md);
    object-fit:cover; border:1px solid var(--border);
}
.video-meta-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:1.15rem; font-weight:600; margin-bottom:6px;
}
.video-meta-row{
    color:var(--text-muted); font-size:0.9rem; margin-bottom:4px;
    display:flex; align-items:center; gap:6px;
}
.video-pill{
    display:inline-flex; align-items:center; gap:6px;
    background:var(--surface-2);
    border:1px solid var(--border);
    border-radius:100px; padding:4px 12px;
    font-size:0.78rem; color:var(--text-muted);
    margin-top:8px; margin-right:6px;
    font-family:'JetBrains Mono', monospace;
}

/* ---------- Empty state ---------- */
.empty-state{
    text-align:center; padding:64px 20px 46px 20px;
}
.empty-icon{
    font-size:56px; margin-bottom:14px;
    filter:drop-shadow(0 6px 20px rgba(124,92,252,0.35));
}
.empty-title{
    font-family:'Space Grotesk', sans-serif;
    font-size:1.5rem; font-weight:600; margin-bottom:8px;
}
.empty-desc{
    color:var(--text-muted); font-size:0.98rem; max-width:460px; margin:0 auto 14px auto;
}
.empty-example{
    display:inline-block; margin-top:6px;
    background:var(--surface-2); border:1px solid var(--border);
    border-radius:100px; padding:8px 16px; font-size:0.85rem;
    color:var(--accent-2); font-family:'JetBrains Mono', monospace;
}

/* ---------- Error cards ---------- */
.error-card{
    display:flex; gap:14px; align-items:flex-start;
    background:rgba(248,113,113,0.08);
    border:1px solid rgba(248,113,113,0.35);
    border-radius:var(--radius-md);
    padding:16px 18px; margin:14px 0;
}
.error-icon{ font-size:22px; margin-top:1px; }
.error-title{ font-weight:600; color:#FCA5A5; margin-bottom:3px; font-family:'Space Grotesk', sans-serif;}
.error-msg{ color:#E8B4B4; font-size:0.9rem; line-height:1.5; }
.error-detail{ color:#C98F8F; font-size:0.78rem; font-family:'JetBrains Mono', monospace; }

/* ---------- Progress steps ---------- */
.step-row{
    display:flex; align-items:center; gap:12px; padding:10px 4px;
    border-bottom:1px solid var(--border);
}
.step-row:last-child{ border-bottom:none; }
.step-dot{
    width:26px; height:26px; border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-size:13px; flex-shrink:0;
    border:1px solid var(--border-strong);
    color:var(--text-faint);
    background:var(--surface-2);
}
.step-dot.done{
    background:var(--accent-grad); color:white; border:none;
}
.step-dot.active{
    background:var(--surface-2); border:1px solid var(--accent-1);
    color:var(--accent-2);
    animation:pulseGlow 1.1s ease-in-out infinite;
}
@keyframes pulseGlow{
    0%,100%{ box-shadow:0 0 0 0 rgba(124,92,252,0.35);}
    50%{ box-shadow:0 0 0 6px rgba(124,92,252,0.10);}
}
.step-label{ font-size:0.92rem; color:var(--text-muted); }
.step-label.done{ color:var(--text); }
.step-label.active{ color:var(--text); font-weight:500; }

/* ---------- Chat bubbles ---------- */
.chat-scroll{
    max-height:560px; overflow-y:auto; padding:6px 6px 10px 6px; margin-bottom:6px;
}
.msg-row{ display:flex; margin:10px 0; }
.msg-row.user{ justify-content:flex-end; }
.msg-row.assistant{ justify-content:flex-start; }
.bubble{
    max-width:72%; padding:13px 16px; border-radius:16px; font-size:0.94rem; line-height:1.55;
    box-shadow:0 4px 14px rgba(0,0,0,0.25);
}
.bubble.user{
    background:linear-gradient(135deg,#3B82F6,#2563EB);
    color:white; border-bottom-right-radius:4px;
}
.bubble.assistant{
    background:var(--surface-2);
    border:1px solid var(--border);
    color:var(--text); border-bottom-left-radius:4px;
    white-space:pre-wrap;
}
.msg-tag{
    font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em;
    color:var(--text-faint); margin-bottom:4px; font-family:'JetBrains Mono', monospace;
}

/* ---------- Suggested question chips ---------- */
div[data-testid="stButton"] button{
    background:var(--surface-2) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:100px !important;
    font-size:0.86rem !important;
    padding:6px 16px !important;
    transition:all 0.2s ease !important;
}
div[data-testid="stButton"] button:hover{
    border-color:var(--accent-1) !important;
    color:var(--accent-2) !important;
    background:rgba(124,92,252,0.08) !important;
}

/* Primary analyze button */
.analyze-btn button{
    background:var(--accent-grad) !important;
    color:white !important;
    border:none !important;
    border-radius:14px !important;
    font-weight:600 !important;
    font-size:1.02rem !important;
    padding:12px 0 !important;
    box-shadow:var(--shadow-glow) !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:var(--bg-soft);
    border-right:1px solid var(--border);
}
section[data-testid="stSidebar"] .card{
    background:var(--surface);
}

/* Text input */
div[data-testid="stTextInput"] input{
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:14px !important;
    padding:14px 16px !important;
    font-size:0.98rem !important;
}
div[data-testid="stTextInput"] input:focus{
    border-color:var(--accent-1) !important;
    box-shadow:0 0 0 3px rgba(124,92,252,0.15) !important;
}

/* Chat input */
div[data-testid="stChatInput"] textarea{
    background:var(--surface) !important;
    border:1px solid var(--border) !important;
    color:var(--text) !important;
    border-radius:14px !important;
}

hr{ border-color:var(--border) !important; }

::-webkit-scrollbar{ width:8px; height:8px; }
::-webkit-scrollbar-track{ background:transparent; }
::-webkit-scrollbar-thumb{ background:#2A3350; border-radius:8px; }

@media (max-width: 768px){
    .video-card{ flex-direction:column; }
    .video-thumb{ width:100%; }
    .bubble{ max-width:88%; }
    .hero-title{ font-size:1.5rem; }
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SESSION STATE
# =============================================================================

_defaults = {
    "stage": "idle",          # idle -> processing -> ready
    "video_id": None,
    "video_meta": None,
    "transcript": None,
    "retriever": None,
    "rag_chain": None,
    "summary_raw": None,
    "chat_history": [],       # list of (role, text)
    "pending_question": None,
    "error": None,
    "url_value": "",
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


def reset_app():
    for _k, _v in _defaults.items():
        st.session_state[_k] = _v


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px;">
        <div style="width:38px;height:38px;border-radius:11px;background:linear-gradient(120deg,#7C5CFC,#22D3EE);
                    display:flex;align-items:center;justify-content:center;font-size:19px;">🎥</div>
        <div style="font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:1.05rem;">
            YT AI Assistant
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Chat with any YouTube video's transcript.")
    st.markdown("---")

    with st.expander("ℹ️ About this project", expanded=False):
        st.markdown(
            "An AI assistant that reads a YouTube video's transcript, "
            "builds a searchable knowledge base from it, and lets you chat "
            "with the video — answers come **only** from what was actually said."
        )

    with st.expander("✨ Features"):
        st.markdown(
            "- Automatic transcript extraction\n"
            "- Semantic search over video content (FAISS)\n"
            "- Executive summary, key points, objectives, concepts\n"
            "- Retrieval-Augmented Generation chat\n"
            "- Stays grounded — won't answer outside the video"
        )

    with st.expander("🛠️ Tech stack"):
        st.markdown(
            "- **UI:** Streamlit\n"
            "- **Orchestration:** LangChain\n"
            "- **Vector store:** FAISS\n"
            "- **Embeddings:** HuggingFace `all-MiniLM-L6-v2`\n"
            "- **LLM:** Groq (`llama-3.1-8b-instant`)"
        )

    with st.expander("📖 Instructions"):
        st.markdown(
            "1. Paste a YouTube URL\n"
            "2. Click **Analyze Video**\n"
            "3. Read the generated insights\n"
            "4. Ask anything about the video in the chat"
        )

    with st.expander("🔗 Example URLs"):
        st.code("https://www.youtube.com/watch?v=aircAruvnKk", language="text")
        st.code("https://www.youtube.com/watch?v=dQw4w9WgXcQ", language="text")

    with st.expander("⚙️ Settings"):
        st.caption("Runtime configuration (read-only)")
        st.markdown(f"""
        <span class="video-pill">chunk_size: 1000</span>
        <span class="video-pill">chunk_overlap: 200</span>
        <span class="video-pill">retriever_k: 4</span>
        """, unsafe_allow_html=True)
        key_status = "✅ configured" if os.getenv("GROQ_API_KEY") else "⚠️ missing GROQ_API_KEY"
        st.caption(f"Groq API key: {key_status}")
        if st.session_state.stage != "idle":
            if st.button("🔄 Analyze a different video", use_container_width=True):
                reset_app()
                st.rerun()

    st.markdown("---")
    st.caption("Built by **Your Name** · Powered by LangChain + Groq")

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-badge">🎥</div>
    <div>
        <div class="hero-title">YouTube AI Video Assistant</div>
        <div class="hero-subtitle">Upload a YouTube video and chat with its content using AI.</div>
    </div>
</div>
<hr class="hero-divider"/>
""", unsafe_allow_html=True)

# =============================================================================
# PIPELINE RUNNER — calls the unchanged backend functions in sequence,
# rendering a live progress card while it works.
# =============================================================================

PIPELINE_STEPS = [
    "Fetching transcript",
    "Splitting documents",
    "Creating embeddings",
    "Building vector store",
    "Generating summary",
    "Preparing chat assistant",
]


def run_pipeline(url):
    progress_holder = st.empty()

    def render_steps(active_idx, done_idx):
        rows = ""
        for i, label in enumerate(PIPELINE_STEPS):
            if i < done_idx:
                cls, dot = "done", "✓"
            elif i == active_idx:
                cls, dot = "active", "●"
            else:
                cls, dot = "", str(i + 1)
            rows += f"""
            <div class="step-row">
                <div class="step-dot {cls}">{dot}</div>
                <div class="step-label {cls}">{label}</div>
            </div>"""
        progress_holder.markdown(f'<div class="card">{rows}</div>', unsafe_allow_html=True)

    try:
        # Step 0: validate + fetch transcript
        render_steps(0, 0)
        video_id = extract_video_id(url)
        video_meta = fetch_video_meta(video_id)
        transcript = get_transcript(video_id)

        # Step 1 & 2 happen inside build_retriever, but we show them distinctly
        render_steps(1, 1)
        time.sleep(0.15)
        render_steps(2, 2)
        retriever = build_retriever(transcript)

        render_steps(3, 3)
        time.sleep(0.15)

        render_steps(4, 4)
        summary_raw = generate_summary(transcript)

        render_steps(5, 5)
        rag_chain = build_rag_chain(retriever)
        time.sleep(0.2)

        render_steps(6, 6)
        progress_holder.empty()

        st.session_state.video_id = video_id
        st.session_state.video_meta = video_meta
        st.session_state.transcript = transcript
        st.session_state.retriever = retriever
        st.session_state.rag_chain = rag_chain
        st.session_state.summary_raw = summary_raw
        st.session_state.stage = "ready"
        st.session_state.error = None

    except ValueError:
        progress_holder.empty()
        st.session_state.error = ("invalid_url", "")
        st.session_state.stage = "idle"
    except (NoTranscriptFound, TranscriptsDisabled):
        progress_holder.empty()
        st.session_state.error = ("no_transcript", "")
        st.session_state.stage = "idle"
    except Exception as e:
        progress_holder.empty()
        st.exception(e)   # TEMP: shows full traceback in the app — remove after debugging
        msg = str(e).lower()
        if "network" in msg or "timeout" in msg or "connection" in msg:
            st.session_state.error = ("network", "")
        else:
            st.session_state.error = ("generic", "")
        st.session_state.stage = "idle"


# =============================================================================
# INPUT SECTION (idle state)
# =============================================================================

if st.session_state.stage == "idle":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    col1, col2 = st.columns([5, 1.3])
    with col1:
        url_input = st.text_input(
            "YouTube URL",
            value=st.session_state.url_value,
            placeholder="Paste a YouTube URL here...",
            label_visibility="collapsed",
        )
    with col2:
        st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
        analyze_clicked = st.button("Analyze Video ▶", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state.error:
        kind, detail = st.session_state.error
        render_error_card(kind, detail)

    if analyze_clicked:
        if not url_input or not url_input.strip():
            st.session_state.error = ("invalid_url", "")
        else:
            st.session_state.url_value = url_input.strip()
            st.session_state.error = None
            with st.spinner("Warming up the pipeline..."):
                run_pipeline(url_input.strip())
            st.rerun()

    if not st.session_state.error:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🪄</div>
            <div class="empty-title">Start by pasting a YouTube video</div>
            <div class="empty-desc">
                Drop in any YouTube link with English captions and I'll read the whole
                transcript, summarize it, and let you chat with it directly.
            </div>
            <div class="empty-example">https://www.youtube.com/watch?v=aircAruvnKk</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# READY STATE — video card, insights, chat
# =============================================================================

if st.session_state.stage == "ready":
    meta = st.session_state.video_meta or {}
    vid = st.session_state.video_id

    # ---- Video info card ----
    st.markdown(f"""
    <div class="video-card">
        <img class="video-thumb" src="{meta.get('thumbnail', '')}" />
        <div>
            <div class="video-meta-title">{meta.get('title', 'YouTube Video')}</div>
            <div class="video-meta-row">📺 {meta.get('author', 'Unknown Channel')}</div>
            <div class="video-meta-row">🔗 <span class="mono">youtube.com/watch?v={vid}</span></div>
            <span class="video-pill">id: {vid}</span>
            <span class="video-pill">status: ready ✓</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="wave-divider">
        <div class="wave-bar" style="animation-delay:0s"></div>
        <div class="wave-bar" style="animation-delay:0.1s"></div>
        <div class="wave-bar" style="animation-delay:0.2s"></div>
        <div class="wave-bar" style="animation-delay:0.3s"></div>
        <div class="wave-bar" style="animation-delay:0.4s"></div>
        <div class="wave-bar" style="animation-delay:0.5s"></div>
        <div class="wave-bar" style="animation-delay:0.6s"></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### 🧩 AI-generated insights")

    sections = parse_summary_sections(st.session_state.summary_raw or "")
    for title, body in sections:
        st.markdown(f"""
            <div class="insight-card">
                <div class="insight-header">
                    <div class="insight-icon">{icon_for(title)}</div>
                    <div class="insight-title">{title}</div>
                </div>
                <div class="insight-body">{body}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 💬 Chat with this video")

    # ---- Suggested questions ----
    st.caption("Try asking:")
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with chip_cols[i]:
            if st.button(q, key=f"chip_{i}", use_container_width=True):
                st.session_state.pending_question = q

    # ---- Render chat history ----
    st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"""
            <div class="msg-row user">
                <div>
                    <div class="msg-tag" style="text-align:right;">You</div>
                    <div class="bubble user">{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="msg-row assistant">
                <div>
                    <div class="msg-tag">Assistant</div>
                    <div class="bubble assistant">{text}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    typed_placeholder = st.empty()

    # ---- Chat input ----
    chat_box = st.chat_input("Ask something about this video...")
    question_to_ask = st.session_state.pending_question or chat_box

    if question_to_ask:
        st.session_state.pending_question = None
        st.session_state.chat_history.append(("user", question_to_ask))

        try:
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(question_to_ask)
        except Exception:
            answer = None

        if answer is None:
            render_error_card("network")
        else:
            # lightweight typing effect
            words = answer.split(" ")
            shown = ""
            for w in words:
                shown += w + " "
                typed_placeholder.markdown(f"""
                <div class="msg-row assistant">
                    <div>
                        <div class="msg-tag">Assistant</div>
                        <div class="bubble assistant">{shown}▍</div>
                    </div>
                </div>""", unsafe_allow_html=True)
                time.sleep(0.012)
            typed_placeholder.empty()
            st.session_state.chat_history.append(("assistant", answer))

        st.rerun()

    if st.session_state.error:
        kind, detail = st.session_state.error
        render_error_card(kind, detail)
        st.session_state.error = None

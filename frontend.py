import streamlit as st
import sqlite3
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChatBot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Replace ONLY the CSS block in your current code with this ──

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

/* ─────────────────────────────────────────────
   GLOBAL
───────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"]{
    background:#f7f4ef !important;
    font-family:'Poppins',sans-serif !important;
    color:#222 !important;
}

*{
    box-sizing:border-box;
}

---------------------------------------------------

/* remove top spacing */
.block-container{
    padding-top:1rem !important;
    padding-bottom:0rem !important;
    max-width:100% !important;
}

/* ─────────────────────────────────────────────
   SIDEBAR
───────────────────────────────────────────── */
[data-testid="stSidebar"]{
    background:#f8f4ee !important;
    border-right:1px solid #eadfd2 !important;
}

.sidebar-header{
    padding:22px 20px 14px 20px;
}

.sidebar-title{
    font-size:38px;
    font-weight:700;
    color:#f57c00;
    margin:0;
    line-height:1;
}

.sidebar-sub{
    font-size:15px;
    color:#f57c00;
    margin-top:6px;
    font-weight:500;
}

/* ─────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────── */
.stButton > button{
    border:none !important;
    border-radius:16px !important;

    background:linear-gradient(
        135deg,
        #ff7b00 0%,
        #ff9d2f 100%
    ) !important;

    color:white !important;

    font-weight:600 !important;

    padding:14px 18px !important;

    transition:0.2s ease !important;

    box-shadow:none !important;
}

.stButton > button:hover{
    transform:translateY(-1px);
    opacity:0.96;
}

/* ─────────────────────────────────────────────
   THREADS
───────────────────────────────────────────── */
.section-label{
    color:#f57c00;
    font-size:12px;
    font-weight:700;
    letter-spacing:1px;
    margin-top:14px;
}

.thread-item{
    background:transparent;
    border-radius:18px;
    padding:14px;
    transition:0.18s ease;
}

.thread-item:hover{
    background:#f5e7d6;
}

.thread-item.active{
    background:#f5e7d6;
}

.thread-label{
    font-size:13px;
    color:#222;
    font-weight:500;
}

.thread-id-small{
    color:#8d7d6d;
    font-size:11px;
}

/* ─────────────────────────────────────────────
   CHAT HEADER
───────────────────────────────────────────── */
.chat-title{
    font-size:54px;
    font-weight:700;
    color:#f57c00;
    margin-bottom:8px;
    line-height:1;
}

.chat-thread-tag{
    display:inline-block;

    background:#f5e7d6;

    color:#b56700;

    border-radius:12px;

    padding:8px 14px;

    font-size:13px;

    margin-bottom:28px;
}

/* ─────────────────────────────────────────────
   MESSAGES
───────────────────────────────────────────── */
.msg-label{
    font-size:12px;
    font-weight:700;
    color:#f57c00;
    margin-bottom:8px;
}

.msg-label.user{
    text-align:right;
}

.msg-bubble{
    padding:18px 22px;
    border-radius:18px;
    line-height:1.7;
    font-size:15px;
}

.msg-bubble.user{
    background:linear-gradient(
        135deg,
        #ff7b00 0%,
        #ff9d2f 100%
    );

    color:white;

    border-top-right-radius:4px;
}

.msg-bubble.ai{
    background:white;

    border:1px solid #f0e4d6;

    color:#222;

    border-top-left-radius:4px;
}

/* ─────────────────────────────────────────────
   INPUT AREA
───────────────────────────────────────────── */
.input-wrapper{

    position:sticky;
    bottom:0;

    background:#f8f4ee;

    padding-top:0px !important;

    border:none !important;

    box-shadow:none !important;
}

/* textarea */
[data-testid="stTextArea"]{
    margin-top:0 !important;
}

[data-testid="stTextArea"] textarea{

    background:white !important;

    border:1.5px solid #ffb36b !important;

    border-radius:18px !important;

    padding:18px !important;

    font-size:16px !important;

    color:#222 !important;

    min-height:120px !important;

    box-shadow:none !important;
}

[data-testid="stTextArea"] textarea:focus{
    border:1.5px solid #ff8c1a !important;
    box-shadow:none !important;
}

/* remove label spacing */
[data-testid="stTextArea"] label{
    display:none !important;
}

/* ─────────────────────────────────────────────
   FILE UPLOADER
───────────────────────────────────────────── */

/* whole uploader */
[data-testid="stFileUploader"]{
    background:transparent !important;
    border:none !important;
    padding-top:0 !important;
}

/* remove drag and drop area */
[data-testid="stFileUploaderDropzone"]{
    background:transparent !important;
    border:none !important;
    padding:0 !important;
}

/* remove ALL text below uploader */
[data-testid="stFileUploaderDropzoneInstructions"]{
    display:none !important;
}

/* remove secondary text */
[data-testid="stFileUploaderFileData"]{
    display:none !important;
}

/* upload button */
[data-testid="stFileUploader"] section button{

    background:white !important;

    border:1px solid #e4d6c8 !important;

    border-radius:16px !important;

    color:#222 !important;

    height:58px !important;

    padding:0px 24px !important;

    font-weight:500 !important;
}

/* ─────────────────────────────────────────────
   SEND BUTTON
───────────────────────────────────────────── */
div[data-testid="column"]:last-child .stButton button{

    width:60px !important;
    height:60px !important;

    border-radius:16px !important;

    padding:0 !important;

    font-size:20px !important;
}

/* ─────────────────────────────────────────────
   WELCOME
───────────────────────────────────────────── */
.welcome-wrap{
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    height:70vh;
}

.welcome-heading{
    font-size:60px;
    color:#f57c00;
    font-weight:700;
}

.welcome-sub{
    color:#8c7b6a;
    font-size:16px;
    max-width:650px;
    text-align:center;
    line-height:1.8;
}

/* ─────────────────────────────────────────────
   SCROLLBAR
───────────────────────────────────────────── */
::-webkit-scrollbar{
    width:6px;
}

::-webkit-scrollbar-thumb{
    background:#d6c3b2;
    border-radius:20px;
}

/* ─────────────────────────────────────────────
   IMAGE
───────────────────────────────────────────── */
img{
    border-radius:16px !important;
}

</style>
""", unsafe_allow_html=True)


# ── Backend imports (lazy, so Streamlit can still render if backend missing) ──
@st.cache_resource
def load_backend():
    try:
        import sqlite3 as _sqlite3
        from langgraph.checkpoint.sqlite import SqliteSaver
        from workflow import build_graph          # your graph module
        from langchain_core.messages import HumanMessage as HM

        conn = _sqlite3.connect("chatbot.db", check_same_thread=False)
        checkpointer = SqliteSaver(conn=conn)
        chatbot = build_graph(checkpointer=checkpointer)
        return chatbot, checkpointer, HM
    except Exception as e:
        return None, None, None


chatbot, checkpointer, HM = load_backend()


# ── Thread helpers ──────────────────────────────────────────────────────────────
def retrieve_all_threads():
    if checkpointer is None:
        return st.session_state.get("mock_threads", [])
    try:
        all_threads = set()
        for checkpoint in checkpointer.list(None):
            all_threads.add(checkpoint.config["configurable"]["thread_id"])
        return sorted(list(all_threads), reverse=True)
    except Exception:
        return []


def get_thread_history(thread_id: str) -> list[dict]:
    """Pull messages for a thread from the checkpointer state."""
    if checkpointer is None:
        return st.session_state.get("mock_history", {}).get(thread_id, [])
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = chatbot.get_state(config)
        messages = []
        for msg in state.values.get("messages", []):
            if hasattr(msg, "type"):
                role = "user" if msg.type == "human" else "ai"
            else:
                role = "user" if isinstance(msg, HumanMessage) else "ai"
            messages.append({"role": role, "content": msg.content})
        return messages
    except Exception:
        return []


def invoke_chatbot(thread_id: str, user_text: str, image_bytes=None) -> str:
    if chatbot is None:
        # Mock response for dev/demo
        return f"[Mock] Echo: {user_text}"
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "query": user_text,
            "image_bytes": image_bytes,
            "ocr_text": "",
            "final_query": "",
            "messages": [HM(content=user_text)],
        }
        response = chatbot.invoke(state, config=config)
        return response["messages"][-1].content
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# ── Session state init ──────────────────────────────────────────────────────────
if "active_thread" not in st.session_state:
    st.session_state.active_thread = None

if "chat_history" not in st.session_state:
    # local mirror so we don't re-query checkpointer on every render
    st.session_state.chat_history = {}   # thread_id → list of {role, content, image?}

if "threads" not in st.session_state:
    st.session_state.threads = retrieve_all_threads()

if "mock_threads" not in st.session_state:
    st.session_state.mock_threads = []

if "mock_history" not in st.session_state:
    st.session_state.mock_history = {}


def new_chat():
    tid = f"th-{uuid.uuid4().hex[:8]}"
    st.session_state.active_thread = tid
    st.session_state.chat_history[tid] = []
    if tid not in st.session_state.threads:
        st.session_state.threads.insert(0, tid)
    # mock support
    st.session_state.mock_threads.insert(0, tid)
    st.session_state.mock_history[tid] = []


def switch_thread(tid: str):
    st.session_state.active_thread = tid
    if tid not in st.session_state.chat_history:
        # load from checkpointer
        st.session_state.chat_history[tid] = get_thread_history(tid)


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <p class="sidebar-title">🤖 Nexus</p>
        <p class="sidebar-sub">AI Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋  New Chat", key="new_chat_btn"):
        new_chat()

    st.markdown('<p class="section-label">Recent Threads</p>', unsafe_allow_html=True)

    threads = st.session_state.threads or retrieve_all_threads()

    if not threads:
        st.markdown(
            '<p style="font-size:11px;color:#333355;padding:0 20px;">No threads yet.<br>Start a new chat!</p>',
            unsafe_allow_html=True,
        )
    else:
        for tid in threads:
            is_active = tid == st.session_state.active_thread
            css_class = "thread-item active" if is_active else "thread-item"

            # derive a display label from local history
            history = st.session_state.chat_history.get(tid, [])
            if history:
                first_user = next((m["content"] for m in history if m["role"] == "user"), tid)
                label = first_user[:30] + ("…" if len(first_user) > 30 else "")
            else:
                label = tid

            col1, col2 = st.columns([8, 2])
            with col1:
                st.markdown(f"""
                <div class="{css_class}">
                    <span class="thread-icon">💬</span>
                    <div>
                        <div class="thread-label">{label}</div>
                        <div class="thread-id-small">{tid}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("→", key=f"switch_{tid}", help=f"Open {tid}"):
                    switch_thread(tid)


# ── MAIN AREA ──────────────────────────────────────────────────────────────────
# ── MAIN AREA ──────────────────────────────────────────────────────────────────
active = st.session_state.active_thread

if active is None:
    # Welcome screen
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">✦</div>
        <div class="welcome-heading">Hello, Explorer.</div>
        <div class="welcome-sub">
            Ask anything — I can chat, reason over documents, read images, and remember our conversation.
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat header ──
    st.markdown(f"""
    <div class="chat-title">Chat</div>
    <span class="chat-thread-tag">thread · {active}</span>
    """, unsafe_allow_html=True)

    # ── Messages display ──
    history = st.session_state.chat_history.get(active, [])

    messages_placeholder = st.container()

    with messages_placeholder:

        if not history:
            st.markdown(
                '''
                <p style="
                    color:#333355;
                    font-size:12px;
                    text-align:center;
                    padding-top:40px;
                ">
                    Send a message to start the conversation.
                </p>
                ''',
                unsafe_allow_html=True,
            )

        else:
            for msg in history:

                role = msg["role"]
                content = msg["content"]

                label = "YOU" if role == "user" else "NEXUS"
                css = role

                st.markdown(
                    f'<div class="msg-label {css}">{label}</div>',
                    unsafe_allow_html=True
                )

                col_left, col_right = (
                    st.columns([3, 7])
                    if role == "user"
                    else st.columns([7, 3])
                )

                target_col = (
                    col_right
                    if role == "user"
                    else col_left
                )

                with target_col:

                    # Show image if attached
                    if msg.get("image") is not None:
                        st.image(msg["image"], width=220)

                    st.markdown(
                        f'''
                        <div class="msg-bubble {css}">
                            {content}
                        </div>
                        ''',
                        unsafe_allow_html=True,
                    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input area ──
    with st.container():

        st.markdown(
            '<div class="input-wrapper">',
            unsafe_allow_html=True
        )

        col_text, col_img, col_send = st.columns([7, 2, 1])

        # ─────────────────────────────────────────────
        # TEXT INPUT
        # ─────────────────────────────────────────────
        with col_text:

            input_key = (
                f"input_{active}_"
                f"{st.session_state.get(f'input_reset_{active}', 0)}"
            )

            user_input = st.text_area(
                label="",
                placeholder="Type your message… (Shift+Enter for new line)",
                key=input_key,
                height=68,
                label_visibility="collapsed",
            )

        # ─────────────────────────────────────────────
        # FILE UPLOADER
        # ─────────────────────────────────────────────
        with col_img:

            uploader_key = (
                f"upload_{active}_"
                f"{st.session_state.get(f'uploader_reset_{active}', 0)}"
            )

            uploaded_file = st.file_uploader(
                label="",
                type=["png", "jpg", "jpeg", "webp", "gif"],
                key=uploader_key,
                label_visibility="collapsed",
                help="Attach an image",
            )

        # ─────────────────────────────────────────────
        # SEND BUTTON
        # ─────────────────────────────────────────────
        with col_send:

            send_clicked = st.button(
                "➤",
                key=f"send_{active}",
                help="Send"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle send ─────────────────────────────────────────────
    if send_clicked and user_input and user_input.strip():

        image_bytes = None
        img_data = None

        # Read uploaded image
        if uploaded_file is not None:

            image_bytes = uploaded_file.read()
            img_data = image_bytes

        # ─────────────────────────────────────────────
        # STORE USER MESSAGE
        # ─────────────────────────────────────────────
        user_msg = {
            "role": "user",
            "content": user_input.strip()
        }

        if img_data:
            user_msg["image"] = img_data

        st.session_state.chat_history.setdefault(
            active,
            []
        ).append(user_msg)

        # ─────────────────────────────────────────────
        # BACKEND INVOCATION
        # ─────────────────────────────────────────────
        with st.spinner("Thinking…"):

            reply = invoke_chatbot(
                active,
                user_input.strip(),
                image_bytes
            )

        # ─────────────────────────────────────────────
        # STORE AI MESSAGE
        # ─────────────────────────────────────────────
        st.session_state.chat_history[active].append({
            "role": "ai",
            "content": reply
        })

        # Update mock history
        st.session_state.mock_history[active] = (
            st.session_state.chat_history[active]
        )

        # ─────────────────────────────────────────────
        # RESET TEXTAREA
        # ─────────────────────────────────────────────
        st.session_state[f"input_reset_{active}"] = (
            st.session_state.get(
                f"input_reset_{active}",
                0
            ) + 1
        )

        # ─────────────────────────────────────────────
        # RESET FILE UPLOADER
        # ─────────────────────────────────────────────
        st.session_state[f"uploader_reset_{active}"] = (
            st.session_state.get(
                f"uploader_reset_{active}",
                0
            ) + 1
        )

        # ─────────────────────────────────────────────
        # RERUN
        # ─────────────────────────────────────────────
        st.rerun()

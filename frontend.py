"""
Streamlit Frontend Interface for Mirai Autonomous Student Policy Advisor.
Communicates with the FastAPI backend to deliver fact-grounded policy guidance.
"""

import os
import requests
import streamlit as st

# Configure page configuration
st.set_page_config(
    page_title="Mirai - MSOT Student Policy Advisor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configurable backend URL with default fallback
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")

# Custom UI Styling
st.markdown(
    """
    <style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .sub-header {
        color: #475569;
        font-size: 1.05rem;
        margin-bottom: 1.2rem;
    }
    .disclaimer-box {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 20px;
        font-size: 0.85rem;
        color: #334155;
    }
    .page-tag {
        background-color: #EEF2FF;
        color: #4338CA;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
        border: 1px solid #C7D2FE;
    }
    .badge-grounded {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-insufficient {
        background-color: #FEECDC;
        color: #9C4221;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_backend_health():
    """Queries backend /health diagnostic endpoint safely."""
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=3.0)
        if resp.status_code == 200:
            return resp.json(), None
        return None, f"Health check returned status {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return None, f"Cannot connect to backend at {BACKEND_URL}. Ensure backend.py is running."
    except requests.exceptions.Timeout:
        return None, "Health check timed out."
    except Exception as exc:
        return None, f"Diagnostic error: {str(exc)}"


def send_policy_query(question: str):
    """
    Submits student question to backend /ask endpoint.
    Handles network timeouts, 503 unconfigured states, and 422 validations.
    """
    cleaned_question = question.strip()
    if not cleaned_question:
        return None, "Please enter a valid policy question."

    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": cleaned_question},
            timeout=65.0,
        )

        if resp.status_code == 200:
            return resp.json(), None

        # Handle specific error status codes
        try:
            error_detail = resp.json().get("detail", resp.text)
        except Exception:
            error_detail = resp.text

        if resp.status_code == 503:
            return None, (
                f"⚠️ Configuration / Service Error (503): {error_detail}\n\n"
                "Check backend `.env` configuration for `GEMINI_API_KEY` and ensure handbook is ingested."
            )
        elif resp.status_code == 504:
            return None, "⏳ Timeout Error (504): The policy advisor took too long to process your request."
        elif resp.status_code == 422:
            return None, f"Invalid Input (422): {error_detail}"
        else:
            return None, f"Server Error ({resp.status_code}): {error_detail}"

    except requests.exceptions.Timeout:
        return None, "⏳ Request Timed Out: Backend server did not respond within 35 seconds."
    except requests.exceptions.ConnectionError:
        return None, (
            f"🔴 Connection Failed: Unable to reach Mirai Backend at `{BACKEND_URL}`.\n\n"
            "Please confirm that `backend.py` is running (e.g. `uvicorn backend:app --port 8000`)."
        )
    except Exception as e:
        return None, f"Unexpected Client Error: {str(e)}"


# --- SIDEBAR: Controls, Health Status & Grounded Examples ---
with st.sidebar:
    st.image(
        "https://api.iconify.design/lucide:graduation-cap.svg?color=%233b82f6",
        width=46,
    )
    st.title("Mirai Advisor")
    st.caption("Official MSOT Policy Consultant")

    st.markdown("---")
    st.subheader("Backend Status")
    health, health_err = fetch_backend_health()

    if health:
        status_label = health.get("status", "unknown").upper()
        if status_label == "OK":
            st.success(f"🟢 **Status:** `ONLINE (OK)`")
        else:
            st.warning(f"🟡 **Status:** `{status_label}` (Degraded)")

        st.markdown(f"**Target Host:** `{BACKEND_URL}`")
        st.markdown(f"**Vector Store:** `{'Ready ✅' if health.get('vector_db_ready') else 'Unindexed ⚠️'}`")
        st.markdown(f"**Indexed Vectors:** `{health.get('vector_db_count', 0)}`")
        st.markdown(f"**API Key:** `{'Configured ✅' if health.get('api_key_configured') else 'Missing ⚠️'}`")
        st.markdown(f"**LLM Model:** `{health.get('model', 'gemini-3.8-flash')}`")
    else:
        st.error("🔴 **Backend Offline**")
        st.caption(health_err or f"Cannot reach {BACKEND_URL}")

    st.markdown("---")
    st.subheader("Conversation Control")
    if st.button("🗑️ Clear / Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.active_prompt = ""
        st.rerun()

    st.markdown("---")
    st.subheader("Handbook Examples")
    sample_queries = [
        ("📊 72% Attendance Marks", "I have 72% attendance. How many attendance marks will I get?"),
        ("🏥 Ratnam Medical Leave", "I study at the Ratnam campus. I got sick and need medical leave. Who do I email and how many days do I have to submit my documents?"),
        ("🛡️ Proposing Tech Society", "We want to start a new Cybersecurity society under the Tech Club. Do we ask Management directly?"),
        ("🚭 Smoking Policy / Fine", "How much is the fine for smoking a cigarette on campus?"),
        ("📅 7-Day Leave Rule", "What is the 7-day submission rule for medical leave and what happens if I miss the deadline?"),
        ("🚫 Debarment Rule", "What is the minimum attendance threshold below which a student is debarred from final exams?"),
    ]

    selected_sample = None
    for label, query in sample_queries:
        if st.button(label, help=query, use_container_width=True, key=f"smp_{hash(label)}"):
            selected_sample = query


# --- MAIN INTERFACE ---
st.markdown("<h1 class='main-header'>Mirai Student Policy Advisor</h1>", unsafe_allow_html=True)
st.markdown(
    "<p class='sub-header'>Autonomous, strictly source-grounded policy assistant for "
    "<strong>Mirai School of Technology</strong>. Answers are derived solely from the official 2026 Policy Handbook.</p>",
    unsafe_allow_html=True,
)

# Prominent AI Disclaimer
st.markdown(
    """
    <div class='disclaimer-box'>
        ⚠️ <strong>Official Disclaimer:</strong> Mirai is an automated AI policy advisor. All responses are synthesized directly from the
        <em>Mirai SoT Policy Handbook 2026</em>. For binding academic decisions, official petitions, or disciplinary matters,
        please consult the cited handbook pages or your designated Campus Manager.
    </div>
    """,
    unsafe_allow_html=True,
)

# Initialize conversation history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! I am Mirai, your institutional policy advisor. "
                "Ask me any question regarding attendance grading tiers, medical and duty leave procedures, "
                "club proposals, examination rules, or campus conduct."
            ),
            "grounded": True,
            "page_numbers": [],
            "citations": [],
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display cited pages and grounding status
        if msg.get("page_numbers"):
            tags = " ".join([f"<span class='page-tag'>Page {p}</span>" for p in msg["page_numbers"]])
            badge = "<span class='badge-grounded'>Handbook Grounded ✅</span>" if msg.get("grounded") else "<span class='badge-insufficient'>Insufficient Evidence ⚠️</span>"
            st.markdown(f"<div style='margin-top: 8px;'>{badge} &nbsp; <strong>Cited Pages:</strong> {tags}</div>", unsafe_allow_html=True)

        # Expandable source evidence excerpts
        if msg.get("citations"):
            with st.expander("🔍 View Policy Handbook Excerpts"):
                for idx, cit in enumerate(msg["citations"], start=1):
                    src_name = cit.get("source", "Handbook.pdf")
                    page_num = cit.get("page", "?")
                    st.markdown(f"**Excerpt {idx} — {src_name} (Page {page_num}):**")
                    st.caption(f"\"{cit.get('snippet', '')}\"")


# Question Submission Area (Supporting both Chat Input & Form with Ask Button)
chat_input = st.chat_input("Type your policy inquiry here...")
active_query = selected_sample or chat_input

if active_query:
    # Append student message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Process response with spinner
    with st.chat_message("assistant"):
        with st.spinner("Analyzing handbook policies and retrieving evidence..."):
            response_data, error_msg = send_policy_query(active_query)

            if error_msg:
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ {error_msg}",
                    "grounded": False,
                    "page_numbers": [],
                    "citations": [],
                })
            else:
                answer = response_data.get("answer", "")
                page_numbers = response_data.get("page_numbers", [])
                citations = response_data.get("citations", [])
                grounded = response_data.get("grounded", True)
                insufficient = response_data.get("insufficient_evidence", False)

                st.markdown(answer)

                if page_numbers:
                    tags = " ".join([f"<span class='page-tag'>Page {p}</span>" for p in page_numbers])
                    badge = "<span class='badge-grounded'>Handbook Grounded ✅</span>" if grounded else "<span class='badge-insufficient'>Insufficient Evidence ⚠️</span>"
                    st.markdown(f"<div style='margin-top: 8px;'>{badge} &nbsp; <strong>Cited Pages:</strong> {tags}</div>", unsafe_allow_html=True)

                if citations:
                    with st.expander("🔍 View Policy Handbook Excerpts"):
                        for idx, cit in enumerate(citations, start=1):
                            src_name = cit.get("source", "Handbook.pdf")
                            page_num = cit.get("page", "?")
                            st.markdown(f"**Excerpt {idx} — {src_name} (Page {page_num}):**")
                            st.caption(f"\"{cit.get('snippet', '')}\"")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "grounded": grounded,
                    "page_numbers": page_numbers,
                    "citations": citations,
                })

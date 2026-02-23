"""
app.py — Streamlit frontend for Smart Document Q&A

PAGES:
  1. Login / Register — auth forms, stores JWT in session_state
  2. My Documents — upload, list, delete documents
  3. Ask Questions — Q&A interface with source citations

HOW STREAMLIT SESSION STATE WORKS (explain to interviewer):
  - st.session_state persists values across reruns (like a React state)
  - We store the JWT token in session_state["token"]
  - Every API call sends this token in Authorization header
  - No token = redirect to login page
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# --- Configuration ---
# On Streamlit Cloud: set BACKEND_URL in app Secrets
# Locally: set in .env or leave as default
API_BASE = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="Smart Doc Q&A",
    page_icon="📄",
    layout="wide"
)


# --- API Helper Functions ---

def api_post(endpoint: str, data: dict, token: str = None) -> tuple:
    """Make a POST request to the API. Returns (response_data, error_message)"""
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=data, headers=headers, timeout=30)
        if resp.status_code in (200, 201):
            return resp.json(), None
        # Try to parse error detail from JSON, fallback to status code
        try:
            error_detail = resp.json().get("detail", f"HTTP {resp.status_code} error")
        except Exception:
            error_detail = f"HTTP {resp.status_code} — {resp.text[:200] if resp.text else 'empty response'}"
        return None, error_detail
    except requests.exceptions.ConnectionError:
        return None, "Backend is not running. Start it with: uvicorn main:app --reload --port 8000"
    except Exception as e:
        return None, str(e)


def api_get(endpoint: str, token: str) -> tuple:
    """Make a GET request to the API."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", headers=headers, timeout=30)
        if resp.status_code == 200:
            return resp.json(), None
        return None, resp.json().get("detail", "Request failed")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as e:
        return None, str(e)


def api_delete(endpoint: str, token: str) -> tuple:
    """Make a DELETE request to the API."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.delete(f"{API_BASE}{endpoint}", headers=headers, timeout=30)
        if resp.status_code == 204:
            return True, None
        return False, resp.json().get("detail", "Delete failed")
    except Exception as e:
        return False, str(e)


def api_upload(file_bytes: bytes, filename: str, token: str) -> tuple:
    """Upload a file to the API."""
    headers = {"Authorization": f"Bearer {token}"}
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        resp = requests.post(f"{API_BASE}/documents/upload", files=files, headers=headers, timeout=60)
        if resp.status_code == 201:
            return resp.json(), None
        return None, resp.json().get("detail", "Upload failed")
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as e:
        return None, str(e)


# --- Auth Page ---

def show_auth_page():
    st.title("Smart Document Q&A")
    st.markdown("*Upload documents and ask questions using AI*")
    st.divider()

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        st.subheader("Login to your account")
        with st.form("login_form"):
            email = st.text_input("Email", placeholder="you@example.com")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if not email or not password:
                st.error("Please fill in all fields")
            else:
                data, error = api_post("/auth/login", {"email": email, "password": password})
                if error:
                    st.error(f"Login failed: {error}")
                else:
                    st.session_state["token"] = data["access_token"]
                    st.session_state["email"] = email
                    st.success("Logged in successfully!")
                    st.rerun()

    with tab2:
        st.subheader("Create a new account")
        with st.form("register_form"):
            reg_email = st.text_input("Email", placeholder="you@example.com", key="reg_email")
            reg_password = st.text_input("Password", type="password", key="reg_pass")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            reg_submitted = st.form_submit_button("Register", use_container_width=True)

        if reg_submitted:
            if not reg_email or not reg_password:
                st.error("Please fill in all fields")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                data, error = api_post("/auth/register", {"email": reg_email, "password": reg_password})
                if error:
                    st.error(f"Registration failed: {error}")
                else:
                    st.success("Account created! Please login.")


# --- Documents Page ---

def show_documents_page():
    st.header("My Documents")
    token = st.session_state["token"]

    # Upload section
    st.subheader("Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"],
        help="Maximum file size: 10MB"
    )

    if uploaded_file:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"Selected: **{uploaded_file.name}** ({len(uploaded_file.getvalue()) / 1024:.1f} KB)")
        with col2:
            if st.button("Process & Upload", use_container_width=True, type="primary"):
                with st.spinner(f"Processing {uploaded_file.name}... This may take a moment."):
                    data, error = api_upload(uploaded_file.getvalue(), uploaded_file.name, token)
                if error:
                    st.error(f"Upload failed: {error}")
                else:
                    st.success(f"Successfully uploaded and processed **{uploaded_file.name}**!")
                    st.rerun()

    st.divider()

    # Documents list
    st.subheader("Your Documents")
    docs, error = api_get("/documents/list", token)

    if error:
        st.error(f"Could not load documents: {error}")
        return

    if not docs:
        st.info("No documents uploaded yet. Upload a PDF or TXT file above to get started.")
        return

    for doc in docs:
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            icon = "📄" if doc["filename"].endswith(".pdf") else "📝"
            st.write(f"{icon} **{doc['filename']}**")
        with col2:
            uploaded = datetime.fromisoformat(doc["uploaded_at"].replace("Z", ""))
            st.write(f"Uploaded: {uploaded.strftime('%b %d, %Y')}")
        with col3:
            if st.button("Delete", key=f"del_{doc['id']}", type="secondary"):
                success, err = api_delete(f"/documents/{doc['id']}", token)
                if err:
                    st.error(f"Delete failed: {err}")
                else:
                    st.success("Document deleted")
                    st.rerun()


# --- Q&A Page ---

def show_qa_page():
    st.header("Ask Questions")
    token = st.session_state["token"]

    # Load user's documents for optional filtering
    docs, _ = api_get("/documents/list", token)

    if not docs:
        st.warning("You need to upload documents first before asking questions.")
        st.info("Go to the **My Documents** tab to upload PDFs or text files.")
        return

    # Document filter (optional)
    doc_options = {"All Documents": None}
    for doc in docs:
        doc_options[doc["filename"]] = doc["id"]

    selected_doc_name = st.selectbox(
        "Search in:",
        options=list(doc_options.keys()),
        help="Search across all documents, or select a specific one"
    )
    selected_doc_id = doc_options[selected_doc_name]

    st.divider()

    # Question input
    question = st.text_input(
        "Your question",
        placeholder="What is the main topic of the document?",
        help="Ask any question about your uploaded documents"
    )

    if st.button("Get Answer", type="primary", use_container_width=False):
        if not question.strip():
            st.warning("Please enter a question")
            return

        with st.spinner("Searching documents and generating answer..."):
            payload = {"question": question}
            if selected_doc_id:
                payload["document_id"] = selected_doc_id

            data, error = api_post("/qa/ask", payload, token)

        if error:
            st.error(f"Error: {error}")
            return

        # Display answer
        st.subheader("Answer")
        st.markdown(data["answer"])

        # Display sources
        if data["sources"]:
            st.divider()
            st.subheader(f"Sources ({len(data['sources'])} chunks retrieved)")
            for i, source in enumerate(data["sources"]):
                with st.expander(f"Source {i+1} — {source['filename']} (chunk {source['chunk_index']})"):
                    st.markdown(f"```\n{source['content']}\n```")

        st.caption(f"Searched across {data['documents_searched']} document(s)")


# --- Main App ---

def main():
    # Check authentication
    if "token" not in st.session_state:
        show_auth_page()
        return

    # Sidebar
    with st.sidebar:
        st.title("Smart Doc Q&A")
        st.write(f"Logged in as: **{st.session_state.get('email', 'User')}**")
        st.divider()

        page = st.radio(
            "Navigate",
            ["My Documents", "Ask Questions"],
            label_visibility="collapsed"
        )

        st.divider()
        if st.button("Logout", use_container_width=True):
            del st.session_state["token"]
            del st.session_state["email"]
            st.rerun()

    # Render selected page
    if page == "My Documents":
        show_documents_page()
    elif page == "Ask Questions":
        show_qa_page()


if __name__ == "__main__":
    main()

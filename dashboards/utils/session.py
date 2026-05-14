# utils/session.py

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()

def init_session():
    if "api_url" not in st.session_state:
        st.session_state["api_url"] = os.getenv("MLOPS_API_URL")

    if "model_url" not in st.session_state:
        st.session_state["model_url"] = os.getenv("FRAUD_MODEL_MCP_URL")
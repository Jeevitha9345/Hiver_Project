import os
import sys
sys.path.insert(0, os.path.abspath("."))

import streamlit as st
from src.agent import HiverSupportAgent

st.set_page_config(
    page_title="Hiver Support Agent — AmazonHelp",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Hiver AI Customer Support Agent")
st.markdown("**Domain:** `AmazonHelp` Twitter Customer Support | **Evaluation Benchmark:** 7-Intent Operational Taxonomy")

@st.cache_resource
def load_agent():
    return HiverSupportAgent()

with st.spinner("Loading agent models and FAISS retrieval index..."):
    agent = load_agent()

st.sidebar.header("Sample Customer Inquiries")
sample_options = [
    "Where is my package? The tracking number has not updated in 4 days.",
    "The coffee maker arrived with a shattered glass carafe. How do I get a replacement?",
    "Please cancel my order #114-8829104-99210 immediately.",
    "I was charged $139 for Prime on my credit card without my consent!",
    "Someone hacked into my account and ordered items to another state!",
    "Your representative was extremely rude and hung up on me. Transfer me to a manager right now!",
    "When will the PlayStation 5 console be back in stock for purchase?"
]
selected_sample = st.sidebar.selectbox("Choose a pre-filled scenario:", ["Custom Query"] + sample_options)

default_text = "" if selected_sample == "Custom Query" else selected_sample
user_input = st.text_area("Enter customer support message:", value=default_text, height=100)

col1, col2 = st.columns([1, 5])
with col1:
    submit = st.button("Process Inquiry", type="primary")

if submit and user_input.strip():
    with st.spinner("Analyzing message, retrieving historical resolutions, and applying escalation policy..."):
        result = agent.process_message(user_input.strip())
        
    st.markdown("---")
    
    # 1. Triage Status Banner
    is_escalate = result["decision"] == "ESCALATE"
    if is_escalate:
        st.error(f"🚨 **Action Required:** {result['decision']} (Trigger: `{result['escalation_trigger']}`)")
    else:
        st.success(f"✅ **Automated Triage:** {result['decision']} (Trigger: `{result['escalation_trigger']}`)")
        
    st.info(f"**Decision Reason:** {result['decision_reason']}")
    
    # 2. Main Columns
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("🎯 Intent Classification")
        st.metric("Predicted Intent", result["intent"]["label"])
        st.metric("Classifier Confidence", f"{result['intent']['confidence'] * 100:.1f}%")
        
        with st.expander("Show Class Probability Distribution"):
            st.json(result["intent"]["probabilities"])
            
        st.subheader("💬 Generated Support Response")
        st.markdown(f"> {result['reply']}")
        
        if result["warnings"]:
            st.warning(" | ".join(result["warnings"]))
            
        st.caption(f"Overall Pipeline Confidence: **{result['confidence'] * 100:.0f}%**")

    with c2:
        st.subheader("🔍 Historical Support Retrieval (RAG Evidence)")
        retrieved = result.get("retrieved_examples", [])
        if retrieved:
            for i, ex in enumerate(retrieved, 1):
                with st.expander(f"Match #{i} — Similarity: {ex['score']:.2f} ({ex['conversation_id']})"):
                    st.markdown(f"**Customer Issue:** {ex['historical_customer_issue']}")
                    st.markdown(f"**Historical Brand Reply:** {ex['historical_support_reply']}")
        else:
            st.write("No historical matches found.")

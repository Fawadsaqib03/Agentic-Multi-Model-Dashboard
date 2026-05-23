import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

# Import the executor from agent.py
from agent import executor

# Page Config
st.set_page_config(
    page_title="ML Agent - Multi Model AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium dark theme
st.markdown("""
<style>x
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Strict Dark Mode Background Overrides & Hide Scrollbars */
    ::-webkit-scrollbar { display: none; }
    * { -ms-overflow-style: none; scrollbar-width: none; }
    
    .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: #0f172a !important; /* Slate 900 */
        color: #f8fafc !important; /* Slate 50 */
    }
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0 1.5rem 0;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -0.05em;
        background: linear-gradient(135deg, #a78bfa 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        color: #94a3b8 !important; /* Slate 400 */
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Attractive Chat Bubbles */
    .stChatMessage {
        background-color: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(51, 65, 85, 0.5) !important;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important; /* Slate 800 */
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] * {
        scrollbar-width: none !important;
    }
    section[data-testid="stSidebar"] *::-webkit-scrollbar {
        display: none !important;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 0 !important;
    }
    header[data-testid="stSidebarHeader"] {
        display: none !important;
        padding: 0 !important;
        height: 0 !important;
        min-height: 0 !important;
    }
    
    .model-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(51, 65, 85, 0.8);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
    }
    .model-card:hover {
        border-color: #c084fc;
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    }
    .model-card h4 { margin: 0 0 4px 0; font-size: 0.9rem; font-weight: 600; color: #f8fafc;}
    .model-badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.65rem; font-weight: 700; letter-spacing: 0.5px; margin-bottom: 0;
    }
    .badge-nn { background: rgba(168,85,247,0.2); color: #d8b4fe; border: 1px solid rgba(168,85,247,0.4); }
    .badge-lr { background: rgba(59,130,246,0.2); color: #93c5fd; border: 1px solid rgba(59,130,246,0.4); }
    .badge-svm { background: rgba(34,197,94,0.2); color: #86efac; border: 1px solid rgba(34,197,94,0.4); }
    
    .status-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #4ade80; margin-right: 8px;
        box-shadow: 0 0 8px rgba(74,222,128,0.6);
    }
</style>
""", unsafe_allow_html=True)

import base64
import os

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

logo_path = r"C:\Users\Fawad\OneDrive\Pictures\Saved Pictures\COMSATS-University-logo with out background.png"
logo_b64 = get_base64_image(logo_path)
img_src = f"data:image/png;base64,{logo_b64}" if logo_b64 else "https://upload.wikimedia.org/wikipedia/en/thumb/8/87/COMSATS_University_Islamabad_logo.png/300px-COMSATS_University_Islamabad_logo.png"

# Sidebar
with st.sidebar:
    st.markdown(f"""
    <div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin-top: -30px; margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #334155;">
        <img src="{img_src}" style="height: 110px; margin-bottom: 12px; background: transparent; border-radius: 0%; padding: 5px; filter: drop-shadow(0 4px 10px rgba(0,0,0,0.3));">
        <h3 style="margin: 0; font-size: 1rem; font-weight: 700; color: #f8fafc; line-height: 1.3;">COMSATS University Islamabad</h3>
        <p style="margin: 8px 0 0 0; font-size: 0.75rem; color: #c084fc; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">Final Year Project 2026 of Machine Learning</p>
        <div style="margin-top: 15px; font-size: 0.85rem; color: #f8fafc; font-weight: 500; line-height: 1.6; background: rgba(30, 41, 59, 0.5); padding: 10px; border-radius: 8px; width: 100%; border: 1px solid rgba(51, 65, 85, 0.5);">
            <div style="color: #f472b6; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: 700;">👨‍🏫 Supervised By</div>
            <div style="margin-bottom: 12px; color: #f8fafc;">Sir Shah Zain</div>
            <div style="color: #c084fc; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; font-weight: 700;">👥 Project Team</div>
            <div>Fawad Saqib</div>
            <div>Awais Manzoor</div>
            <div>Muhammad Haisam</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Connected Worker Models**")
    
    st.markdown("""
    <div class="model-card" style="padding: 10px 12px; display: flex; justify-content: space-between; align-items: center;">
        <h4 style="margin:0; font-size: 0.85rem;"><span class="status-dot"></span>Bank Customer Churn</h4>
        <span class="model-badge badge-nn" style="margin:0;">NN</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card" style="padding: 10px 12px; display: flex; justify-content: space-between; align-items: center;">
        <h4 style="margin:0; font-size: 0.85rem;"><span class="status-dot"></span>Diabetes Prediction</h4>
        <span class="model-badge badge-lr" style="margin:0;">LR</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card" style="padding: 10px 12px; display: flex; justify-content: space-between; align-items: center;">
        <h4 style="margin:0; font-size: 0.85rem;"><span class="status-dot"></span>SMS Spam Detection</h4>
        <span class="model-badge badge-svm" style="margin:0;">SVM</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption("3 Models | 1 Agent | Powered by Groq")

# Main Header
st.markdown("""
<div style="background: linear-gradient(145deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.9)); border: 1px solid rgba(167, 139, 250, 0.2); border-radius: 24px; padding: 3rem 2rem; margin-bottom: 2.5rem; box-shadow: 0 20px 40px -10px rgba(0,0,0,0.5); backdrop-filter: blur(12px); text-align: center; position: relative; overflow: hidden;">
    <div style="position: absolute; top: -50px; left: -50px; width: 100px; height: 100px; background: rgba(167, 139, 250, 0.2); border-radius: 50%; filter: blur(40px);"></div>
    <div style="position: absolute; bottom: -50px; right: -50px; width: 150px; height: 150px; background: rgba(244, 114, 182, 0.15); border-radius: 50%; filter: blur(50px);"></div>
    <h1 style="font-size: 3.2rem; font-weight: 800; letter-spacing: -0.05em; margin-bottom: 0.8rem; background: linear-gradient(135deg, #a78bfa 0%, #c084fc 50%, #f472b6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; position: relative; z-index: 1;">ML Agent Assistant</h1>
    <p style="color: #e2e8f0; font-size: 1.1rem; font-weight: 300; max-width: 650px; margin: 0 auto; line-height: 1.6; position: relative; z-index: 1;">Just describe your problem naturally — I'll automatically pick the right model, handle missing data, and provide professional analysis.</p>
</div>
""", unsafe_allow_html=True)

# Initialize Conversational Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Render previous chat messages
for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# Show suggestion chips when chat is empty and no input is pending
if not st.session_state.chat_history and not st.session_state.get("pending_input"):
    st.markdown("#### Try asking:")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Try Customer Would Leave", use_container_width=True, key="sug1"):
            st.session_state.pending_input = "Will a 42 year old male customer from Germany with credit score 650 and balance 75000 churn?"
            st.rerun()
    with col2:
        if st.button("Check Diabetes Risk", use_container_width=True, key="sug2"):
            st.session_state.pending_input = "Check diabetes risk for a patient with glucose level 148, BMI 33.6, and age 50"
            st.rerun()
    with col3:
        if st.button("Is This Spam?", use_container_width=True, key="sug3"):
            st.session_state.pending_input = "Is this message spam? 'Congratulations! You have won a free iPhone 15. Click here to claim your prize now!'"
            st.rerun()
    
    

# Handle pending input from suggestion chips
pending = st.session_state.pop("pending_input", None)

# Handle New User Input
user_input = st.chat_input("Describe your problem naturally -- e.g. 'Will this customer churn?' or 'Is this spam?'")

if pending and not user_input:
    user_input = pending

if user_input:
    # Save and display user message
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    # Process and display agent response
    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing your request and selecting the right model..."):
            try:
                response = executor.invoke({
                    "input": user_input,
                    "chat_history": st.session_state.chat_history
                })
                answer = response["output"]
                
                # Sanitize answer to remove markdown code blocks and inline code that cause green text
                import re
                answer = re.sub(r'```[a-zA-Z]*\n?', '', answer)
                answer = answer.replace('```', '')
                answer = answer.replace('`', '') # Strip inline code backticks which cause green text
                
                # Also remove Streamlit color syntax (e.g., :green[text])
                answer = re.sub(r':[a-z]+\[(.*?)\]', r'\1', answer)
                
                st.markdown(answer)
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(AIMessage(content=error_msg))
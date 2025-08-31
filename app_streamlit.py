# app_streamlit.py
import joblib
import json
import os
import streamlit as st
import plotly.graph_objects as go

# ----------------- Load Artifacts -----------------
MODEL_PATH = "phishing_model.pkl"
VECT_PATH = "vectorizer.pkl"
LE_PATH = "label_encoder.pkl"
HUMAN_MAP_PATH = "human_label_map.json"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECT_PATH)

try:
    with open(HUMAN_MAP_PATH, "r") as f:
        human_map = json.load(f)
        human_map = {int(k): v for k, v in human_map.items()}
except Exception:
    human_map = {1: "Phishing", 0: "Safe"}

# ----------------- Page Setup -----------------
st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="✉️",
    layout="wide"
)

# ----------------- Sidebar Navigation -----------------
st.sidebar.title("🔎 Navigation")
menu = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📩 Detector", "ℹ️ About", "🛠 Admin Controls"]
)

# ----------------- Custom Styling -----------------
st.markdown(
    """
    <style>
        .main-title {font-size:32px; font-weight:bold; color:#2E86C1;}
        .sub-title {font-size:20px; font-weight:600; color:#1ABC9C;}
        .email-box {
            background-color:#808080; padding:12px; border-radius:8px;
            font-family:Consolas, monospace; font-size:14px; line-height:1.4;
            white-space:pre-wrap; word-wrap:break-word;
            border:1px solid #ddd;
        }
        .footer {text-align:center; font-size:12px; color:gray; margin-top:40px;}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- Home Page -----------------
if menu == "🏠 Home":
    st.markdown('<p class="main-title">✉️ Phishing Email Detector</p>', unsafe_allow_html=True)
    st.caption("Smart AI tool to analyze emails and detect phishing attempts instantly.")

    st.markdown("### ✅ Features")
    st.markdown("""
    - Paste or upload email text to analyze  
    - Get instant result with **confidence score**  
    - Visual phishing risk gauge  
    - Easy-to-use and secure  
    """)

    st.success("➡️ Go to **📩 Detector** from the sidebar to test emails.")

# ----------------- Detector Page -----------------
elif menu == "📩 Detector":
    st.markdown('<p class="main-title">📩 Email Detector</p>', unsafe_allow_html=True)

    with st.expander("Provide Email Content", expanded=True):
        input_method = st.radio("Input method:", ("Paste Email Text", "Upload .txt File"), horizontal=True)
        email_text = ""

        if input_method == "Paste Email Text":
            email_text = st.text_area("Paste the email content below:", height=200)
        else:
            uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])
            if uploaded_file is not None:
                try:
                    email_text = uploaded_file.read().decode("utf-8")
                except Exception:
                    email_text = uploaded_file.read().decode("utf-8", errors="ignore")

    # ----------------- Prediction Section -----------------
    if st.button("🔍 Analyze Email"):
        if not email_text.strip():
            st.warning("⚠️ Please provide email content before analysis.")
        else:
            X = vectorizer.transform([email_text])
            pred_encoded = model.predict(X)[0]
            friendly = human_map.get(int(pred_encoded), str(pred_encoded))

            prob_val = 0
            prob_text = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)[0]
                pred_index = list(model.classes_).index(pred_encoded)
                prob_val = probs[pred_index]
                prob_text = f"Confidence: {prob_val*100:.2f}%"

            # ----------- Professional Output Layout -----------
            left, right = st.columns([2, 1])

            with left:
                if friendly.lower().startswith("phish"):
                    st.error(f"🚨 Result: PHISHING detected!\n\n{prob_text}")
                else:
                    st.success(f"✅ Result: This email looks SAFE.\n\n{prob_text}")

                st.subheader("📜 Email Preview")
                preview_text = email_text[:1000] + ("..." if len(email_text) > 1000 else "")
                st.markdown(f'<div class="email-box">{preview_text}</div>', unsafe_allow_html=True)

            with right:
                st.subheader("📊 Risk Gauge")
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob_val * 100,
                    title={'text': "Phishing Risk (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if friendly.lower().startswith("phish") else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "#00C853"},
                            {'range': [40, 70], 'color': "#FFAB00"},
                            {'range': [70, 100], 'color': "#D50000"}
                        ]
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)

# ----------------- About Page -----------------
elif menu == "ℹ️ About":
    st.markdown('<p class="main-title">ℹ️ About This Project</p>', unsafe_allow_html=True)
    st.info("""
    This project uses **Machine Learning** with Logistic Regression to detect phishing emails.  
    It works by analyzing the email content, extracting patterns, and predicting if it's **Safe or Phishing**.  
    """)

    st.markdown("### 🔮 Future Enhancements")
    st.markdown("""
    - Support for **attachments** scanning  
    - Real-time browser extension  
    - Multi-language phishing detection  
    - Email sender domain verification  
    """)

# ----------------- Admin Controls -----------------
elif menu == "🛠 Admin Controls":
    st.markdown('<p class="main-title">🛠 Admin Controls</p>', unsafe_allow_html=True)

    ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")  # fallback password
    pwd = st.text_input("Enter Admin Password:", type="password")

    if pwd == ADMIN_PASSWORD:
        st.success("✅ Access Granted!")
        st.write("Here you can add advanced logs, retrain the model, or monitor usage.")
    elif pwd:
        st.error("❌ Incorrect Password!")
        


# ----------------- Footer -----------------
st.markdown('<div class="footer">Made with Python using Streamlit & ML | 2025</div>', unsafe_allow_html=True)

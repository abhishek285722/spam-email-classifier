"""Spam Email / SMS Classifier - Streamlit Web Application.

Features:
- Real-time classification with dual confidence meters
- One-click demo presets for video walkthroughs
- Explainable AI (XAI) word-level highlight & token weights
- Batch CSV file processing and download
- Model performance metrics and confusion matrix inspection
"""
import html
import io
import json
import os

import pandas as pd
import streamlit as st

from data import clean_text
from predict import load_model, predict_message

# Page setup
st.set_page_config(
    page_title="AI Spam Email & SMS Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for a modern visual design
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: #ffffff;
        padding: 24px 30px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff !important;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #e0e7ff;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }

    .result-card-spam {
        background: linear-gradient(135deg, #fff5f5 0%, #fed7d7 100%);
        border: 2px solid #e53e3e;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        color: #742a2a;
    }
    .result-card-ham {
        background: linear-gradient(135deg, #f0fff4 0%, #c6f6d5 100%);
        border: 2px solid #38a169;
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        color: #22543d;
    }
    .token-spam {
        background-color: #feb2b2;
        color: #9b2c2c;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .token-ham {
        background-color: #9ae6b4;
        color: #22543d;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        margin: 2px;
        display: inline-block;
    }
    .preset-btn {
        margin-bottom: 5px;
    }
    /* Suppress Streamlit disconnect popup so clean shutdown is seamless */
    [data-testid="stConnectionStatus"],
    .stConnectionStatus,
    div[role="dialog"]:has([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def get_cached_pipeline():
    """Load and cache the trained sklearn pipeline for reuse across Streamlit reruns."""
    return load_model()


@st.cache_data
def get_metadata():
    """Load and cache model training metadata (accuracy, confusion matrix, keywords)."""
    meta_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


model = get_cached_pipeline()
metadata = get_metadata()

# Header banner
st.markdown(
    """
<div class="main-header">
    <h1>📧 AI Spam Email & SMS Classifier</h1>
    <p>Powered by scikit-learn TF-IDF Vectorization & Multinomial Naive Bayes</p>
</div>
""",
    unsafe_allow_html=True,
)

# Sidebar with Architecture & Project Info
with st.sidebar:
    st.markdown("<div style='font-size: 5rem; text-align: center;'>📧</div>", unsafe_allow_html=True)
    st.subheader("Model Architecture")
    st.markdown(
        """
    - **Vectorization**: TF-IDF (Unigrams & Bigrams, 5000 max features)
    - **Classifier**: Multinomial Naive Bayes (`alpha=0.1`)
    - **Training Data**: 5,574 labeled messages
    - **Framework**: `scikit-learn` & `Streamlit`
    """
    )

    if metadata:
        st.divider()
        st.subheader("Benchmark Accuracy")
        st.metric("Test Accuracy", f"{metadata['accuracy'] * 100:.2f}%")
        st.metric("Dataset Size", f"{metadata['dataset_stats']['total']:,} messages")
        st.caption(f"Last Trained: {metadata.get('trained_at', 'N/A')}")

    st.divider()
    if st.button("🛑 Stop Session", use_container_width=True, type="secondary", help="Stop application and close window"):
        import os
        os._exit(0)

# Tabs
tab_interactive, tab_batch, tab_metrics = st.tabs([
    "🔍 Live Classifier & Explainability",
    "📁 Batch CSV Processing",
    "📊 Model Performance & Insights",
])

# -------------------------------------------------------------
# TAB 1: Live Interactive Classifier
# -------------------------------------------------------------
with tab_interactive:
    st.markdown("### Test Any Email or SMS Message")
    st.write("Type or paste any text below, or choose a pre-loaded sample to test immediately.")

    # Preset buttons for quick LinkedIn demo
    st.markdown("**Quick Preset Demos (Click to Load):**")
    preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

    preset_message = None
    with preset_col1:
        if st.button("🎁 Lottery Winner (Spam)", use_container_width=True):
            st.session_state["msg_input"] = (
                "WINNER!! As a valued customer you have been selected to receive a £1,000 cash prize! "
                "Claim now by calling 09061701461. Valid 24 hours only. T&Cs apply."
            )
            st.session_state["auto_classify"] = True
    with preset_col2:
        if st.button("⚠️ Urgent Bank Alert (Spam)", use_container_width=True):
            st.session_state["msg_input"] = (
                "URGENT: Your mobile account security has been compromised. "
                "Text YES to 87121 immediately to claim your free security update and prize."
            )
            st.session_state["auto_classify"] = True
    with preset_col3:
        if st.button("☕ Lunch Invitation (Ham)", use_container_width=True):
            st.session_state["msg_input"] = (
                "Hey! Are we still meeting for lunch at the cafeteria around 1:30 PM today? Let me know."
            )
            st.session_state["auto_classify"] = True
    with preset_col4:
        if st.button("📅 Project Meeting (Ham)", use_container_width=True):
            st.session_state["msg_input"] = (
                "Hi Team, please find the updated project schedule attached. Let us review the sprint goals tomorrow morning."
            )
            st.session_state["auto_classify"] = True

    # Text Input Area
    default_text = st.session_state.get("msg_input", "")
    user_text = st.text_area(
        "Enter Message Content:",
        value=default_text,
        height=140,
        placeholder="Paste your email or SMS message text here...",
        key="main_text_input",
    )

    # Word & Char counts
    words_count = len(user_text.split()) if user_text.strip() else 0
    chars_count = len(user_text)
    st.caption(f"Length: **{chars_count}** characters | **{words_count}** words")

    classify_clicked = st.button("🚀 Classify Message", type="primary", use_container_width=True)

    should_auto_classify = st.session_state.pop("auto_classify", False)
    if classify_clicked or should_auto_classify:
        if not user_text.strip():
            st.warning("⚠️ Please enter or select a message to classify.")
        else:
            with st.spinner("Analyzing message with TF-IDF + Naive Bayes..."):
                res = predict_message(user_text, model=model)

            # Results Section
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                if res["label"] == "spam":
                    st.markdown(
                        f"""
                    <div class="result-card-spam">
                        <h2 style="margin:0; color:#e53e3e;">🚨 SPAM DETECTED</h2>
                        <p style="margin:8px 0 0 0; font-size:1.1rem;">
                            The model classified this message as <strong>SPAM</strong> with 
                            <strong>{res['spam_prob'] * 100:.2f}%</strong> confidence.
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                    <div class="result-card-ham">
                        <h2 style="margin:0; color:#38a169;">✅ LEGITIMATE (HAM)</h2>
                        <p style="margin:8px 0 0 0; font-size:1.1rem;">
                            The model classified this message as <strong>LEGITIMATE (HAM)</strong> with 
                            <strong>{res['ham_prob'] * 100:.2f}%</strong> confidence.
                        </p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

            with res_col2:
                st.markdown("#### Probability Breakdown")
                st.write(f"**Spam Probability:** `{res['spam_prob'] * 100:.2f}%`")
                st.progress(res["spam_prob"])
                st.write(f"**Legitimate (Ham) Probability:** `{res['ham_prob'] * 100:.2f}%`")
                st.progress(res["ham_prob"])

            # Explainability / Contributing Words
            st.divider()
            st.markdown("#### 🧠 Explainable AI: Influential Word Breakdown")
            st.write(
                "Words identified in the message that most heavily influenced the classifier:"
            )

            if res["tokens"]:
                token_spans = []
                for t in res["tokens"]:
                    token_class = "token-spam" if t["lean"] == "spam" else "token-ham"
                    sign = "+" if t["score"] > 0 else ""
                    token_spans.append(
                        f'<span class="{token_class}">{html.escape(t["word"])} ({sign}{t["score"]:.1f})</span>'
                    )
                st.markdown(" ".join(token_spans), unsafe_allow_html=True)
                st.caption(
                    "Red badges denote spam-indicative keywords; green badges denote ham-indicative keywords."
                )
            else:
                st.info("No strongly weighted keywords from the model vocabulary were found in this text.")

# -------------------------------------------------------------
# TAB 2: Batch CSV Processing
# -------------------------------------------------------------
with tab_batch:
    st.markdown("### 📁 Batch Classify Multiple Messages via CSV")
    st.write("Upload a CSV file containing a column of messages to classify them in bulk.")

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    with col_sample:
        st.markdown("**Need a sample CSV?**")
        sample_df = pd.DataFrame({
            "message": [
                "Win a guaranteed £500 shopping voucher by texting WIN to 80085!",
                "Are we meeting in the library at 3pm?",
                "URGENT! Call 09061701461 to claim your holiday prize!",
                "Can you forward me yesterday's lecture slides please?",
            ]
        })
        sample_csv = sample_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Sample CSV",
            data=sample_csv,
            file_name="sample_emails_to_classify.csv",
            mime="text/csv",
        )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write(f"Loaded **{len(df)}** rows. Preview:")
            st.dataframe(df.head(5), use_container_width=True)

            text_cols = df.columns.tolist()
            selected_col = st.selectbox(
                "Select column containing message text:",
                options=text_cols,
                index=0,
            )

            if st.button("⚡ Run Batch Classification", type="primary"):
                with st.spinner(f"Classifying {len(df)} messages..."):
                    # Vectorized batch prediction (much faster than per-row loops)
                    messages = df[selected_col].astype(str).tolist()
                    cleaned_messages = [clean_text(m) for m in messages]
                    probas = model.predict_proba(cleaned_messages)

                    classes = list(model.classes_)
                    spam_col_idx = classes.index(1)
                    spam_probs_arr = probas[:, spam_col_idx]

                    df["Prediction"] = ["SPAM" if p >= 0.5 else "HAM" for p in spam_probs_arr]
                    df["Spam_Confidence_%"] = [round(float(p) * 100, 2) for p in spam_probs_arr]

                st.success("Batch classification complete!")

                # Batch summary metrics
                batch_spam_cnt = (df["Prediction"] == "SPAM").sum()
                batch_ham_cnt = (df["Prediction"] == "HAM").sum()
                b_col1, b_col2, b_col3 = st.columns(3)
                b_col1.metric("Total Messages", len(df))
                b_col2.metric("Spam Flagged", f"{batch_spam_cnt} ({(batch_spam_cnt/len(df))*100:.1f}%)")
                b_col3.metric("Ham (Safe)", f"{batch_ham_cnt} ({(batch_ham_cnt/len(df))*100:.1f}%)")

                st.dataframe(df, use_container_width=True)

                # Download processed CSV
                out_buffer = io.StringIO()
                df.to_csv(out_buffer, index=False)
                st.download_button(
                    label="📥 Download Classified CSV",
                    data=out_buffer.getvalue(),
                    file_name="classified_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading CSV: {e}")

# -------------------------------------------------------------
# TAB 3: Model Performance & Dataset Insights
# -------------------------------------------------------------
with tab_metrics:
    st.markdown("### 📊 Model Architecture & Benchmark Performance")

    if metadata:
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        m_col1.metric("Accuracy", f"{metadata['accuracy'] * 100:.2f}%")
        m_col2.metric("Ham Precision", f"{metadata['classification_report']['0']['precision'] * 100:.1f}%")
        m_col3.metric("Spam Precision", f"{metadata['classification_report']['1']['precision'] * 100:.1f}%")
        m_col4.metric("Spam Recall", f"{metadata['classification_report']['1']['recall'] * 100:.1f}%")
        m_col5.metric("Spam F1-Score", f"{metadata['classification_report']['1']['f1-score'] * 100:.1f}%")

        st.divider()
        perf_col1, perf_col2 = st.columns(2)

        with perf_col1:
            test_samples = metadata.get("test_samples")
            samples_str = f"{test_samples:,}" if isinstance(test_samples, int) else str(test_samples or "N/A")
            st.markdown(f"#### Confusion Matrix (Test Set: {samples_str} samples)")
            cm = metadata["confusion_matrix"]
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Ham", "Actual Spam"],
                columns=["Predicted Ham", "Predicted Spam"],
            )
            st.dataframe(cm_df, use_container_width=True)
            st.caption(
                f"True Negatives: {cm[0][0]} | False Positives: {cm[0][1]} | False Negatives: {cm[1][0]} | True Positives: {cm[1][1]}"
            )

        with perf_col2:
            st.markdown("#### Dataset Class Distribution")
            stats = metadata["dataset_stats"]
            dist_df = pd.DataFrame({
                "Category": ["Ham (Legitimate)", "Spam"],
                "Count": [stats["ham_count"], stats["spam_count"]],
            }).set_index("Category")
            st.bar_chart(dist_df)

        st.divider()
        kw_col1, kw_col2 = st.columns(2)

        with kw_col1:
            st.markdown("#### 🚨 Top Spam Trigger Keywords")
            spam_kws = pd.DataFrame(metadata["top_spam_keywords"])
            if not spam_kws.empty:
                st.dataframe(spam_kws.head(10), use_container_width=True)

        with kw_col2:
            st.markdown("#### 🛡️ Top Ham (Safe) Keywords")
            ham_kws = pd.DataFrame(metadata["top_ham_keywords"])
            if not ham_kws.empty:
                st.dataframe(ham_kws.head(10), use_container_width=True)
    else:
        st.info("Run `python train.py` to generate the latest model performance metadata.")
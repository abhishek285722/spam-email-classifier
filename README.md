# 📧 AI Spam Email & SMS Classifier

**Kodbud Artificial Intelligence (AI) Internship — Task 1 Submission**

An end-to-end Machine Learning text classification system that detects whether an email or SMS message is **Spam** or **Ham (Legitimate)** using **scikit-learn (TF-IDF Vectorization + Multinomial Naive Bayes)** and an interactive **Streamlit** dashboard with **Explainable AI (XAI)**.

---

## 🎯 Task 1 Requirements Coverage

| Requirement | Implementation Details |
| :--- | :--- |
| **Use scikit-learn** | Built using `TfidfVectorizer`, `MultinomialNB`, and `Pipeline` from `scikit-learn`. |
| **Train using labeled dataset (ham/spam)** | Trained on the SMS Spam Collection dataset (5,574 labeled messages) loaded via `data.py`. |
| **Use TF-IDF Vectorizer & Naive Bayes** | Unigram + Bigram TF-IDF with sublinear term-frequency scaling coupled with Multinomial Naive Bayes (`alpha=0.1`). |
| **Predict whether a new message is spam** | Multi-channel prediction via **Streamlit Web UI** (`app.py`), **CLI command** (`predict.py`), and python module imports. |

---

## 📊 Model Performance

Evaluated on an independent stratified test set (1,115 samples):

- **Overall Test Accuracy:** **98.57%**
- **5-Fold Cross-Validation:** **98.44% ± 0.56%** (stable across folds)
- **Ham (Legitimate):** Precision: `99%` | Recall: `100%` | F1-Score: `0.99`
- **Spam:** Precision: `97%` | Recall: `92%` | F1-Score: `0.94`

### Confusion Matrix

| | Predicted Ham | Predicted Spam |
| :--- | :---: | :---: |
| **Actual Ham** | **962** (True Negatives) | **4** (False Positives) |
| **Actual Spam** | **12** (False Negatives) | **137** (True Positives) |

---

## 🔧 Technical Architecture

### Data Pipeline (`data.py`)

Raw messages undergo text preprocessing before model training and inference:

1. **HTML entity decoding** — converts `&gt;` → `>`, `&amp;` → `&`
2. **HTML tag removal** — strips any `<tags>` from the text
3. **URL normalization** — replaces all URLs with a `URL` placeholder
4. **Phone number normalization** — replaces number sequences with `PHONENUMBER`
5. **Currency normalization** — replaces `£500`, `$1,000` with `MONEY`
6. **Lowercasing** — ensures case-consistent vocabulary
7. **Whitespace collapsing** — normalizes spacing

### ML Pipeline (`train.py`)

```
Raw Text  →  clean_text()  →  TF-IDF Vectorizer  →  Multinomial Naive Bayes  →  Prediction
                               (1,2)-grams            alpha=0.1
                               5000 max features
                               sublinear TF
```

### Model Validation

- **Stratified train/test split** (80/20) preserves the 87%/13% class imbalance
- **5-fold stratified cross-validation** for robust accuracy estimation
- **Model serialization** via `joblib` (sklearn-recommended, faster than `pickle`)

---

## 📁 Project Structure

```
spam email classifier/
├── app.py                # Streamlit dashboard: Live classifier, Batch CSV, Model insights
├── train.py              # Training pipeline with cross-validation and metadata export
├── predict.py            # Cached inference module with explainability and CLI interface
├── data.py               # Dataset loader with text preprocessing (clean_text)
├── test_classifier.py    # Automated unit test suite (9 tests)
├── model.pkl             # Serialized trained scikit-learn pipeline (via joblib)
├── model_metadata.json   # Exported metrics, confusion matrix, CV scores & keyword weights
├── requirements.txt      # Pinned dependencies with version bounds
├── .gitignore            # Python / IDE / Streamlit ignore rules
├── README.md             # Project documentation & LinkedIn video guide
└── data/
    ├── spam.csv          # Labeled dataset (5,574 rows: label, message)
    └── SMSSpamCollection.txt  # Raw tab-separated dataset backup
```

---

## 🚀 Quick Start Guide (Run in 2 Steps)

The trained model (`model.pkl`) and dataset are **already included**!

### Option 1: One-Click Launch (Easiest)
On Windows, simply **double-click `run.bat`**! It will automatically install dependencies and launch the app in your browser.

### Option 2: Run via Terminal
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Launch Web Application:**
   ```bash
   python -m streamlit run app.py
   ```
   Open your browser at **http://localhost:8501**.

Features included in the Web App:
- **One-Click Demo Presets:** Test instant spam and ham samples with a single click.
- **Dual Confidence Meters:** Real-time progress bars for Spam % vs Ham %.
- **Explainable AI (Word Highlighter):** Color-coded keyword tokens showing why a message was flagged.
- **Batch CSV Processing:** Upload a CSV of emails, classify them in bulk, and download the tagged results.
- **Model Insights:** Visual confusion matrix, dataset distribution chart, and top spam/ham keywords.

### 2. Predict from Command Line (CLI)
```bash
# Single message
python predict.py "WINNER! You have won a £1,000 cash prize. Call now to claim."

# Interactive mode (continuous classification)
python predict.py --interactive

# Adjust spam threshold (default: 0.5)
python predict.py --threshold 0.7 "Check out this offer"
```

### 3. Re-train the Model
```bash
python train.py
```
This will:
- Load and preprocess the dataset with `clean_text()` normalization
- Run 5-fold stratified cross-validation
- Train the final model on 80% of the data
- Evaluate on the held-out 20% test set
- Save `model.pkl` (via joblib) and `model_metadata.json`

### 4. Run Automated Unit Tests
```bash
python test_classifier.py
```
Runs 9 test cases covering: dataset loading, pipeline structure, spam/ham predictions, empty input handling, explainability tokens, metadata validation, backward compatibility, and text preprocessing.

---

## 🧪 Testing & Validation

The project includes a comprehensive test suite (`test_classifier.py`) with **9 automated tests**:

| Test | What It Validates |
| :--- | :--- |
| `test_01_dataset_loader` | Dataset loads with >5,000 samples and valid binary labels |
| `test_02_model_pipeline_structure` | Model is a valid sklearn Pipeline with `tfidf` and `clf` steps |
| `test_03_spam_predictions` | Known spam messages are classified as spam with >70% confidence |
| `test_04_ham_predictions` | Known legitimate messages are classified as ham with >70% confidence |
| `test_05_empty_input_handling` | Empty/whitespace strings return ham gracefully (no exceptions) |
| `test_06_explainability_tokens` | XAI extracts relevant spam-indicative keywords (prize, claim, etc.) |
| `test_07_metadata_file` | `model_metadata.json` exists and reports >95% accuracy |
| `test_08_backward_compatibility` | Legacy `predict()` function returns `(str, float)` tuple |
| `test_09_clean_text_preprocessing` | `clean_text()` normalizes HTML entities, URLs, currency, and case |

---

## 📹 LinkedIn Video Presentation Guide (Compulsory Submission)

For your mandatory project video upload on LinkedIn (make sure to tag **Kodbud**), you can use this simple 60–90 second speaking walkthrough:

> **[0:00 - 0:15] Introduction:**  
> *"Hello everyone! As part of my Artificial Intelligence Internship at Kodbud, I have completed Task 1: Building an AI Spam Email Classifier using Python and scikit-learn."*
>
> **[0:15 - 0:40] Model & Architecture:**  
> *"For this project, I used the SMS Spam Collection dataset containing over 5,500 labeled messages. I built a text preprocessing pipeline that normalizes URLs, phone numbers, and currency, then applied TF-IDF Vectorization with unigrams and bigrams coupled with a Multinomial Naive Bayes classifier. Using 5-fold stratified cross-validation, the model achieves a stable 98.4% accuracy with 97% spam precision."*
>
> **[0:40 - 1:00] Live Demonstration:**  
> *(Click the '🎁 Lottery Winner' preset button in the Streamlit app)*  
> *"Here in the interactive dashboard, when I test a lottery message, the model instantly flags it as SPAM with high confidence, and our Explainable AI feature highlights the specific trigger keywords like 'claim', 'prize', and 'cash'."*  
> *(Click the '☕ Lunch Invitation' preset button)*  
> *"Conversely, testing a casual message correctly identifies it as legitimate HAM with 99%+ confidence."*
>
> **[1:00 - 1:15] Conclusion & Tags:**  
> *"The app also supports batch CSV classification and displays real-time model analytics. Thank you Kodbud for this opportunity!"*  
> *(Tag **@Kodbud** in your post with hashtags like #AI #MachineLearning #DataScience #Internship #Kodbud)*
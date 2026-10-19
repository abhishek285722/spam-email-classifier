"""Dataset loader and statistics helper for spam/ham email classification."""
import csv
import html
import os
import re


def clean_text(text):
    """Normalize and clean a raw message string for consistent model input.

    Applies:
        - HTML entity decoding (e.g. &gt; → >, &amp; → &)
        - HTML tag stripping
        - URL normalization → 'URL' placeholder
        - Phone number normalization → 'PHONENUMBER' placeholder
        - Currency amount normalization → 'MONEY' placeholder
        - Lowercasing
        - Whitespace collapsing
    """
    # Decode HTML entities (&gt; → >, &amp; → &, etc.)
    text = html.unescape(text)
    # Strip any remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Normalize URLs to a placeholder token
    text = re.sub(r"https?://\S+|www\.\S+", " URL ", text)
    # Normalize phone numbers (7+ digit sequences with optional separators)
    text = re.sub(r"\b\d[\d\s\-\.]{6,}\d\b", " PHONENUMBER ", text)
    # Normalize currency amounts (£500, $1,000.00, €25, etc.)
    text = re.sub(r"[£$€]\s?\d[\d,]*\.?\d*", " MONEY ", text)
    # Lowercase
    text = text.lower()
    # Collapse extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(apply_cleaning=True):
    """Load labeled emails from data/spam.csv or fallback to data/SMSSpamCollection.txt.

    Args:
        apply_cleaning: If True, applies clean_text() preprocessing to all messages.

    Returns:
        texts: list of email/SMS message strings
        labels: list of ints (1 for spam, 0 for ham)
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "data", "spam.csv")
    txt_path = os.path.join(base_dir, "data", "SMSSpamCollection.txt")

    texts = []
    labels = []

    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                msg = row.get("message", "").strip()
                lbl = row.get("label", "").strip().lower()
                if msg and lbl in ("spam", "ham"):
                    texts.append(msg)
                    labels.append(1 if lbl == "spam" else 0)
    elif os.path.exists(txt_path):
        with open(txt_path, newline="", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    lbl, msg = parts[0].strip().lower(), parts[1].strip()
                    if msg and lbl in ("spam", "ham"):
                        texts.append(msg)
                        labels.append(1 if lbl == "spam" else 0)
    else:
        raise FileNotFoundError(f"Neither {csv_path} nor {txt_path} was found.")

    if apply_cleaning:
        texts = [clean_text(t) for t in texts]

    return texts, labels


def get_dataset_stats(texts, labels):
    """Compute summary statistics for the dataset."""
    total = len(labels)
    spam_count = sum(labels)
    ham_count = total - spam_count
    spam_pct = (spam_count / total * 100) if total > 0 else 0
    ham_pct = (ham_count / total * 100) if total > 0 else 0

    return {
        "total": total,
        "spam_count": spam_count,
        "ham_count": ham_count,
        "spam_pct": spam_pct,
        "ham_pct": ham_pct,
    }
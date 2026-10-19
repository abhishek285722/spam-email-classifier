"""Predict whether a new message is spam using the trained model with explainability."""
import argparse
import logging
import os
import pickle

import joblib

from data import clean_text

logger = logging.getLogger(__name__)

_CACHED_MODEL = None
_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")


def load_model(model_path=None):
    """Load the trained model with in-memory caching for the default model path.

    Supports both joblib (preferred) and pickle serialization formats.
    """
    global _CACHED_MODEL

    use_default = model_path is None
    if use_default and _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if use_default:
        model_path = _DEFAULT_MODEL_PATH

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}. Run train.py first.")

    try:
        loaded = joblib.load(model_path)
    except Exception:
        logger.info("joblib load failed, falling back to pickle for: %s", model_path)
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)

    if use_default:
        _CACHED_MODEL = loaded

    return loaded


def explain_prediction(message, model, top_k=5):
    """Extract individual words from the message and their impact on spam vs ham scoring."""
    try:
        tfidf = model.named_steps["tfidf"]
        clf = model.named_steps["clf"]
        feature_names = tfidf.get_feature_names_out()
        vocab = tfidf.vocabulary_

        classes = list(clf.classes_)
        ham_idx = classes.index(0)
        spam_idx = classes.index(1)

        ham_log_probs = clf.feature_log_prob_[ham_idx]
        spam_log_probs = clf.feature_log_prob_[spam_idx]
        log_ratios = spam_log_probs - ham_log_probs

        # Analyze tokens in input message
        words = [w.lower().strip(".,!?:;\"'()[]{}") for w in message.split()]
        found_tokens = []
        for w in set(words):
            if w in vocab:
                idx = vocab[w]
                score = float(log_ratios[idx])
                found_tokens.append({
                    "word": w,
                    "score": score,
                    "lean": "spam" if score > 0 else "ham",
                    "magnitude": abs(score),
                })

        found_tokens.sort(key=lambda x: x["magnitude"], reverse=True)
        return found_tokens[:top_k]
    except Exception:
        logger.exception("Explainability extraction failed for message: %.80s...", message)
        return []


def predict_message(message, model=None, threshold=0.5):
    """Full prediction with preprocessing, label, probabilities, confidence, and explainability.

    Args:
        message (str): Input text message (raw, unprocessed).
        model: Optional pre-loaded pipeline.
        threshold (float): Probability threshold for spam classification (default 0.5).

    Returns:
        dict: Detailed prediction result.
    """
    if model is None:
        model = load_model()

    cleaned = clean_text(message)
    if not cleaned:
        return {
            "label": "ham",
            "result": "ham",
            "confidence": 1.0,
            "probabilities": {"ham": 1.0, "spam": 0.0},
            "spam_prob": 0.0,
            "ham_prob": 1.0,
            "tokens": [],
            "empty": True,
        }

    proba = model.predict_proba([cleaned])[0]
    classes = list(model.classes_)
    ham_prob = float(proba[classes.index(0)])
    spam_prob = float(proba[classes.index(1)])

    # Classify with threshold
    is_spam = spam_prob >= threshold
    label = "spam" if is_spam else "ham"
    confidence = spam_prob if is_spam else ham_prob

    tokens = explain_prediction(cleaned, model)

    return {
        "label": label,
        "result": label,
        "confidence": confidence,
        "probabilities": {"ham": ham_prob, "spam": spam_prob},
        "spam_prob": spam_prob,
        "ham_prob": ham_prob,
        "tokens": tokens,
        "empty": False,
    }


def predict(message):
    """Backwards-compatible prediction function returning (result, confidence)."""
    res = predict_message(message)
    return res["result"], res["confidence"]


def main():
    parser = argparse.ArgumentParser(
        description="Spam Email / SMS Classifier CLI",
    )
    parser.add_argument("message", nargs="*", help="Message to classify")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Spam decision threshold between 0.0 and 1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive prompt mode",
    )

    args = parser.parse_args()
    model = load_model()

    if args.interactive or not args.message:
        print("=" * 60)
        print("  Spam Email Classifier - Interactive Mode")
        print("  Type a message and press Enter (or 'quit' / 'exit' to stop)")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nEnter message: ").strip()
                if user_input.lower() in ("quit", "exit", "q"):
                    break
                if not user_input:
                    print("[Notice] Please enter a non-empty message.")
                    continue

                res = predict_message(user_input, model=model, threshold=args.threshold)
                badge = "[SPAM]" if res["label"] == "spam" else "[HAM / LEGIT]"
                print(f"\nResult:     {badge}")
                print(f"Confidence: {res['confidence'] * 100:.2f}%")
                print(f"Probabilities: Spam {res['spam_prob'] * 100:.2f}% | Ham {res['ham_prob'] * 100:.2f}%")
                if res["tokens"]:
                    influential = ", ".join(f"{t['word']} ({'+' if t['score'] > 0 else ''}{t['score']:.2f})" for t in res["tokens"])
                    print(f"Key Tokens: {influential}")
            except (KeyboardInterrupt, EOFError):
                break
        print("\nExiting.")
    else:
        text = " ".join(args.message)
        res = predict_message(text, model=model, threshold=args.threshold)
        print(f"Prediction: {res['label']}")
        print(f"Confidence: {res['confidence']:.2%}")
        print(f"Spam Probability: {res['spam_prob']:.2%}")
        print(f"Ham Probability:  {res['ham_prob']:.2%}")
        if res["tokens"]:
            triggers = ", ".join(f"{t['word']}" for t in res["tokens"] if t["lean"] == res["label"])
            if triggers:
                print(f"Influential Words: {triggers}")


if __name__ == "__main__":
    main()
"""Unit tests for the Spam Email Classifier."""
import json
import os
import unittest

from data import clean_text, get_dataset_stats, load_dataset
from predict import explain_prediction, load_model, predict, predict_message


class TestSpamClassifier(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.model = load_model()

    def test_01_dataset_loader(self):
        """Verify dataset loads successfully with valid binary labels."""
        texts, labels = load_dataset()
        self.assertGreater(len(texts), 5000, "Dataset should contain > 5,000 samples.")
        self.assertEqual(len(texts), len(labels), "Texts and labels must have equal length.")
        self.assertTrue(all(lbl in (0, 1) for lbl in labels), "Labels must only be 0 or 1.")

        stats = get_dataset_stats(texts, labels)
        self.assertGreater(stats["spam_count"], 500)
        self.assertGreater(stats["ham_count"], 4000)

    def test_02_model_pipeline_structure(self):
        """Verify the loaded model is a valid scikit-learn pipeline with TF-IDF and Naive Bayes."""
        self.assertTrue(hasattr(self.model, "named_steps"), "Model should be a Pipeline.")
        self.assertIn("tfidf", self.model.named_steps, "Pipeline must contain 'tfidf' step.")
        self.assertIn("clf", self.model.named_steps, "Pipeline must contain 'clf' step.")

    def test_03_spam_predictions(self):
        """Verify that typical spam messages are correctly classified."""
        spam_samples = [
            "WINNER!! You have won a 1000 cash prize. Call 09061701461 now to claim your reward!",
            "URGENT: Free entry in 2 a weekly competition to win tkts. Text FA to 87121",
            "Congratulations! You are selected for a free holiday gift voucher. Claim now.",
        ]
        for msg in spam_samples:
            res = predict_message(msg, model=self.model)
            self.assertEqual(res["label"], "spam", f"Failed on spam sample: {msg}")
            self.assertGreater(res["spam_prob"], 0.70)

    def test_04_ham_predictions(self):
        """Verify that normal legitimate messages are correctly classified."""
        ham_samples = [
            "Hey, are you free for dinner tonight around 7?",
            "Can you send me the lecture notes from today's class?",
            "Let's schedule a meeting with the team tomorrow morning to review the sprint.",
        ]
        for msg in ham_samples:
            res = predict_message(msg, model=self.model)
            self.assertEqual(res["label"], "ham", f"Failed on ham sample: {msg}")
            self.assertGreater(res["ham_prob"], 0.70)

    def test_05_empty_input_handling(self):
        """Verify that empty or whitespace strings are handled gracefully without exceptions."""
        empty_res = predict_message("   ", model=self.model)
        self.assertEqual(empty_res["label"], "ham")
        self.assertTrue(empty_res["empty"])

    def test_06_explainability_tokens(self):
        """Verify that explain_prediction extracts top tokens with impact scores."""
        msg = "URGENT prize claim now free cash"
        tokens = explain_prediction(msg, self.model)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        words = [t["word"] for t in tokens]
        self.assertTrue(any(w in words for w in ["urgent", "prize", "claim", "free", "cash"]))

    def test_07_metadata_file(self):
        """Verify model_metadata.json exists and reports high accuracy (> 95%)."""
        meta_path = os.path.join(self.base_dir, "model_metadata.json")
        self.assertTrue(os.path.exists(meta_path), "model_metadata.json should exist.")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertIn("accuracy", meta)
        self.assertGreater(meta["accuracy"], 0.95, "Accuracy should be above 95%.")

    def test_08_backward_compatibility(self):
        """Verify the legacy predict() function returns (str, float)."""
        res, conf = predict("Hi there, how are you?")
        self.assertIn(res, ["spam", "ham"])
        self.assertIsInstance(conf, float)

    def test_09_clean_text_preprocessing(self):
        """Verify that clean_text properly normalizes HTML entities, URLs, currency, and case."""
        # HTML entity decoding
        self.assertNotIn("&gt;", clean_text("Hello &gt; World"))
        self.assertNotIn("&lt;", clean_text("Price &lt; $50"))

        # URL normalization
        cleaned = clean_text("Visit https://example.com/page for details")
        self.assertIn("url", cleaned)
        self.assertNotIn("https", cleaned)

        # Currency normalization
        cleaned = clean_text("Win £1000 cash prize")
        self.assertIn("money", cleaned)

        # Phone number normalization
        cleaned = clean_text("Call 09061701461 to claim your prize")
        self.assertIn("phonenumber", cleaned)
        self.assertNotIn("09061701461", cleaned)

        # Lowercasing
        self.assertEqual(clean_text("HELLO WORLD"), "hello world")

        # Whitespace collapsing
        self.assertNotIn("  ", clean_text("Too   many    spaces"))

    def test_10_threshold_behavior(self):
        """Verify that adjusting classification threshold shifts borderline predictions."""
        msg = "free voucher for you"
        res_default = predict_message(msg, model=self.model, threshold=0.5)
        self.assertEqual(res_default["label"], "spam")

        # Raising threshold to 0.90 requires higher certainty, reclassifying to ham
        res_strict = predict_message(msg, model=self.model, threshold=0.90)
        self.assertEqual(res_strict["label"], "ham")

        # Verify threshold is strictly respected based on spam_prob
        self.assertGreaterEqual(res_default["spam_prob"], 0.5)
        self.assertLess(res_default["spam_prob"], 0.90)


if __name__ == "__main__":
    unittest.main(verbosity=2)

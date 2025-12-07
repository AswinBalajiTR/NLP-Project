from transformers import pipeline
import re

class ZeroShotJobClassifier:

    def __init__(self):
        print("Loading DeBERTa V3 zero-shot classifier...")
        self.classifier = pipeline(
            "zero-shot-classification",
            model="MoritzLaurer/deberta-v3-large-zeroshot-v1",
            framework="pt",
            device=-1
        )

        print("Loading LED extractive summarizer...")
        self.summarizer = pipeline(
            "summarization",
            model="pszemraj/led-base-book-summary",
            tokenizer="pszemraj/led-base-book-summary",
            framework="pt",
            device=-1
        )

        self.labels = [
            "This email is about job applications, recruiters, hiring, interviews, resumes, or any career opportunity.",
            "This email is not related to jobs or careers and is about something else."
        ]

        # STRICT whole-word patterns, low false positives
        self.keyword_patterns = [
            r"\binterview\b",
            r"\binterviewer\b",
            r"\brole\b",
            r"\bposition\b",
            r"\bapply\b",
            r"\bapplication\b",
            r"\brecruiter\b",
            r"\bhiring\b",
            r"\bcareer\b",
            r"\bresume\b",
            r"\bcv\b",
            r"\bjob\b",          # whole word only
            r"\boffer\b",
            r"\bopportunity\b"
        ]

    def keyword_score(self, text):
        score = 0
        for pattern in self.keyword_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                score += 1
        return score

    def extractive_summary(self, text):
        snippet = text[:2500]
        try:
            summary = self.summarizer(
                snippet,
                max_length=40,
                min_length=30,
                do_sample=False
            )[0]["summary_text"]
            return summary
        except:
            return snippet[:500]

    def zero_shot_predict(self, text):
        result = self.classifier(
            text,
            candidate_labels=self.labels,
            multi_label=False
        )
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        return top_label, top_score

    def predict(self, text):
        if not isinstance(text, str):
            text = ""

        # Step A: keyword score (robust)
        kscore = self.keyword_score(text)

        # Extractive summary (safer)
        summary = self.extractive_summary(text)

        # Step B: zero-shot
        combined = summary + "\n" + text[:1500]
        label, score = self.zero_shot_predict(combined)

        # Step C: decision logic
        if kscore >= 2 and score >= 0.40:
            return "job"

        if score >= 0.55:
            return "job"

        return "not_job"

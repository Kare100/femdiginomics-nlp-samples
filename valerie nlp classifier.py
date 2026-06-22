"""
NLP Text Classifier — Women's Economic Narratives
Author: Valerie Wangari Mwangi | vawmwangi@usiu.ac.ke
Prepared for: FemDigiNomics Application — Africa Centre for Health Systems and Gender Justice

Description:
    A lightweight NLP classification pipeline that categorises short Swahili/English
    mixed-language narratives into economic need categories relevant to FemDigiNomics:
    savings, debt, chama obligations, business risk, care work, health shocks,
    and lending readiness.

    Demonstrates: text preprocessing, feature extraction (TF-IDF), multi-class
    classification (Logistic Regression), annotation-aware label schema, and
    responsible output formatting with confidence scores.

Dependencies:
    pip install scikit-learn pandas numpy
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import re

# ─────────────────────────────────────────────
# 1. LABEL SCHEMA
#    Mirrors FemDigiNomics language categories
# ─────────────────────────────────────────────

CATEGORIES = {
    0: "savings",
    1: "debt",
    2: "chama_obligations",
    3: "business_risk",
    4: "care_work",
    5: "health_shock",
    6: "lending_readiness"
}

# ─────────────────────────────────────────────
# 2. SAMPLE TRAINING DATA
#    Mixed Swahili/English narratives (code-switched)
#    as commonly spoken in Nairobi informal economies
# ─────────────────────────────────────────────

training_data = [
    # savings
    ("Ninaweka pesa kidogo kidogo kila wiki kwa akaunti yangu.", 0),
    ("I save a small amount every Friday before I spend anything.", 0),
    ("Naweka aside at least 200 bob daily for my future.", 0),
    ("My goal is to have three months of expenses saved by December.", 0),
    ("Niweke savings yangu kwa M-PESA lockbox ili nisitumie.", 0),

    # debt
    ("Nina deni ya Fuliza ambayo inanilemea sana kila mwezi.", 1),
    ("I owe my supplier from last month and she is following up.", 1),
    ("Nilichukua loan ya 5k lakini bado sijaweza kulipa.", 1),
    ("The interest on my mobile loan has doubled what I borrowed.", 1),
    ("Deni yangu inakua kila siku, sijui nitaanza wapi.", 1),

    # chama obligations
    ("Ni wiki yangu kulipa chama, lazima nitafute pesa leo.", 2),
    ("Our merry-go-round is on Saturday and I must contribute 1000.", 2),
    ("Chama yetu inakutana kila Ijumaa, mchango ni lazima.", 2),
    ("I cannot miss my table banking contribution or I lose my slot.", 2),
    ("Wanawake wenzangu wanategemea mchango wangu wa chama.", 2),

    # business risk
    ("Biashara yangu imepoteza wateja wengi tangu baada ya mvua.", 3),
    ("My stock got damaged and I have no insurance to cover it.", 3),
    ("Competition imekuwa mingi, bei yangu haiwezi kushindana.", 3),
    ("Customers are buying on credit but not paying back on time.", 3),
    ("Nilinunua bidhaa nyingi lakini soko imeshuka ghafla.", 3),

    # care work
    ("Lazima nibaki nyumbani kumhudumia mama yangu mgonjwa.", 4),
    ("I spend most of my day taking care of my children and mother.", 4),
    ("Sina mtu wa kuachia watoto, siwezi kwenda kazini.", 4),
    ("My earnings go down every time a family member needs care.", 4),
    ("Ninatumia muda mwingi kwa kazi za nyumbani bila malipo.", 4),

    # health shock
    ("Mtoto wangu alilazwa hospitalini, gharama zilikuwa kubwa.", 5),
    ("I spent all my savings on hospital bills last month.", 5),
    ("Ugonjwa wa ghafla umevunja mpango wangu wote wa fedha.", 5),
    ("Medical emergency imefanya nishindwe kulipa rent this month.", 5),
    ("Hospitali ilitaka deposit kubwa ambayo sikuwa nayo.", 5),

    # lending readiness
    ("Nataka kukopa lakini sijui kama nitaweza kulipa kwa wakati.", 6),
    ("I have a steady income and I think I am ready for a loan.", 6),
    ("Napanga kukopa kwa biashara yangu, nina mpango mzuri.", 6),
    ("My cash flow is stable now and I want to expand with credit.", 6),
    ("Ninahitaji mkopo mdogo wa kuanzia, nina dhamana.", 6),
]

df = pd.DataFrame(training_data, columns=["text", "label"])

# ─────────────────────────────────────────────
# 3. TEXT PREPROCESSING
#    Handles Swahili, English, and code-switched text
# ─────────────────────────────────────────────

def preprocess(text: str) -> str:
    """
    Lowercase, remove punctuation, normalise whitespace.
    Preserves Swahili morphology (prefixes like ni-, ku-, wa-)
    intentionally — stripping them would lose meaning.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["text"].apply(preprocess)

# ─────────────────────────────────────────────
# 4. PIPELINE: TF-IDF + LOGISTIC REGRESSION
# ─────────────────────────────────────────────

X = df["clean_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        min_df=1,
        max_features=500,
        sublinear_tf=True     # dampens high-frequency terms
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",  # handles unequal class sizes
        random_state=42
    ))
])

pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────

y_pred = pipeline.predict(X_test)

print("=" * 60)
print("CLASSIFICATION REPORT — Women's Economic Narratives NLP")
print("=" * 60)
print(classification_report(
    y_test, y_pred,
    target_names=list(CATEGORIES.values())
))

# ─────────────────────────────────────────────
# 6. INFERENCE WITH CONFIDENCE SCORES
#    Responsible output: shows model uncertainty
#    so annotators can flag low-confidence predictions
# ─────────────────────────────────────────────

def classify_narrative(text: str, threshold: float = 0.45):
    """
    Classify a new narrative and return the predicted category
    with a confidence score. Flags low-confidence predictions
    for human review — important for responsible AI pipelines.

    Args:
        text: Raw narrative (Swahili, English, or code-switched)
        threshold: Minimum confidence to accept prediction;
                   below this, flags for human annotator review

    Returns:
        dict with predicted label, confidence, and review flag
    """
    clean = preprocess(text)
    proba = pipeline.predict_proba([clean])[0]
    pred_idx = np.argmax(proba)
    confidence = proba[pred_idx]

    return {
        "text": text,
        "predicted_category": CATEGORIES[pred_idx],
        "confidence": round(float(confidence), 3),
        "needs_human_review": confidence < threshold,
        "all_scores": {CATEGORIES[i]: round(float(p), 3) for i, p in enumerate(proba)}
    }

# ─────────────────────────────────────────────
# 7. DEMO — Unseen mixed-language narratives
# ─────────────────────────────────────────────

test_narratives = [
    "Nilikuwa na savings lakini medical bill imezimaliza zote.",
    "My chama meeting is tomorrow and I haven't managed to save.",
    "I want a small loan to restock my shop after a slow month.",
    "Sina mtu wa kuwaacha watoto, nimeshindwa kwenda kufanya kazi leo.",
    "Deni yangu kwa supplier inakua na sijui nitaanza kulipa vipi.",
]

print("\n" + "=" * 60)
print("DEMO — Classifying unseen narratives")
print("=" * 60)
for narrative in test_narratives:
    result = classify_narrative(narrative)
    flag = " ⚑ REVIEW" if result["needs_human_review"] else ""
    print(f"\nText     : {result['text']}")
    print(f"Category : {result['predicted_category'].upper()}{flag}")
    print(f"Confidence: {result['confidence']:.1%}")

print("\n" + "=" * 60)
print("Note: Low-confidence predictions are flagged for human")
print("annotator review — a key step in responsible AI pipelines.")
print("=" * 60)

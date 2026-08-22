import sys
import os
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import scipy.sparse as sparse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from google import genai
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# DATABASE
# ============================================================

from database.database import get_engine


# ============================================================
# CONFIGURATION
# ============================================================

TARGET = "churn"

NUMERICAL_FEATURES = [
    "montant",
    "frequence_rech",
    "revenue",
    "arpu_segment",
    "frequence",
    "data_volume",
    "on_net",
    "orange",
    "tigo",
    "zone1",
    "zone2",
    "regularity",
    "freq_top_pack"
]

CATEGORICAL_FEATURES = [
    "region",
    "tenure",
    "mrg",
    "top_pack"
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

CLASSIFICATION_THRESHOLD = 0.60

MODEL_DIR = PROJECT_ROOT / "models"

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

PREDICTION_FILE = (
    PROJECT_ROOT
    / "reports"
    / "customer_churn_predictions.csv"
)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Telecom Churn Intelligence API",
    description=(
        "AI-powered telecom customer churn prediction, "
        "risk analysis and RAG-based intelligence API."
    ),
    version="1.0.0"
)


# ============================================================
# LOAD MACHINE LEARNING COMPONENTS
# ============================================================

print("=" * 70)
print("LOADING MACHINE LEARNING COMPONENTS")
print("=" * 70)

MODEL_PATH = MODEL_DIR / "tuned_random_forest.pkl"

PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"

model = joblib.load(MODEL_PATH)

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Random Forest loaded.")

print("Preprocessor loaded.")

print(
    f"Classification threshold: "
    f"{CLASSIFICATION_THRESHOLD}"
)


# ============================================================
# LOAD CUSTOMER PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("LOADING CUSTOMER PREDICTIONS")
print("=" * 70)

if PREDICTION_FILE.exists():

    predictions_df = pd.read_csv(
        PREDICTION_FILE
    )

    print(
        f"Customer predictions loaded: "
        f"{len(predictions_df):,}"
    )

else:

    predictions_df = pd.DataFrame()

    print(
        "Warning: customer prediction file "
        "was not found."
    )


# ============================================================
# LOAD RAG KNOWLEDGE BASE
# ============================================================

print("\n" + "=" * 70)
print("LOADING RAG KNOWLEDGE BASE")
print("=" * 70)

knowledge_base_path = (
    KNOWLEDGE_BASE_DIR
    / "knowledge_base.json"
)

embeddings_path = (
    KNOWLEDGE_BASE_DIR
    / "knowledge_embeddings.npy"
)


import json


if not knowledge_base_path.exists():

    raise FileNotFoundError(
        "knowledge_base.json not found."
    )


if not embeddings_path.exists():

    raise FileNotFoundError(
        "knowledge_embeddings.npy not found."
    )


with open(
    knowledge_base_path,
    "r",
    encoding="utf-8"
) as file:

    knowledge_base = json.load(file)


knowledge_embeddings = np.load(
    embeddings_path
)


print(
    f"Knowledge documents loaded: "
    f"{len(knowledge_base)}"
)

print(
    f"Embedding matrix shape: "
    f"{knowledge_embeddings.shape}"
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\n" + "=" * 70)
print("LOADING EMBEDDING MODEL")
print("=" * 70)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print(
    "Embedding model loaded."
)


# ============================================================
# GEMINI CLIENT
# ============================================================

print("\n" + "=" * 70)
print("CONFIGURING GEMINI")
print("=" * 70)

gemini_api_key = os.getenv(
    "GEMINI_API_KEY"
)

if not gemini_api_key:

    print(
        "WARNING: GEMINI_API_KEY is not set."
    )

    gemini_client = None

else:

    gemini_client = genai.Client(
        api_key=gemini_api_key
    )

    print(
        "Gemini client configured."
    )


# ============================================================
# REQUEST MODELS
# ============================================================

class CustomerRequest(BaseModel):

    user_id: str


class QuestionRequest(BaseModel):

    question: str


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "AI-Powered Telecom Churn Intelligence",
        "status": "online",
        "version": "1.0.0",
        "services": [
            "customer risk prediction",
            "RAG retrieval",
            "Gemini AI assistant"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "knowledge_base_loaded": len(knowledge_base) > 0,
        "gemini_configured": gemini_client is not None
    }


# ============================================================
# CUSTOMER RISK ENDPOINT
# ============================================================

@app.post("/predict")
def predict_customer(
    request: CustomerRequest
):

    if predictions_df.empty:

        raise HTTPException(
            status_code=500,
            detail=(
                "Customer prediction file "
                "is not available."
            )
        )


    customer = predictions_df[
        predictions_df["user_id"].astype(str)
        == str(request.user_id)
    ]


    if customer.empty:

        raise HTTPException(
            status_code=404,
            detail="Customer not found."
        )


    customer = customer.iloc[0]


    return {
        "user_id": str(
            customer["user_id"]
        ),

        "churn_probability": float(
            customer["churn_probability"]
        ),

        "churn_prediction": int(
            customer["churn_prediction"]
        ),

        "risk_level": str(
            customer["risk_level"]
        ),

        "classification_threshold":
            CLASSIFICATION_THRESHOLD
    }


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_documents(
    question,
    top_k=3
):

    query_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True
    )


    similarities = cosine_similarity(
        query_embedding,
        knowledge_embeddings
    )[0]


    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]


    results = []


    for index in top_indices:

        document = knowledge_base[index]

        results.append(
            {
                "title":
                    document["title"],

                "category":
                    document["category"],

                "content":
                    document["content"],

                "similarity":
                    float(
                        similarities[index]
                    )
            }
        )


    return results


# ============================================================
# GEMINI RAG ASSISTANT
# ============================================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    if gemini_client is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Gemini API is not configured. "
                "Set GEMINI_API_KEY."
            )
        )


    question = request.question.strip()


    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )


    # --------------------------------------------------------
    # RETRIEVE KNOWLEDGE
    # --------------------------------------------------------

    retrieved_documents = retrieve_documents(
        question,
        top_k=3
    )


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context_parts = []


    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"""
SOURCE {i}

Title:
{document["title"]}

Category:
{document["category"]}

Similarity:
{document["similarity"]:.4f}

Content:
{document["content"]}
"""
        )


    context = "\n".join(
        context_parts
    )


    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an AI assistant for a Telecom Churn
Intelligence system.

Answer the user's question using ONLY the
project knowledge provided below.

IMPORTANT RULES:

1. Use the provided project knowledge as the
   primary source.

2. Do not invent model results, statistics,
   customer information, or business findings.

3. If the provided context does not contain
   enough information, clearly say that the
   project knowledge base does not contain
   enough information.

4. Explain technical results in a way that
   a business user can understand.

5. When numerical results are available,
   preserve the actual values.

PROJECT KNOWLEDGE:

{context}

USER QUESTION:

{question}
"""


    # --------------------------------------------------------
    # GENERATE ANSWER
    # --------------------------------------------------------

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )


    answer = response.text


    # --------------------------------------------------------
    # RETURN RESPONSE
    # --------------------------------------------------------

    return {

        "question": question,

        "answer": answer,

        "sources": [
            {
                "title":
                    document["title"],

                "category":
                    document["category"],

                "similarity":
                    round(
                        document["similarity"],
                        4
                    )
            }

            for document
            in retrieved_documents
        ]
    }
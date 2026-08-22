import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "knowledge_base.json"
)

EMBEDDINGS_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "knowledge_embeddings.npy"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

TOP_K = 3


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    return np.load(
        EMBEDDINGS_PATH
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# SEMANTIC RETRIEVAL
# ============================================================

def retrieve_documents(
    query,
    knowledge_base,
    embeddings,
    model,
    top_k=TOP_K
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    similarity_scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarity_scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "score": float(
                    similarity_scores[index]
                ),
                "document": knowledge_base[index]
            }
        )

    return results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        title = document.get(
            "title",
            "Unknown"
        )

        category = document.get(
            "category",
            "Unknown"
        )

        content = document.get(
            "content",
            ""
        )

        score = result["score"]

        context = f"""
SOURCE {number}

Title: {title}

Category: {category}

Similarity Score: {score:.4f}

Content:
{content}
"""

        context_parts.append(
            context.strip()
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# BUILD LLM PROMPT
# ============================================================

def build_prompt(
    question,
    context
):

    prompt = f"""
You are an AI assistant for a Telecom Churn
Intelligence system.

Answer the user's question using the project
knowledge provided below.

IMPORTANT RULES:

1. Use the provided project knowledge as the
   primary source of information.

2. Do not invent model results, statistics,
   customer information, or business findings.

3. If the provided context does not contain
   enough information to answer the question,
   clearly say that the project knowledge base
   does not contain enough information.

4. Explain technical results in a way that
   a business user can understand.

5. When numerical results are available,
   preserve the actual values from the project.

PROJECT KNOWLEDGE:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    return prompt.strip()


# ============================================================
# RAG PIPELINE
# ============================================================

def run_rag(
    question,
    top_k=TOP_K
):

    knowledge_base = (
        load_knowledge_base()
    )

    embeddings = (
        load_embeddings()
    )

    model = (
        load_embedding_model()
    )

    results = retrieve_documents(
        query=question,
        knowledge_base=knowledge_base,
        embeddings=embeddings,
        model=model,
        top_k=top_k
    )

    context = build_context(
        results
    )

    prompt = build_prompt(
        question=question,
        context=context
    )

    return {
        "question": question,
        "results": results,
        "context": context,
        "prompt": prompt
    }


# ============================================================
# TEST RAG ENGINE
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TELECOM CHURN RAG ENGINE")
    print("=" * 70)

    question = input(
        "\nEnter your question: "
    )

    result = run_rag(
        question
    )

    print("\n" + "=" * 70)
    print("RETRIEVED KNOWLEDGE")
    print("=" * 70)

    for number, item in enumerate(
        result["results"],
        start=1
    ):

        document = item["document"]

        print(
            f"\n{number}. "
            f"{document.get('title', '')}"
        )

        print(
            f"Similarity: "
            f"{item['score']:.4f}"
        )

    print("\n" + "=" * 70)
    print("GENERATED RAG CONTEXT")
    print("=" * 70)

    print(
        "\n" + result["context"]
    )

    print("\n" + "=" * 70)
    print("LLM PROMPT")
    print("=" * 70)

    print(
        "\n" + result["prompt"]
    )

    print("\n" + "=" * 70)
    print("RAG ENGINE TEST COMPLETED")
    print("=" * 70)
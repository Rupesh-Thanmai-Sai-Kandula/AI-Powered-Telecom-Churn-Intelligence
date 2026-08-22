import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_PATH = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "knowledge_base.json"
)

VECTOR_DIR = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
)

EMBEDDINGS_PATH = (
    VECTOR_DIR
    / "knowledge_embeddings.npy"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    print("\n" + "=" * 70)
    print("LOADING KNOWLEDGE BASE")
    print("=" * 70)

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        knowledge_base = json.load(file)

    print(
        f"\nDocuments loaded: "
        f"{len(knowledge_base)}"
    )

    return knowledge_base


# ============================================================
# CREATE DOCUMENT TEXT
# ============================================================

def create_document_text(document):

    title = str(
        document.get("title", "")
    )

    category = str(
        document.get("category", "")
    )

    content = str(
        document.get("content", "")
    )

    return (
        f"Title: {title}\n"
        f"Category: {category}\n"
        f"Content: {content}"
    )


# ============================================================
# BUILD EMBEDDINGS
# ============================================================

def build_embeddings(
    knowledge_base,
    model
):

    print("\n" + "=" * 70)
    print("GENERATING DOCUMENT EMBEDDINGS")
    print("=" * 70)

    documents = [
        create_document_text(document)
        for document in knowledge_base
    ]

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print(
        f"\nEmbedding matrix shape: "
        f"{embeddings.shape}"
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(embeddings):

    VECTOR_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.save(
        EMBEDDINGS_PATH,
        embeddings
    )

    print(
        "\nEmbeddings saved to:"
        f"\n{EMBEDDINGS_PATH}"
    )


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    knowledge_base,
    embeddings,
    model,
    top_k=3
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Because both vectors are normalized,
    # dot product = cosine similarity.
    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append(
            {
                "score": float(
                    scores[index]
                ),
                "document": knowledge_base[index]
            }
        )

    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n" + "=" * 70)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 70)

    print(
        f"\nQuery:\n{query}"
    )

    print(
        f"\nTop {len(results)} relevant documents:"
    )

    for position, result in enumerate(
        results,
        start=1
    ):

        document = result["document"]

        print("\n" + "-" * 70)

        print(
            f"RESULT {position}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print(
            f"ID: "
            f"{document.get('id', '')}"
        )

        print(
            f"Title: "
            f"{document.get('title', '')}"
        )

        print(
            f"Category: "
            f"{document.get('category', '')}"
        )

        print(
            f"\nContent:\n"
            f"{document.get('content', '')}"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TELECOM CHURN SEMANTIC RETRIEVER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        MODEL_NAME
    )

    print(
        f"Embedding model loaded: "
        f"{MODEL_NAME}"
    )

    # --------------------------------------------------------
    # Load knowledge base
    # --------------------------------------------------------

    knowledge_base = (
        load_knowledge_base()
    )

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    if EMBEDDINGS_PATH.exists():

        print(
            "\nExisting embeddings found."
        )

        embeddings = np.load(
            EMBEDDINGS_PATH
        )

        print(
            f"Embeddings loaded: "
            f"{embeddings.shape}"
        )

    else:

        embeddings = build_embeddings(
            knowledge_base,
            model
        )

        save_embeddings(
            embeddings
        )

    # --------------------------------------------------------
    # Ask question
    # --------------------------------------------------------

    query = input(
        "\nEnter your question: "
    )

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    results = semantic_search(
        query=query,
        knowledge_base=knowledge_base,
        embeddings=embeddings,
        model=model,
        top_k=3
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    display_results(
        query,
        results
    )

    print("\n" + "=" * 70)
    print("SEMANTIC RETRIEVAL COMPLETED")
    print("=" * 70)
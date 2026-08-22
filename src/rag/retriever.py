import json
from pathlib import Path
from difflib import SequenceMatcher


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


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

def load_knowledge_base():

    with open(
        KNOWLEDGE_BASE_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        knowledge_base = json.load(file)

    return knowledge_base


# ============================================================
# TEXT SIMILARITY
# ============================================================

def calculate_similarity(query, text):

    query_words = set(
        query.lower().split()
    )

    text_words = set(
        text.lower().split()
    )

    if not query_words or not text_words:
        return 0

    common_words = (
        query_words.intersection(text_words)
    )

    word_score = (
        len(common_words)
        / len(query_words)
    )

    sequence_score = SequenceMatcher(
        None,
        query.lower(),
        text.lower()
    ).ratio()

    return (
        0.7 * word_score
        +
        0.3 * sequence_score
    )


# ============================================================
# RETRIEVE RELEVANT DOCUMENTS
# ============================================================

def retrieve_documents(
    query,
    top_k=3
):

    knowledge_base = load_knowledge_base()

    results = []

    for document in knowledge_base:

        # Combine the important fields of each document
        searchable_text = " ".join(
            str(value)
            for value in document.values()
        )

        score = calculate_similarity(
            query,
            searchable_text
        )

        results.append(
            {
                "score": score,
                "document": document
            }
        )

    # Highest similarity first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n" + "=" * 70)
    print("RAG RETRIEVAL RESULTS")
    print("=" * 70)

    print(f"\nQuery:")
    print(query)

    print(
        f"\nTop {len(results)} relevant documents:"
    )

    for index, result in enumerate(
        results,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"RESULT {index}"
        )

        print(
            f"Similarity Score: "
            f"{result['score']:.4f}"
        )

        print("\nDocument:")

        print(
            json.dumps(
                result["document"],
                indent=2,
                ensure_ascii=False
            )
        )


# ============================================================
# TEST RETRIEVER
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TELECOM CHURN RAG RETRIEVER")
    print("=" * 70)

    query = input(
        "\nEnter your question: "
    )

    results = retrieve_documents(
        query,
        top_k=3
    )

    display_results(
        query,
        results
    )
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORTS_DIR = PROJECT_ROOT / "reports"

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

PREDICTIONS_FILE = (
    REPORTS_DIR / "customer_churn_predictions.csv"
)

KNOWLEDGE_BASE_FILE = (
    KNOWLEDGE_BASE_DIR / "knowledge_base.json"
)

EMBEDDINGS_FILE = (
    KNOWLEDGE_BASE_DIR / "knowledge_embeddings.npy"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Telecom Churn Intelligence",
    page_icon="📡",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #8b949e;
        margin-bottom: 30px;
    }

    .risk-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #161b22;
        margin-bottom: 15px;
    }

    .ai-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #30363d;
        background-color: #161b22;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD CUSTOMER PREDICTIONS
# ============================================================

@st.cache_data
def load_predictions():

    if not PREDICTIONS_FILE.exists():
        return None

    return pd.read_csv(
        PREDICTIONS_FILE
    )


df = load_predictions()


if df is None:

    st.error(
        "Customer prediction file was not found."
    )

    st.write(
        f"Expected file:\n\n{PREDICTIONS_FILE}"
    )

    st.stop()


# ============================================================
# LOAD KNOWLEDGE BASE
# ============================================================

@st.cache_data
def load_knowledge_base():

    if not KNOWLEDGE_BASE_FILE.exists():
        return []

    with open(
        KNOWLEDGE_BASE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


knowledge_base = load_knowledge_base()


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

@st.cache_data
def load_embeddings():

    if not EMBEDDINGS_FILE.exists():
        return None

    return np.load(
        EMBEDDINGS_FILE
    )


embeddings = load_embeddings()


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_embedding_model()


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

api_key = os.getenv(
    "GEMINI_API_KEY"
)

if api_key:

    gemini_client = genai.Client(
        api_key=api_key
    )

else:

    gemini_client = None


GEMINI_MODEL = "gemini-2.5-flash"


# ============================================================
# RAG RETRIEVAL FUNCTION
# ============================================================

def retrieve_knowledge(
    query,
    top_k=3
):

    if not knowledge_base:
        return []

    if embeddings is None:
        return []

    query_embedding = (
        embedding_model
        .encode(
            [query],
            normalize_embeddings=True
        )[0]
    )

    scores = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        document = knowledge_base[index]

        results.append(
            {
                "id": document.get(
                    "id",
                    ""
                ),

                "title": document.get(
                    "title",
                    ""
                ),

                "category": document.get(
                    "category",
                    ""
                ),

                "content": document.get(
                    "content",
                    ""
                ),

                "similarity": float(
                    scores[index]
                )
            }
        )

    return results


# ============================================================
# GROUNDED GEMINI RESPONSE
# ============================================================

def generate_ai_answer(
    question,
    customer_context="",
    retrieved_documents=None
):

    if gemini_client is None:

        return (
            "Gemini API key is not configured. "
            "Set the GEMINI_API_KEY environment variable "
            "before using AI analysis."
        )

    if retrieved_documents is None:
        retrieved_documents = []


    # --------------------------------------------------------
    # BUILD RAG CONTEXT
    # --------------------------------------------------------

    rag_context = ""

    for i, document in enumerate(
        retrieved_documents,
        start=1
    ):

        rag_context += f"""
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


    # --------------------------------------------------------
    # SYSTEM INSTRUCTION
    # --------------------------------------------------------

    system_instruction = """
You are an AI assistant for a Telecom Churn Intelligence system.

Your job is to explain customer churn predictions and project
findings using ONLY the information provided.

IMPORTANT GROUNDING RULES:

1. Use the supplied project/customer information as the
   primary source.

2. NEVER invent customer characteristics, churn reasons,
   complaints, revenue information, communication preferences,
   model results, statistics, or business findings.

3. A churn probability is a MODEL ESTIMATE.
   It is NOT a guarantee that the customer will churn.

4. Do not describe a predicted churn as a confirmed real-world event.

5. Do not assume why a particular customer will churn unless
   the supplied information explicitly states the reason.

6. Do not invent customer dissatisfaction, network problems,
   payment problems, usage problems, or service problems.

7. Do not recommend a specific communication channel unless
   the supplied information supports it.

8. Business recommendations must be cautious and evidence-based.

9. If the available information is insufficient to determine
   something, explicitly say:

   "The available project/customer information is not sufficient
   to determine this."

10. Preserve numerical values supplied by the project.

11. Clearly distinguish between:
    - model predictions
    - project findings
    - business recommendations

12. Explain technical results in understandable business language.

13. Never fabricate information merely to make an answer
    appear more complete.

14. The purpose of this system is decision support, not certainty.
"""


    # --------------------------------------------------------
    # FINAL PROMPT
    # --------------------------------------------------------

    prompt = f"""
{system_instruction}

============================================================
RETRIEVED PROJECT KNOWLEDGE
============================================================

{rag_context}

============================================================
CUSTOMER INFORMATION
============================================================

{customer_context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER
============================================================

Provide a clear, concise and evidence-grounded answer.
"""


    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    return response.text


# ============================================================
# PAGE HEADER
# ============================================================

st.markdown(
    '<div class="main-title">'
    '📡 Telecom Churn Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Powered Customer Churn Prediction, Risk Analysis, '
    'RAG Intelligence and Gemini-Powered Decision Support'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Navigation"
)

page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard",
        "Customer Risk",
        "RAG Intelligence",
        "AI Customer Analysis",
        "Model Information"
    ]
)


# ============================================================
# GLOBAL METRICS
# ============================================================

total_customers = len(df)

predicted_churn = (
    df["churn_prediction"] == 1
).sum()

predicted_no_churn = (
    df["churn_prediction"] == 0
).sum()

churn_rate = (
    predicted_churn /
    total_customers
) * 100


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.header(
        "Executive Dashboard"
    )

    st.write(
        "Overview of customer churn predictions generated "
        "by the final tuned Random Forest model."
    )

    st.divider()


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:

        st.metric(
            "Predicted Churn",
            f"{predicted_churn:,}"
        )

    with col3:

        st.metric(
            "Predicted No Churn",
            f"{predicted_no_churn:,}"
        )

    with col4:

        st.metric(
            "Predicted Churn Rate",
            f"{churn_rate:.2f}%"
        )


    st.divider()


    # --------------------------------------------------------
    # RISK DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "Customer Risk Distribution"
    )

    risk_counts = (
        df["risk_level"]
        .value_counts()
        .reindex(
            [
                "Very High",
                "High",
                "Medium",
                "Low",
                "Very Low"
            ],
            fill_value=0
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.dataframe(
            risk_counts.rename(
                "Customers"
            ),
            use_container_width=True
        )


    with col2:

        st.bar_chart(
            risk_counts
        )


    st.divider()


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader(
        "Final Model Performance"
    )

    st.caption(
        "Tuned Random Forest with optimized "
        "classification threshold of 0.60."
    )


    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            "0.8545"
        )

    with col2:
        st.metric(
            "Precision",
            "0.5753"
        )

    with col3:
        st.metric(
            "Recall",
            "0.8560"
        )

    with col4:
        st.metric(
            "F1 Score",
            "0.6881"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            "0.9281"
        )


# ============================================================
# CUSTOMER RISK
# ============================================================

elif page == "Customer Risk":

    st.header(
        "Customer Risk Explorer"
    )

    st.write(
        "Search for a customer and inspect their "
        "predicted churn risk."
    )

    st.divider()


    customer_id = st.text_input(
        "Enter Customer ID"
    )


    if customer_id:

        customer_id = customer_id.strip()

        customer = df[
            df["user_id"].astype(str)
            == customer_id
        ]


        if customer.empty:

            st.error(
                "Customer ID was not found."
            )

        else:

            customer = customer.iloc[0]


            probability = float(
                customer[
                    "churn_probability"
                ]
            )

            prediction = int(
                customer[
                    "churn_prediction"
                ]
            )

            risk = customer[
                "risk_level"
            ]


            st.subheader(
                "Customer Risk Profile"
            )


            col1, col2, col3, col4 = st.columns(4)


            with col1:

                st.metric(
                    "Customer ID",
                    customer_id
                )


            with col2:

                st.metric(
                    "Churn Probability",
                    f"{probability * 100:.2f}%"
                )


            with col3:

                st.metric(
                    "Prediction",
                    "Churn"
                    if prediction == 1
                    else "No Churn"
                )


            with col4:

                st.metric(
                    "Risk Level",
                    risk
                )


            st.divider()


            st.subheader(
                "Customer Recommendation"
            )


            if risk == "Very High":

                st.warning(
                    "Prioritize this customer for "
                    "retention analysis and intervention."
                )

            elif risk == "High":

                st.warning(
                    "Consider this customer for "
                    "retention analysis."
                )

            elif risk == "Medium":

                st.info(
                    "Monitor this customer and consider "
                    "appropriate retention actions."
                )

            else:

                st.success(
                    "Customer is currently in a lower "
                    "predicted churn-risk segment."
                )


# ============================================================
# RAG INTELLIGENCE
# ============================================================

elif page == "RAG Intelligence":

    st.header(
        "🔎 RAG Intelligence"
    )

    st.write(
        "Ask questions about the project's model, "
        "data analysis and business findings."
    )

    st.divider()


    question = st.text_area(
        "Ask a question",
        placeholder=(
            "Example: Why did we choose a threshold of 0.60?"
        ),
        height=100
    )


    if st.button(
        "Search Knowledge Base",
        type="primary"
    ):

        if not question.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            with st.spinner(
                "Searching project knowledge..."
            ):

                results = retrieve_knowledge(
                    question,
                    top_k=3
                )


            st.subheader(
                "Retrieved Knowledge"
            )


            for i, result in enumerate(
                results,
                start=1
            ):

                with st.expander(
                    f"{i}. {result['title']} "
                    f"— Similarity {result['similarity']:.4f}"
                ):

                    st.write(
                        f"**Category:** "
                        f"{result['category']}"
                    )

                    st.write(
                        result["content"]
                    )


            if results:

                with st.spinner(
                    "Generating grounded Gemini answer..."
                ):

                    answer = generate_ai_answer(
                        question=question,
                        retrieved_documents=results
                    )


                st.divider()

                st.subheader(
                    "🤖 Gemini Answer"
                )

                st.markdown(
                    answer
                )


# ============================================================
# AI CUSTOMER ANALYSIS
# ============================================================

elif page == "AI Customer Analysis":

    st.header(
        "🤖 AI Customer Risk Analysis"
    )

    st.write(
        "Gemini analyzes a customer's model prediction "
        "using the project's available evidence."
    )

    st.divider()


    customer_id = st.text_input(
        "Enter Customer ID",
        key="ai_customer_id"
    )


    question = st.text_area(
        "What would you like to know?",
        value=(
            "What does this customer's predicted risk mean, "
            "why should they be prioritized, and what business "
            "action is appropriate?"
        ),
        height=120
    )


    if st.button(
        "Analyze Customer",
        type="primary"
    ):

        if not customer_id.strip():

            st.warning(
                "Please enter a Customer ID."
            )

        else:

            customer = df[
                df["user_id"].astype(str)
                == customer_id.strip()
            ]


            if customer.empty:

                st.error(
                    "Customer ID was not found."
                )

            else:

                customer = customer.iloc[0]


                probability = float(
                    customer[
                        "churn_probability"
                    ]
                )

                prediction = int(
                    customer[
                        "churn_prediction"
                    ]
                )

                risk = customer[
                    "risk_level"
                ]


                customer_context = f"""
Customer ID:
{customer_id}

Model-estimated churn probability:
{probability:.6f}

Model-estimated churn probability:
{probability * 100:.2f}%

Predicted class:
{
    "Churn"
    if prediction == 1
    else "No Churn"
}

Risk level:
{risk}

This information represents model output.
It does not establish the customer's actual future behavior,
specific reason for churn, financial value, dissatisfaction,
or preferred retention strategy.
"""


                # ------------------------------------------------
                # RETRIEVE RELATED KNOWLEDGE
                # ------------------------------------------------

                with st.spinner(
                    "Retrieving relevant project knowledge..."
                ):

                    results = retrieve_knowledge(
                        question,
                        top_k=3
                    )


                # ------------------------------------------------
                # GENERATE AI ANALYSIS
                # ------------------------------------------------

                with st.spinner(
                    "Generating grounded AI analysis..."
                ):

                    answer = generate_ai_answer(
                        question=question,
                        customer_context=customer_context,
                        retrieved_documents=results
                    )


                # ------------------------------------------------
                # CUSTOMER SUMMARY
                # ------------------------------------------------

                st.subheader(
                    "Customer Prediction"
                )


                col1, col2, col3, col4 = st.columns(4)


                with col1:

                    st.metric(
                        "Customer",
                        customer_id
                    )


                with col2:

                    st.metric(
                        "Churn Probability",
                        f"{probability * 100:.2f}%"
                    )


                with col3:

                    st.metric(
                        "Prediction",
                        "Churn"
                        if prediction == 1
                        else "No Churn"
                    )


                with col4:

                    st.metric(
                        "Risk",
                        risk
                    )


                st.divider()


                # ------------------------------------------------
                # AI RESPONSE
                # ------------------------------------------------

                st.subheader(
                    "AI Risk Analysis"
                )

                st.markdown(
                    answer
                )


                # ------------------------------------------------
                # SOURCES
                # ------------------------------------------------

                st.divider()

                st.subheader(
                    "RAG Sources Used"
                )


                for result in results:

                    st.caption(
                        f"**{result['title']}** "
                        f"— similarity "
                        f"{result['similarity']:.4f}"
                    )


# ============================================================
# MODEL INFORMATION
# ============================================================

elif page == "Model Information":

    st.header(
        "Model & System Information"
    )

    st.write(
        "Technical information about the Telecom Churn "
        "Intelligence system."
    )

    st.divider()


    st.subheader(
        "Machine Learning"
    )

    st.write(
        """
        **Model:** Tuned Random Forest

        **Classification Threshold:** 0.60

        The threshold was optimized to maximize the F1 score
        for the churn classification task.
        """
    )


    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "Accuracy",
            "0.8545"
        )

    with col2:
        st.metric(
            "Precision",
            "0.5753"
        )

    with col3:
        st.metric(
            "Recall",
            "0.8560"
        )

    with col4:
        st.metric(
            "F1",
            "0.6881"
        )

    with col5:
        st.metric(
            "ROC-AUC",
            "0.9281"
        )


    st.divider()


    st.subheader(
        "RAG System"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.info(
            "**Knowledge Documents**\n\n"
            f"{len(knowledge_base)}"
        )

    with col2:

        st.info(
            "**Embedding Model**\n\n"
            "all-MiniLM-L6-v2"
        )

    with col3:

        st.info(
            "**Vector Dimensions**\n\n"
            "384"
        )


    st.divider()


    st.subheader(
        "Generative AI"
    )

    st.info(
        "**LLM:** Gemini 2.5 Flash\n\n"
        "**Purpose:** Grounded explanation and "
        "business-oriented decision support."
    )


    st.divider()


    st.subheader(
        "System Architecture"
    )

    st.code(
        """
PostgreSQL
     │
     ▼
Data Processing
     │
     ▼
Feature Engineering
     │
     ▼
Tuned Random Forest
     │
     ▼
Customer Churn Predictions
     │
     ├──────────────► Risk Analysis
     │
     ▼
Knowledge Base
     │
     ▼
Semantic Embeddings
     │
     ▼
RAG Retrieval
     │
     ▼
Gemini LLM
     │
     ▼
Streamlit Dashboard
        """,
        language="text"
    )
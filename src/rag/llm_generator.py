import os
import sys
from pathlib import Path

from google import genai


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))


# ============================================================
# GEMINI API CONFIGURATION
# ============================================================

def get_gemini_api_key():

    # --------------------------------------------------------
    # 1. Check local environment variable
    # --------------------------------------------------------

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        return api_key

    # --------------------------------------------------------
    # 2. Check Streamlit Secrets
    # --------------------------------------------------------

    try:
        import streamlit as st

        api_key = st.secrets.get("GEMINI_API_KEY")

        if api_key:
            return api_key

    except Exception:
        pass

    return None


api_key = get_gemini_api_key()


if not api_key:
    raise ValueError(
        "Gemini API key not found.\n"
        "Set GEMINI_API_KEY as an environment variable "
        "or configure it in Streamlit Secrets."
    )


client = genai.Client(
    api_key=api_key
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# SYSTEM INSTRUCTION
# ============================================================

SYSTEM_INSTRUCTION = """
You are an AI assistant for a Telecom Churn Intelligence system.

Your job is to explain customer churn predictions and project findings
using ONLY the information provided in the project knowledge and
customer prediction context.

IMPORTANT GROUNDING RULES:

1. Use the provided project/customer information as the primary and
   authoritative source.

2. NEVER invent:

   - customer characteristics
   - reasons why a customer may churn
   - customer complaints
   - revenue information
   - communication preferences
   - customer behavior that is not provided
   - model results
   - statistics
   - business findings

3. A churn probability is a MODEL ESTIMATE.

   Do NOT describe it as a certainty or guarantee that a customer
   will churn.

   GOOD:
   "The model assigns this customer an estimated churn probability
   of approximately 90.01%."

   BAD:
   "This customer has a 90% guaranteed chance of leaving."

4. When discussing risk:

   Explain the risk level using the project's defined risk categories
   and the available prediction information.

5. When recommending business actions:

   Only recommend actions that are supported by the project knowledge.

   If the available information is insufficient to determine a specific
   retention action, explicitly say so.

6. Do NOT assume a specific churn reason unless the provided information
   explicitly identifies one.

7. Do NOT recommend a specific communication channel such as phone,
   email, or SMS unless the provided information supports it.

8. Do NOT claim that a customer is experiencing:

   - revenue loss
   - dissatisfaction
   - service problems
   - payment problems
   - network problems
   - usage problems

   unless the provided information explicitly states this.

9. General business assumptions must NOT be presented as project findings.

10. If the provided information is insufficient to answer part of the
    question, clearly state:

    "The available project/customer information is not sufficient
    to determine this."

11. When numerical values are provided, preserve the actual values
    from the project.

12. Explain technical results in clear business language.

13. For customer-specific questions, structure the answer when useful
    using:

    - What the prediction means
    - Why the customer should be prioritized
    - Appropriate business action

14. Keep the distinction clear between:

    - What the model predicts
    - What the project data tells us
    - What can reasonably be recommended

15. Never turn a model prediction into a confirmed real-world event.

16. Do not fabricate information simply to make the answer more
    complete.

17. If the customer information only contains a probability,
    prediction, and risk level, limit the explanation to those
    available facts.

18. Recommendations should be cautious and evidence-based.

19. The purpose of this system is decision support, not certainty.
"""


# ============================================================
# PROJECT KNOWLEDGE
# ============================================================

PROJECT_KNOWLEDGE = """
The Telecom Churn Intelligence project contains a machine-learning
based customer churn prediction system.

The final model is a tuned Random Forest.

The classification threshold was optimized to 0.60.

At threshold 0.60, the final model achieved:

Accuracy = 0.8545
Precision = 0.5753
Recall = 0.8560
F1 = 0.6881
ROC-AUC = 0.9281

The threshold of 0.60 was selected because it produced the best
F1 score during threshold optimization.

The prediction system converts churn probability into five
risk categories:

Very Low
Low
Medium
High
Very High

The risk categories are intended to provide a more understandable
business interpretation of numerical churn probabilities.

The project focuses on identifying customers who may churn and
helping prioritize customers for retention analysis.
"""


# ============================================================
# CUSTOMER CONTEXT
# ============================================================

def create_customer_context(
    customer_id,
    churn_probability,
    churn_prediction,
    risk_level
):
    """
    Creates a strictly factual customer context.

    Only information actually available from the prediction
    system is included.
    """

    probability_percentage = (
        churn_probability * 100
    )

    prediction_label = (
        "Churn"
        if churn_prediction == 1
        else "No Churn"
    )

    return f"""
Customer ID:
{customer_id}

Model-estimated churn probability:
{churn_probability:.6f}

Model-estimated churn probability as percentage:
{probability_percentage:.2f}%

Predicted class:
{prediction_label}

Risk level:
{risk_level}

IMPORTANT:
The above information represents model output.
It does not establish the customer's actual future behavior,
specific reason for churn, financial value, dissatisfaction,
or preferred retention strategy.
"""


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    question,
    customer_context=None
):
    """
    Generate a grounded answer using Gemini.
    """

    if customer_context is None:
        customer_context = """
No customer-specific information was provided.
"""

    prompt = f"""
{SYSTEM_INSTRUCTION}

============================================================
PROJECT KNOWLEDGE
============================================================

{PROJECT_KNOWLEDGE}

============================================================
CUSTOMER INFORMATION
============================================================

{customer_context}

============================================================
USER QUESTION
============================================================

{question}

============================================================
ANSWER REQUIREMENTS
============================================================

Answer the user's question using the supplied information.

Be clear and concise.

Do not invent information.

If the available information is insufficient to answer something,
explicitly state that the available project/customer information
is not sufficient to determine it.

Remember that model predictions are estimates and not guarantees.
"""


    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TELECOM CHURN GEMINI LLM GENERATOR")
    print("=" * 70)

    print("\nGemini API key found.")

    print("\n" + "=" * 70)
    print("PROJECT KNOWLEDGE LOADED")
    print("=" * 70)

    print("\nGenerating answer...")


    # ========================================================
    # TEST QUESTION
    # ========================================================

    question = input(
        "\nEnter your question: "
    )


    # ========================================================
    # OPTIONAL CUSTOMER TEST
    # ========================================================

    use_customer = input(
        "\nIs this a customer-specific question? (y/n): "
    ).strip().lower()


    customer_context = None


    if use_customer == "y":

        print("\nEnter customer information:")

        customer_id = input(
            "Customer ID: "
        ).strip()

        churn_probability = float(
            input(
                "Churn probability (0-1): "
            )
        )

        churn_prediction = int(
            input(
                "Churn prediction (0 or 1): "
            )
        )

        risk_level = input(
            "Risk level: "
        ).strip()


        customer_context = create_customer_context(
            customer_id=customer_id,
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
            risk_level=risk_level
        )


    # ========================================================
    # GENERATE RESPONSE
    # ========================================================

    answer = generate_answer(
        question=question,
        customer_context=customer_context
    )


    # ========================================================
    # DISPLAY RESPONSE
    # ========================================================

    print("\n" + "=" * 70)
    print("GENERATED ANSWER")
    print("=" * 70)

    print()

    print(answer)

    print("\n" + "=" * 70)
    print("LLM GENERATION COMPLETED")
    print("=" * 70)
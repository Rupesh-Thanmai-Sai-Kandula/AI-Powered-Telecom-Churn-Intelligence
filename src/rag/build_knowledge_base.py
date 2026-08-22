from pathlib import Path
import json


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

KNOWLEDGE_BASE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PROJECT KNOWLEDGE
# ============================================================
#
# These documents summarize findings and decisions made during
# the development of the telecom churn prediction system.
#
# They will later be used by the RAG retrieval system to provide
# relevant context to the LLM.
# ============================================================

knowledge_documents = [

    # --------------------------------------------------------
    # 1. PROJECT OBJECTIVE
    # --------------------------------------------------------

    {
        "id": "project_objective",
        "title": "Telecom Churn Prediction Project Objective",
        "category": "project_overview",
        "content": """
The objective of the AI-Powered Telecom Churn Intelligence
project is to identify telecom customers who are likely to
churn and assign them a meaningful risk level.

The system uses historical customer behavior such as recharge
activity, revenue, ARPU, service usage, network usage, tenure,
and other customer attributes.

The machine learning component predicts the probability that
a customer will churn.

The final system is intended to go beyond simple churn
prediction by providing customer-specific risk analysis and
retention recommendations.
"""
    },


    # --------------------------------------------------------
    # 2. DATASET
    # --------------------------------------------------------

    {
        "id": "dataset_structure",
        "title": "Telecom Customer Dataset Structure",
        "category": "data",
        "content": """
The training dataset contains 2,154,048 customer records.

The target variable is churn.

The dataset contains customer attributes including:

user_id
region
tenure
montant
frequence_rech
revenue
arpu_segment
frequence
data_volume
on_net
orange
tigo
zone1
zone2
mrg
regularity
top_pack
freq_top_pack

The churn target contains two classes:

0 = No Churn
1 = Churn

Target distribution:

No Churn: 1,750,062 customers
Churn: 403,986 customers

Therefore, churn is the minority class in the dataset.
"""
    },


    # --------------------------------------------------------
    # 3. FEATURE ENGINEERING
    # --------------------------------------------------------

    {
        "id": "feature_engineering",
        "title": "Feature Engineering and Preprocessing",
        "category": "machine_learning",
        "content": """
The machine learning preprocessing pipeline separates
numerical and categorical features.

Numerical features are processed using:

1. Median imputation for missing values.
2. StandardScaler for numerical scaling.

Categorical features are processed using:

1. Most-frequent imputation.
2. One-hot encoding.

The OneHotEncoder uses handle_unknown='ignore' so that
previously unseen categorical values do not cause prediction
failures.

The original customer dataset contains 17 predictive features.

After preprocessing, the feature representation contains
172 processed features.

Sparse matrices are used to reduce memory consumption because
one-hot encoding produces many zero values.
"""
    },


    # --------------------------------------------------------
    # 4. MISSING DATA
    # --------------------------------------------------------

    {
        "id": "missing_data",
        "title": "Missing Data Findings",
        "category": "data_analysis",
        "content": """
The dataset contains substantial missing values in several
customer behavior variables.

Important missing-value counts identified during analysis
include:

zone2: 2,017,224
zone1: 1,984,327
tigo: 1,290,016
data_volume: 1,060,433
freq_top_pack: 902,594
top_pack: 902,594
orange: 895,248
region: 849,299
on_net: 786,675
montant: 756,739
frequence_rech: 756,739
frequence: 726,048
revenue: 726,048
arpu_segment: 726,048

Missing values were handled through the preprocessing
pipeline instead of deleting large numbers of customer
records.
"""
    },


    # --------------------------------------------------------
    # 5. RECHARGE BEHAVIOR
    # --------------------------------------------------------

    {
        "id": "recharge_behavior",
        "title": "Recharge Behavior Analysis",
        "category": "customer_behavior",
        "content": """
Recharge frequency was analyzed as an important customer
behavior variable.

The observed recharge-frequency groups were:

No Frequency Data:
726,048 customers

Very Low:
373,282 customers

Low:
375,201 customers

High:
343,015 customers

Very High:
336,502 customers

The analysis shows that a large portion of customers have
missing recharge-frequency information.

Recharge behavior can therefore be useful when interpreting
customer engagement and potential churn risk, but missing
values must be handled carefully.
"""
    },


    # --------------------------------------------------------
    # 6. REVENUE
    # --------------------------------------------------------

    {
        "id": "revenue_behavior",
        "title": "Revenue and ARPU Analysis",
        "category": "customer_behavior",
        "content": """
Revenue and ARPU were analyzed as indicators of customer
value and engagement.

Revenue groups included:

No Revenue Data:
726,048 customers

Low Revenue (<=1000):
375,757 customers

Medium Revenue (1001-3000):
340,308 customers

High Revenue (3001-7000):
338,058 customers

Very High Revenue (>7000):
373,877 customers

ARPU groups included:

No ARPU Data:
726,048 customers

Very Low ARPU:
375,757 customers

Low ARPU:
343,279 customers

High ARPU:
351,996 customers

Very High ARPU:
356,968 customers

Revenue and ARPU can be useful for distinguishing customer
value when interpreting churn risk.
"""
    },


    # --------------------------------------------------------
    # 7. DATA USAGE
    # --------------------------------------------------------

    {
        "id": "data_usage",
        "title": "Data Usage Analysis",
        "category": "customer_behavior",
        "content": """
Data usage was divided into the following groups:

No Data Usage:
1,060,433 customers

Zero Usage:
320,153 customers

Low Usage (1-1000):
348,436 customers

Medium Usage (1001-5000):
230,897 customers

High Usage (5001-20000):
160,136 customers

Very High Usage (>20000):
33,993 customers

Data usage provides an additional measure of customer
engagement with telecom services.
"""
    },


    # --------------------------------------------------------
    # 8. NETWORK USAGE
    # --------------------------------------------------------

    {
        "id": "network_usage",
        "title": "Network Usage Analysis",
        "category": "customer_behavior",
        "content": """
Network usage was analyzed across ON_NET, ORANGE, and TIGO.

ON_NET had:

High: 336,275
Low: 319,590
No Data: 786,675
Very High: 341,731
Very Low: 369,777

ORANGE had:

High: 308,880
Low: 318,596
No Data: 895,248
Very High: 313,014
Very Low: 318,310

TIGO had:

High: 210,101
Low: 159,928
No Data: 1,290,016
Very High: 215,038
Very Low: 278,965

Network usage can provide additional information about
customer engagement and service usage patterns.
"""
    },


    # --------------------------------------------------------
    # 9. TENURE
    # --------------------------------------------------------

    {
        "id": "tenure_behavior",
        "title": "Customer Tenure Analysis",
        "category": "customer_behavior",
        "content": """
Customer tenure was grouped into four categories:

Very Low (1-6):
575,033 customers

Low (7-24):
509,831 customers

High (25-51):
551,956 customers

Very High (52-62):
517,228 customers

Tenure represents the length of the customer's relationship
with the telecom service and can be considered when analyzing
customer churn behavior.
"""
    },


    # --------------------------------------------------------
    # 10. REGULARITY
    # --------------------------------------------------------

    {
        "id": "regularity_behavior",
        "title": "Customer Regularity Analysis",
        "category": "customer_behavior",
        "content": """
Regularity was analyzed using six observed values:

Regularity 0:
813,832 customers

Regularity 1:
487,627 customers

Regularity 2:
125,655 customers

Regularity 3:
127,857 customers

Regularity 4:
212,965 customers

Regularity 5:
386,112 customers

Regularity is therefore another behavioral feature that can
be used by the machine learning model when predicting churn.
"""
    },


    # --------------------------------------------------------
    # 11. MODEL COMPARISON
    # --------------------------------------------------------

    {
        "id": "model_comparison",
        "title": "Machine Learning Model Comparison",
        "category": "machine_learning",
        "content": """
Three classification models were evaluated:

Random Forest
Decision Tree
Logistic Regression

The initial model comparison produced:

Random Forest:
Accuracy = 0.8413
Precision = 0.5468
Recall = 0.8998
F1 = 0.6802
ROC-AUC = 0.9284

Decision Tree:
Accuracy = 0.8355
Precision = 0.5361
Recall = 0.9112
F1 = 0.6751
ROC-AUC = 0.9290

Logistic Regression:
Accuracy = 0.8372
Precision = 0.5396
Recall = 0.8974
F1 = 0.6740
ROC-AUC = 0.9275

Random Forest provided the best initial F1 score and was
selected for further tuning.
"""
    },


    # --------------------------------------------------------
    # 12. HYPERPARAMETER TUNING
    # --------------------------------------------------------

    {
        "id": "random_forest_tuning",
        "title": "Random Forest Hyperparameter Tuning",
        "category": "machine_learning",
        "content": """
Random Forest hyperparameters were tested to improve
classification performance.

The configurations evaluated were:

RF_1:
150 trees
Max depth = 15
Minimum samples leaf = 10
F1 = 0.6803

RF_2:
150 trees
Max depth = 20
Minimum samples leaf = 10
F1 = 0.6817

RF_3:
150 trees
Max depth = 15
Minimum samples leaf = 20
F1 = 0.6828

RF_4:
200 trees
Max depth = 20
Minimum samples leaf = 20
F1 = 0.6810

RF_3 produced the best F1 score and was selected as the
tuned Random Forest configuration.
"""
    },


    # --------------------------------------------------------
    # 13. THRESHOLD OPTIMIZATION
    # --------------------------------------------------------

    {
        "id": "threshold_optimization",
        "title": "Classification Threshold Optimization",
        "category": "machine_learning",
        "content": """
The default classification threshold of 0.50 was optimized
because the project focuses on identifying customers who may
churn.

The tuned Random Forest was evaluated at different thresholds.

The best F1 score was obtained at threshold 0.60.

At threshold 0.60:

Accuracy = 0.8545
Precision = 0.5753
Recall = 0.8560
F1 = 0.6881
ROC-AUC = 0.9281

The threshold of 0.60 was therefore selected for the final
customer churn prediction system.
"""
    },


    # --------------------------------------------------------
    # 14. FINAL MODEL
    # --------------------------------------------------------

    {
        "id": "final_model",
        "title": "Final Churn Prediction Model",
        "category": "machine_learning",
        "content": """
The final model is a tuned Random Forest classifier.

Configuration:

Trees = 150
Maximum depth = 15
Minimum samples per leaf = 20

Classification threshold = 0.60

Final validation performance:

Accuracy = 0.8545
Precision = 0.5753
Recall = 0.8560
F1 Score = 0.6881
ROC-AUC = 0.9281

The model is stored as:

models/tuned_random_forest.pkl

The preprocessing pipeline is stored as:

models/preprocessor.pkl
"""
    },


    # --------------------------------------------------------
    # 15. CONFUSION MATRIX
    # --------------------------------------------------------

    {
        "id": "confusion_matrix",
        "title": "Final Model Confusion Matrix",
        "category": "model_evaluation",
        "content": """
Using the final threshold of 0.60, the confusion matrix was:

Actual No Churn predicted No Churn:
298,952

Actual No Churn predicted Churn:
51,061

Actual Churn predicted No Churn:
11,635

Actual Churn predicted Churn:
69,162

The model correctly identified 69,162 actual churn customers
as churn while incorrectly classifying 11,635 actual churn
customers as No Churn.

The relatively high recall for the churn class is important
because missing customers who are likely to churn can reduce
the effectiveness of retention campaigns.
"""
    },


    # --------------------------------------------------------
    # 16. RISK CATEGORIES
    # --------------------------------------------------------

    {
        "id": "risk_categories",
        "title": "Customer Churn Risk Categories",
        "category": "risk_analysis",
        "content": """
The prediction system converts churn probability into five
risk categories:

Very Low
Low
Medium
High
Very High

The test prediction run generated results for 380,127
customers:

Very High:
73,390

High:
32,371

Medium:
25,162

Low:
46,417

Very Low:
202,787

The risk category provides a more understandable business
interpretation of the numerical churn probability.
"""
    },


    # --------------------------------------------------------
    # 17. FEATURE IMPORTANCE
    # --------------------------------------------------------

    {
        "id": "feature_importance",
        "title": "Model Feature Importance",
        "category": "model_interpretation",
        "content": """
Feature importance was extracted from the final Random Forest
model to understand which processed features contributed most
to the predictions.

The top processed features included:

Feature_27
Feature_11
Feature_0
Feature_2
Feature_171
Feature_3
Feature_6
Feature_1
Feature_4
Feature_13

Feature importance helps provide interpretability for the
machine learning model.

Because categorical variables were one-hot encoded, the model
works with processed feature representations rather than only
the original human-readable column names.
"""
    },


    # --------------------------------------------------------
    # 18. BUSINESS INTERPRETATION
    # --------------------------------------------------------

    {
        "id": "business_interpretation",
        "title": "Business Interpretation of Churn Risk",
        "category": "business_strategy",
        "content": """
The churn prediction system should be used to prioritize
customers for retention rather than treating every customer
equally.

A high churn probability indicates that the model identifies
the customer as having characteristics associated with churn.

It does not prove that the customer will definitely churn.

Similarly, feature importance indicates how the model uses
features for prediction. It does not automatically prove that
a feature directly causes churn.

Business decisions should therefore combine:

- Predicted churn probability
- Risk category
- Customer value
- Recharge behavior
- Revenue
- Usage behavior
- Other available customer information
"""
    },


    # --------------------------------------------------------
    # 19. RETENTION STRATEGY
    # --------------------------------------------------------

    {
        "id": "retention_strategy",
        "title": "Customer Retention Strategy",
        "category": "business_strategy",
        "content": """
Retention actions should be personalized according to the
customer's predicted risk and observed behavior.

Potential strategies include:

For high-risk customers:
- Personalized retention offers
- Recharge incentives
- Relevant voice or data bundles
- Loyalty benefits

For medium-risk customers:
- Engagement campaigns
- Personalized service recommendations
- Usage-based offers

For low-risk customers:
- Standard loyalty programs
- Normal engagement
- Cross-selling relevant services

The system should avoid making unsupported claims about why a
customer will churn.
"""
    },


    # --------------------------------------------------------
    # 20. RAG PURPOSE
    # --------------------------------------------------------

    {
        "id": "rag_purpose",
        "title": "Purpose of the RAG System",
        "category": "rag",
        "content": """
The Retrieval-Augmented Generation component will provide
relevant project knowledge to the language model when
analyzing individual customers.

The machine learning model provides:

- Churn probability
- Churn prediction
- Risk category

The RAG system provides relevant contextual knowledge such as:

- Customer behavior interpretations
- Model findings
- Feature information
- Retention strategies
- Business interpretation principles

The LLM can then use both the customer prediction and the
retrieved knowledge to generate an understandable customer
risk explanation and retention recommendation.

RAG is therefore used to ground the LLM's response in the
knowledge and findings of this specific telecom churn project.
"""
    }

]


# ============================================================
# SAVE KNOWLEDGE BASE
# ============================================================

print("=" * 70)
print("BUILDING TELECOM CHURN KNOWLEDGE BASE")
print("=" * 70)

knowledge_base_path = (
    KNOWLEDGE_BASE_DIR / "knowledge_base.json"
)

with open(
    knowledge_base_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        knowledge_documents,
        file,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("KNOWLEDGE BASE CREATED SUCCESSFULLY")
print("=" * 70)

print(
    f"\nDocuments created: "
    f"{len(knowledge_documents)}"
)

print(
    f"\nKnowledge base saved to:"
    f"\n{knowledge_base_path}"
)

print(
    "\nThe knowledge base contains findings from:"
    "\n- Data analysis"
    "\n- Feature engineering"
    "\n- Model comparison"
    "\n- Random Forest tuning"
    "\n- Threshold optimization"
    "\n- Model evaluation"
    "\n- Risk analysis"
    "\n- Feature importance"
    "\n- Business interpretation"
    "\n- Retention strategy"
)

print(
    "\nNext step: Build the RAG retrieval system."
)
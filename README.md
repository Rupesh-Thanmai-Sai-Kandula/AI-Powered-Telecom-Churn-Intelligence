# 📡 AI-Powered Telecom Churn Intelligence

An end-to-end Data Science and Generative AI project that predicts telecom customer churn, identifies high-risk customers, and generates AI-powered business insights using PostgreSQL, Python, Machine Learning, RAG, and Google Gemini.

🌐 Website: https://telecom-churn-risk-intelligence.streamlit.app/

---

# 🚀 Project Overview

Customer churn is a major challenge for telecom companies. Identifying customers who are likely to leave allows businesses to prioritize retention efforts and take proactive action.

This project builds a complete AI-powered telecom churn intelligence system, starting from customer data stored in PostgreSQL and ending with an interactive Streamlit application.

The project includes:

- PostgreSQL Database
- SQL Data Analysis
- Data Preprocessing
- Feature Engineering
- Machine Learning
- Random Forest Model Tuning
- Classification Threshold Optimization
- Customer Churn Prediction
- Customer Risk Classification
- RAG Knowledge Base
- Semantic Search
- Google Gemini LLM Integration
- AI Customer Risk Analysis
- Interactive Streamlit Dashboard

---

# 🏗 Project Architecture

```text
                    TELECOM CUSTOMER DATA
                              │
                              ▼
                    ┌─────────────────┐
                    │   PostgreSQL    │
                    │  expresso_churn │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   SQL Analysis  │
                    │ & Data Validation│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Data Processing │
                    │ & Feature       │
                    │ Engineering     │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Machine Learning│
                    │ Random Forest   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Hyperparameter  │
                    │    Tuning       │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │    Threshold    │
                    │ Optimization    │
                    │     = 0.60      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Churn Prediction│
                    │ & Risk Analysis │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐           ┌─────────────────┐
     │ RAG Knowledge   │           │ Customer Risk   │
     │ Base            │           │ Analysis        │
     └────────┬────────┘           └─────────────────┘
              │
              ▼
     ┌─────────────────┐
     │ Semantic Search │
     │ Embeddings      │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │ Google Gemini   │
     │ LLM             │
     └────────┬────────┘
              │
              ▼
     ┌─────────────────┐
     │   Streamlit     │
     │    Dashboard    │
     └─────────────────┘
```

---

# 📂 Project Structure

```text
AI-Powered-Telecom-Churn-Intelligence/

├── data/
│   └── knowledge_base/
│       ├── knowledge_base.json
│       └── knowledge_embeddings.npy
│
├── models/
│   ├── tuned_random_forest.pkl
│   └── preprocessor.pkl
│
├── reports/
│   └── customer_churn_predictions.csv
│
├── sql/
│   └── SQL analysis scripts
│
├── src/
│   ├── database/
│   ├── ml/
│   │   ├── train_model.py
│   │   ├── predict_customers.py
│   │   └── ...
│   │
│   ├── rag/
│   │   ├── build_knowledge_base.py
│   │   ├── semantic_retriever.py
│   │   ├── rag_engine.py
│   │   └── llm_generator.py
│   │
│   └── dashboard/
│       └── streamlit_dashboard.py
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

# 🗄 Database

The project uses PostgreSQL as the primary database.

Database:

**expresso_churn**

PostgreSQL was used for:

- Data storage
- Data validation
- SQL analysis
- Customer-level analysis
- Exploratory analysis
- Feature investigation
- Preparing data for the Machine Learning pipeline

SQL analysis forms the initial data-analysis layer of the project before the data is passed into the ML pipeline.

---

# 🛠 Technologies Used

## Programming

- Python
- SQL

## Database

- PostgreSQL
- SQLAlchemy
- psycopg2

## Data Science

- Pandas
- NumPy
- Scikit-learn

## Machine Learning

- Random Forest Classifier
- Hyperparameter Tuning
- Classification Threshold Optimization
- Model Evaluation

## Generative AI

- Retrieval-Augmented Generation (RAG)
- Sentence Transformers
- Semantic Search
- Google Gemini API
- Prompt Engineering

## Dashboard

- Streamlit

## Version Control

- Git
- GitHub

---

# 🤖 Machine Learning Pipeline

The Machine Learning pipeline was designed to predict whether a telecom customer is likely to churn.

The pipeline includes:

1. Data loading from PostgreSQL
2. Data preprocessing
3. Missing-value handling
4. Feature engineering
5. Categorical feature encoding
6. Model training
7. Model comparison
8. Random Forest tuning
9. Threshold optimization
10. Customer-level prediction
11. Risk classification

The final trained model is saved using Joblib and reused during prediction instead of retraining the model.

---

# 🌲 Final Machine Learning Model

The final prediction model is a:

**Tuned Random Forest Classifier**

The model was selected after evaluating and tuning the machine learning pipeline.

The final classification threshold was optimized rather than simply using the default `0.50`.

---

# 🎯 Classification Threshold Optimization

The project focuses on identifying customers who are likely to churn.

The default classification threshold of:

```text
0.50
```

was evaluated across different threshold values.

The best F1 score was obtained at:

```text
0.60
```

Therefore, the final customer churn prediction system uses:

```text
Classification Threshold = 0.60
```

### Final Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.8545 |
| Precision | 0.5753 |
| Recall | 0.8560 |
| F1 Score | 0.6881 |
| ROC-AUC | 0.9281 |

The threshold of `0.60` provided the best F1 score during threshold optimization.

---

# 📊 Customer Churn Prediction

The final prediction pipeline generated churn predictions for:

```text
380,127 customers
```

The prediction output contains information including:

- Customer ID
- Churn probability
- Churn prediction
- Risk level

The generated predictions are stored in:

```text
reports/customer_churn_predictions.csv
```

---

# ⚠️ Customer Risk Classification

The system converts the predicted churn probability into five business-friendly risk categories:

```text
Very Low
Low
Medium
High
Very High
```

The latest prediction run produced:

| Risk Level | Customers |
|---|---:|
| Very High | 73,390 |
| High | 32,371 |
| Medium | 25,162 |
| Low | 46,417 |
| Very Low | 202,787 |

This allows business users to prioritize customers rather than working with raw probability values alone.

---

# 🧠 Retrieval-Augmented Generation

The project contains a domain-specific RAG system built from the findings generated throughout the Data Science and Machine Learning workflow.

The knowledge base contains:

```text
20 documents
```

The knowledge base covers:

- Data analysis
- Feature engineering
- Model comparison
- Random Forest tuning
- Threshold optimization
- Model evaluation
- Risk analysis
- Feature importance
- Business interpretation
- Retention strategy

The knowledge base is stored in:

```text
data/knowledge_base/knowledge_base.json
```

---

# 🔎 Semantic Retrieval

The RAG system uses:

```text
all-MiniLM-L6-v2
```

to generate semantic embeddings.

The resulting embedding matrix has the shape:

```text
20 × 384
```

When a user asks a question, the system:

```text
User Question
      │
      ▼
Query Embedding
      │
      ▼
Semantic Similarity Search
      │
      ▼
Top Relevant Documents
      │
      ▼
Retrieved Project Knowledge
```

This allows the system to retrieve project-specific information instead of relying only on a general-purpose LLM.

---

# ✨ Google Gemini LLM Integration

Google Gemini is used as the Generative AI layer of the project.

The retrieved RAG context is provided to Gemini along with instructions to:

- Use the project knowledge as the primary source
- Avoid inventing statistics
- Preserve actual project metrics
- Explain technical concepts in business-friendly language
- Clearly state limitations when sufficient information is unavailable

The overall architecture is:

```text
User Question
      │
      ▼
Semantic Retriever
      │
      ▼
Relevant Knowledge
      │
      ▼
RAG Context
      │
      ▼
Gemini LLM
      │
      ▼
Business-Friendly Answer
```

---

# 🤖 AI Customer Risk Analysis

The Streamlit application also provides AI-powered analysis for individual customers.

For example, a user can select a high-risk customer and ask the AI system to explain:

- What the predicted risk means
- Why the customer should be prioritized
- What business actions could be considered
- What limitations exist in the available customer information

The system combines:

```text
Customer Prediction
        +
RAG Knowledge
        +
Gemini LLM
        ↓
AI Risk Analysis
```

The AI does not treat a churn prediction as a guarantee that the customer will leave.

---

# 📊 Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

The dashboard contains multiple sections.

## Executive Dashboard

Displays:

- Total Customers
- Predicted Churn
- Predicted No Churn
- Predicted Churn Rate
- Customer Risk Distribution
- Model Performance
- Model Configuration

---

## Customer Risk Explorer

Users can:

- Filter customers by risk level
- Set minimum churn probability
- View high-risk customers
- Sort customers by churn probability
- Inspect individual customer predictions

---

## AI Risk Analysis

The dashboard allows users to obtain AI-generated explanations for customer risk.

The analysis combines the Machine Learning prediction with the RAG knowledge base and Gemini.

---

# 📈 Model Evaluation

The final model achieved:

```text
Accuracy  : 0.8545
Precision : 0.5753
Recall    : 0.8560
F1 Score  : 0.6881
ROC-AUC   : 0.9281
```

The relatively high recall is particularly relevant to the project's churn-detection objective because failing to identify a customer who actually churns can reduce the effectiveness of retention efforts.

---

# 🗃️ Important Project Outputs

## Trained Machine Learning Model

```text
models/tuned_random_forest.pkl
```

## Preprocessing Pipeline

```text
models/preprocessor.pkl
```

## Customer Predictions

```text
reports/customer_churn_predictions.csv
```

## RAG Knowledge Base

```text
data/knowledge_base/knowledge_base.json
```

## Document Embeddings

```text
data/knowledge_base/knowledge_embeddings.npy
```

---

# 🔐 API Key Security

The Gemini API key is not stored directly in the source code.

The application reads the API key using:

```text
GEMINI_API_KEY
```

For local development, the environment variable can be configured in PowerShell.

For deployment, the API key should be stored using the hosting platform's Secrets configuration.

**Never commit your Gemini API key to GitHub.**

---

# ⚙ Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Powered-Telecom-Churn-Intelligence.git
```

## 2. Move into the project

```bash
cd AI-Powered-Telecom-Churn-Intelligence
```

## 3. Create a virtual environment

```bash
python -m venv venv
```

## 4. Activate the virtual environment

### Windows

```powershell
venv\Scripts\activate
```

## 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Set your Gemini API key:

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Then start the Streamlit application:

```powershell
python -m streamlit run src/dashboard/streamlit_dashboard.py
```

The application will open in your browser at the local Streamlit address displayed in the terminal.

---

# 🌐 Deployment

The application is designed to be deployed as a Streamlit web application.

Deployment architecture:

```text
GitHub Repository
        │
        ▼
Streamlit Cloud
        │
        ├── Application Code
        ├── ML Models
        ├── RAG Knowledge Base
        └── Gemini API Secret
                │
                ▼
        Live Web Application
```

The Gemini API key should be configured through Streamlit's secret-management system rather than being committed to the repository.

---

# ⚠️ Limitations

The churn probability is a model prediction, not a guarantee of future customer behavior.

The system also cannot automatically determine the exact reason why a customer will churn unless the relevant explanatory customer information is available in the underlying data.

AI-generated business recommendations should therefore be treated as decision support, rather than automatic business decisions.

---

# 🎯 Project Objective

The ultimate objective of this project is to transform telecom customer data into an AI-powered decision-support system capable of:

1. Predicting customer churn
2. Quantifying churn probability
3. Categorizing customer risk
4. Prioritizing high-risk customers
5. Retrieving relevant project knowledge
6. Generating AI-powered explanations
7. Supporting proactive customer-retention decisions

---

# 👨‍💻 Author

**Rupesh Kandula**

Aspiring Data Scientist | Python | SQL | PostgreSQL | Machine Learning | RAG | Generative AI

GitHub:

https://github.com/Rupesh-Thanmai-Sai-Kandula

LinkedIn:

https://www.linkedin.com/in/rupesh-thanmai-sai-kandula-a01452288/

---

# ⭐ Project Highlights

```text
PostgreSQL
     ↓
SQL Analytics
     ↓
Data Science
     ↓
Random Forest
     ↓
Threshold Optimization
     ↓
380K+ Customer Predictions
     ↓
Risk Intelligence
     ↓
RAG
     ↓
Semantic Search
     ↓
Gemini LLM
     ↓
Streamlit
     ↓
Live AI-Powered Telecom Churn Intelligence Platform
```

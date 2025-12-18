Credit Card Default Risk Prediction

This project predicts whether a credit card customer is likely to default on payment in the next month, using historical billing, payment, and repayment behavior.

The model is trained on the UCI Credit Card Default dataset and demonstrated using a Gradio-based interactive interface built directly inside a Jupyter/Colab notebook.

Problem Statement

Banks need to identify customers who are at risk of defaulting on their credit card payments in advance.
This enables:

Risk control

Credit limit adjustment

Early intervention

The objective of this project is to build a binary classification model that outputs:

Probability of default

Risk category (Low / Moderate / High / Extreme)

Dataset
Target Column
default.payment.next.month


0 → No default

1 → Default

Input Features Used for Training

Demographic & Credit Information

LIMIT_BAL
SEX
EDUCATION
MARRIAGE
AGE


Repayment Status

PAY_0
PAY_2
PAY_3
PAY_4
PAY_5
PAY_6


Billing & Payment History

BILL_AMT1 – BILL_AMT6
PAY_AMT1 – PAY_AMT6


Only these raw features are used during prediction.
No engineered features are manually entered by the user.

Feature Preprocessing

All preprocessing is handled inside an sklearn Pipeline, ensuring consistency and no data leakage.

Invalid EDUCATION values (0, 5, 6) are mapped to "Others"

Invalid MARRIAGE value (0) is mapped to "Others"

Skewed numeric features are handled using power transformations

Extreme values are handled using percentile-based clipping

Categorical variables are encoded

No data leakage between training and inference

Model

Binary classification model trained using scikit-learn

Outputs probability of default

Risk categories derived using probability thresholds

Risk Categorization
Probability < 0.25   → Low Risk
0.25 – 0.50          → Moderate Risk
0.50 – 0.75          → High Risk
> 0.75               → Extreme Risk

Prediction Interface

A Gradio-based interface is created inside the notebook, allowing users to:

Enter customer credit information

Get default probability

View risk classification instantly

The interface strictly matches the features used during model training.

How to Run
1️⃣ Install Dependencies
pip install -r requirements.txt


Recommended Gradio version

gradio==3.50.2

2️⃣ Run the Notebook

Open the notebook

Run all cells sequentially

The Gradio interface will launch at the end of the notebook

Project Structure
.
├── notebook.ipynb         # Data analysis, model training, and Gradio interface
├── model.pkl              # Trained sklearn pipeline
├── preprocessing.py       # Custom transformers (skewness, outliers)
├── requirements.txt
└── README.md

Key Learnings

Aligning training features with inference inputs

Handling skewed financial data

Avoiding target leakage in preprocessing

Building end-to-end ML pipelines

Deploying ML models using interactive interfaces

Future Improvements

Retrain model using engineered behavioral features

Add SHAP-based explanations

Convert notebook interface into a standalone web app

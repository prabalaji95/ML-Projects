# 📊 AI/ML School Projects — UT Austin Post Graduate Certificate

This repository contains data science and machine learning projects completed as part of the **Post Graduate Certificate in AI & ML at UT Austin McCombs School of Business**. Each project applies real-world datasets to solve business problems using Python-based analytical and modeling techniques.

---

## 📁 Projects

### 1. 🏦 AllLife Bank — Personal Loan Campaign Optimization
**File:** `AllLifeBank_PersonalLoan_ModelSelection.py`

#### Overview
AllLife Bank wants to grow its loan portfolio by converting existing liability customers (depositors) into personal loan customers. A previous pilot campaign achieved a 9% conversion rate. This project builds a predictive classification model to identify which customers are most likely to accept a personal loan offer, enabling more targeted and cost-effective marketing campaigns.

#### Objectives
- Identify demographic and behavioral drivers of personal loan adoption
- Build and compare Decision Tree classifiers (default, pre-pruning, post-pruning)
- Optimize for recall to minimize missed loan conversions (false negatives)
- Deliver actionable segmentation insights for the retail marketing team

#### Methodology
- Exploratory Data Analysis (EDA): univariate and bivariate analysis
- Data preprocessing: anomaly correction (negative experience values), ZIP code feature engineering, categorical encoding
- Model building: sklearn `DecisionTreeClassifier` with default settings, pre-pruning via hyperparameter tuning, and post-pruning via cost-complexity pruning (`ccp_alpha`)
- Model evaluation: accuracy, precision, recall, F1-score, and confusion matrix comparison across train/test sets

#### Key Techniques
- `DecisionTreeClassifier` with class weighting to handle class imbalance
- Cost-complexity pruning path analysis
- Feature importance via Gini impurity
- Model performance comparison across three model variants

#### Tech Stack
`Python` · `pandas` · `NumPy` · `scikit-learn` · `matplotlib` · `seaborn`

---

### 2. 🍕 FoodHub — Food Delivery Data Analysis
**File:** `FoodHub_DataAnalysis.py`

#### Overview
FoodHub is a New York-based food aggregator app that connects customers with restaurants via a delivery platform. This project analyzes order data to uncover demand patterns, delivery performance, and customer behavior — helping the business improve customer experience and operational efficiency.

#### Objectives
- Understand the distribution of orders across cuisine types, days of the week, and cost ranges
- Analyze food preparation and delivery times across different restaurant and cuisine categories
- Identify top-performing restaurants eligible for promotional offers
- Calculate net platform revenue based on a tiered commission structure
- Assess delivery time SLAs and weekend vs. weekday performance

#### Key Questions Answered
- What percentage of orders cost more than $20? *(~29.24%)*
- What is the mean order delivery time? *(~24.16 minutes)*
- Which restaurants qualify for promotional offers (rating count > 50 and avg rating > 4)?
- What is the net revenue generated across all orders?
- What percentage of orders take more than 60 minutes end-to-end?
- How does mean delivery time differ on weekdays vs. weekends?

#### Key Techniques
- Univariate analysis: histograms, boxplots, countplots for all key variables
- Multivariate analysis: cuisine vs. cost, cuisine vs. prep time, day vs. delivery time, rating vs. delivery/prep/cost
- Correlation heatmap across numerical features
- Revenue computation using a custom tiered function via `.apply()`
- Customer segmentation by order frequency (top 5 most frequent customers)

#### Tech Stack
`Python` · `pandas` · `NumPy` · `matplotlib` · `seaborn`

---

## 🛠️ Setup & Usage

Both scripts were originally developed in **Google Colab** and use Google Drive for data loading. To run locally:

1. Clone this repository
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install dependencies
   ```bash
   pip install numpy==2.0.2 pandas==2.2.2 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1 sklearn-pandas==2.2.0
   ```

3. Update the data file paths in each script to point to your local dataset location (replace the `drive.mount` and `pd.read_csv` paths accordingly)

---

## 📂 Datasets

| Project | Dataset | Source |
|---|---|---|
| AllLife Bank | `Loan_Modelling.csv` | Provided as part of UT Austin AI/ML coursework |
| FoodHub | `foodhub_order.csv` | Provided as part of UT Austin AI/ML coursework |

---

## 👤 Author

**Pranav Balaji** — Data Analyst | AI/ML Graduate Student at UT Austin McCombs
*https://prabalaji95.github.io/*
*Passionate about applying machine learning to solve real business problems in finance, operations, and consumer analytics.*

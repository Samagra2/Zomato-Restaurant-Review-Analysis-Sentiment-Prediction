
# 🍽️ Zomato Restaurant Review Analysis & Sentiment Prediction

> **End-to-End NLP & Machine Learning Project for Sentiment Analysis in Food-Tech Domain**

**Author:** Samagra Gupta
**Organization:** Labmentix
**Domain:** Data Science | Machine Learning | Natural Language Processing

---

## 📌 Project Overview

Online food-tech platforms like **Zomato** receive millions of customer reviews daily. These reviews contain valuable insights about food quality, service standards, and overall customer satisfaction. However, due to the **unstructured nature of textual data and its large scale**, manual analysis is inefficient and impractical.

This project implements an **end-to-end NLP and Machine Learning pipeline** to analyze restaurant reviews and **predict customer sentiment (Positive / Negative)**. The solution transforms raw customer feedback into **actionable business insights** that can help restaurants improve performance and platforms enhance user experience.

---

## 🎯 Objectives

* Perform **exploratory data analysis (EDA)** on Zomato review data
* Clean and preprocess unstructured textual data
* Convert text into numerical features using **TF-IDF**
* Build and evaluate multiple **Machine Learning models**
* Predict customer sentiment from reviews
* Interpret results with **business relevance**
* Design a **scalable deployment-ready architecture**

---

## 📂 Dataset Description

Two datasets are used in this project:

### 1️⃣ Zomato Restaurant Metadata

Contains structured information such as:

* Restaurant names
* Identifiers
* Attributes for aggregation and analysis

### 2️⃣ Zomato Restaurant Reviews

Contains:

* Customer review text (unstructured)
* Ratings
* Contextual information

📌 **Target Variable:**
Customer **Sentiment** derived from ratings and textual polarity.

---

## 🔍 Exploratory Data Analysis (EDA)

Key analyses performed:

* Rating distribution analysis
* Review length distribution
* Sentiment class balance
* Restaurant-wise average ratings
* Review frequency per restaurant
* Correlation analysis using heatmaps

📊 These analyses help uncover:

* Customer behavior patterns
* Restaurant performance trends
* Data imbalance and modeling challenges

---

## 🧹 Data Preprocessing & Feature Engineering

### Text Cleaning Pipeline

* Lowercasing
* Removal of punctuation & noise
* Stopword elimination
* Tokenization

### Feature Extraction

* **TF-IDF Vectorization** used to convert text into numerical features
* Ensures meaningful word weighting and dimensionality control

✔ These steps significantly improve model performance and generalization.

---

## 📊 Hypothesis Testing & Statistical Insights

**Null Hypothesis (H₀):**
There is no significant relationship between customer ratings and review sentiment.

**Alternate Hypothesis (H₁):**
A significant relationship exists.

📌 Statistical testing confirms a **strong alignment between ratings and sentiment**, validating the modeling approach.

---

## 🤖 Machine Learning Models

### Models Implemented

* Naive Bayes (Baseline Model)
* Logistic Regression (Final Selected Model)

### Why Logistic Regression?

* Strong performance for text classification
* Interpretable coefficients
* Computationally efficient
* Robust under class imbalance

### Training Strategy

* Train-test split
* Cross-validation
* Hyperparameter tuning using **GridSearchCV**

---

## 📈 Model Evaluation

### Evaluation Metrics Used

* Accuracy
* Precision
* Recall
* F1-Score

📊 Visualizations included:

* Confusion Matrix
* Evaluation Heatmap
* Model comparison charts

✔ Final model achieved a **high F1-score**, making it suitable for real-world deployment.

---

## 🏆 Final Results & Business Impact

### Business Value

**For Customers**

* Improved review transparency
* Better restaurant recommendations

**For Restaurants**

* Actionable feedback on service & food quality
* Identification of operational issues

**For Zomato Platform**

* Automated sentiment monitoring
* Trust & quality assurance
* Scalable analytics solution

---

## 🏗️ Deployment Architecture

The project is designed to be **deployment-ready**, supporting real-time sentiment prediction via APIs and dashboards.

**Pipeline:**
Data Ingestion → NLP Processing → ML Model → API → Dashboard / Alerts

---

## 🧪 Technologies & Tools Used

* **Python**
* **Pandas, NumPy**
* **NLTK**
* **Scikit-learn**
* **Matplotlib, Seaborn**
* **TF-IDF Vectorizer**
* **Jupyter Notebook**
* **PlantUML (Architecture Design)**

---

## 📁 Project Structure

```bash
├── data/
│   ├── Zomato Restaurant reviews.csv
│   ├── Zomato Restaurant names and Metadata.csv
│
├── notebooks/
│   ├── Sample_ML_Submission_TemplateFinal.ipynb
│
├── reports/
│   ├── FINAL_Insights_Report_Zomato_Samagra_Gupta_Labmentix.pdf
│
├── README.md
```

---

## 🚀 Future Enhancements

* Deep Learning models (LSTM / BERT)
* Aspect-based sentiment analysis
* Multilingual review support
* Real-time dashboards
* REST API deployment

---

## ✅ Conclusion

This project demonstrates a **complete, industry-ready NLP & ML solution** for sentiment analysis in the food-tech domain. It highlights how unstructured textual data can be transformed into meaningful insights that drive **better customer experience and business decisions**.

---


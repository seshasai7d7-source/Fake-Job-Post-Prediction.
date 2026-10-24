# Fake Job Posting Prediction

A machine learning project that uses Natural Language Processing (NLP) and classification techniques to detect whether a job posting is real or fake.

## Project Overview

With the rise of online job portals, fraudulent job postings have become a serious issue. Many scam postings look legitimate at first glance, but they are designed to deceive applicants into sharing personal information or paying fake fees. This project aims to build a predictive model that can classify job advertisements as either genuine or fraudulent based on their content and metadata.

The model is trained on a publicly available dataset from Kaggle and uses text processing combined with classification algorithms to improve detection accuracy.

## Problem Statement

The goal of this project is to build a classifier that can analyze job descriptions and related information to determine whether a job posting is real or fake. The system should help recruiters, platforms, and applicants reduce exposure to scam-based employment listings.

## Objectives

- Identify fake job postings using NLP and machine learning.
- Analyze textual and categorical job attributes.
- Compare different classification models.
- Select the best-performing model based on accuracy and F1-score.
- Build a robust solution for real-world fake job detection.

## Dataset

This project uses the Fake Job Posting dataset from Kaggle.

Dataset details:
- Total records: 17,880
- Features: job title, location, company profile, description, requirements, benefits, employment type, and other metadata
- Target variable: fraudulent (0 = real, 1 = fake)

The dataset is highly imbalanced:
- Most postings are real
- A smaller subset are fake
- This imbalance makes model evaluation more important, especially for precision and recall

## Tech Stack

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn

## Project Workflow

1. Data collection
2. Data cleaning and preprocessing
3. Exploratory data analysis
4. Feature engineering
5. Text preprocessing
6. Model training
7. Model evaluation
8. Result interpretation

## Data Preprocessing

Text data was cleaned and transformed before model training. The following steps were performed:

- Tokenization
- Lowercasing
- Stopword removal
- Lemmatization
- Conversion of text into numerical representations using feature extraction techniques

For the final model, features were separated into:
- Textual features
- Numeric/categorical features
- Target label

## Models Used

The project compares multiple classification models, including:

- Naive Bayes
- SGD Classifier

Naive Bayes serves as the baseline model, while SGD is compared to evaluate whether the performance improves.

## Evaluation Metrics

The model was evaluated using:

- Accuracy
- F1-score

These metrics are important because the dataset is imbalanced, and both false positives and false negatives matter in real-world fake job detection.

## Model Results

The following results were observed:

| Model | Accuracy | F1-score |
| Naive Bayes | 0.971 | 0.743 |
| SGD Classifier | 0.974 | 0.790 |

Based on the results, the SGD Classifier performed better than the baseline model and was selected as the final model.

## File Structure

```bash
Fake-Job-Posting-Prediction/
│
├── data/
│   └── dataset files
├── notebooks/
│   └── analysis and model training notebooks
├── images/
│   └── plots and visualizations
├── README.md
├── requirements.txt
└── model_training.py

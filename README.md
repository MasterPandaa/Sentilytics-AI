# Sentilytics-AI — Automated NLP Sentiment Analytics & Model Training Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?logo=python&logoColor=white)](#)
[![Django](https://img.shields.io/badge/Django-5.x-092E20.svg?logo=django&logoColor=white)](#)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-AutoML%20Engine-F7931E.svg?logo=scikit-learn&logoColor=white)](#)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Mining-150458.svg?logo=pandas&logoColor=white)](#)
[![Explainable AI](https://img.shields.io/badge/XAI-Narrative%20Insights-8B5CF6.svg)](#)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#)

A web platform built on **Django 5** for automated natural language processing (NLP), dynamic machine learning model training, and customer review sentiment analytics.

Sentilytics-AI allows data teams and analysts to upload customer feedback CSV datasets, automatically extract TF-IDF text features, train `RandomForest` classifiers on the fly, inspect performance metrics and confusion matrices, and export serialized `.pkl` pipeline bundles for production deployment.

---

## Overview

Analyzing qualitative customer feedback across multiple review sets often requires repetitive data preparation and notebook workflows. Sentilytics-AI operationalizes this pipeline into a web interface:

- **Automated Text Preprocessing & Labeling**: Sanitizes review text with regex normalization and maps rating scales into three-class targets (Negative, Neutral, Positive).
- **Dynamic Model Training**: Extracts n-gram TF-IDF vectors and fits a `RandomForestClassifier` with configurable sample sizes and estimator counts.
- **Evaluation & Diagnostics**: Renders classification reports (Accuracy, Precision, Recall, F1-Score) alongside server-generated Confusion Matrix heatmaps.
- **Top Feature Attribution**: Identifies the top predictive n-grams influencing sentiment decisions.
- **Pipeline Artifact Serialization**: Packages the fitted vectorizer and classifier into a downloadable `.pkl` file ready for microservice inference.

---

## Technical Features

1. **Text Sanitization & Ingestion**: Ingests multi-column CSVs, handles missing values, and processes Indonesian/English textual reviews.
2. **On-the-Fly Model Training**: Fits a Scikit-Learn pipeline (`TfidfVectorizer` + `RandomForestClassifier`) directly in the Django request lifecycle.
3. **In-Memory Confusion Matrix Heatmap**: Generates high-resolution Seaborn matrix plots directly in memory, encoded as Base64 for instant browser rendering without persistent image clutter.
4. **Multi-Dataset Benchmarking**: Comparative sentiment distribution charts comparing multiple product or app datasets side by side.
5. **Downloadable Pipeline Bundle**: Exports serialized pipeline objects for integration with external APIs or batch workers.

---

## Installation & Setup

### Prerequisites

- Python 3.9+
- pip & virtualenv

### Quickstart

1. **Clone the repository**:
   ```bash
   git clone https://github.com/MasterPandaa/Sentilytics-AI.git
   cd Sentilytics-AI
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run migrations and start the Django server**:
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```
   Open `http://127.0.0.1:8000` in your browser.

---

## Project Structure

```text
Sentilytics-AI/
├── manage.py                   # Django CLI entry point
├── requirements.txt            # Python dependencies
├── sentilytics/                # Django project settings & URL routing
├── analytics/                  # Core application (ML training, views, templates)
│   ├── views.py                # Pipeline execution & chart generation
│   ├── ml_pipeline.py          # TF-IDF & Random Forest training logic
│   └── templates/              # Dashboard and report views
└── README.md                   # Project documentation
```

---

## License

This project is licensed under the [MIT License](LICENSE).

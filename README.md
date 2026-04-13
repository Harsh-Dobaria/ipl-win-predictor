# 🏏 IPL Win Probability Predictor

## Overview

This project predicts the probability of a team winning an IPL match at any given point using machine learning. It takes real-time match conditions such as score, overs, and wickets, and outputs a win probability along with an explanation of the prediction.

---

## Problem Statement

Given the current match situation:

* Current score
* Target score
* Overs completed
* Wickets lost

Predict the likelihood of the chasing team winning the match.

---

## Key Features

* 🎯 Real-time win probability prediction
* 📊 SHAP-based explanation for each prediction
* ⚡ Interactive Streamlit web app
* 🧠 Feature engineering capturing match dynamics
* 📉 Handles pressure, run rate, and wickets impact

---

## Feature Engineering

The model uses match-state features such as:

* Runs remaining
* Balls remaining
* Wickets remaining
* Current Run Rate (CRR)
* Required Run Rate (RRR)
* Pressure (RRR - CRR)
* Runs per wicket
* Match phase (Powerplay / Middle / Death)

---

## Model

* **XGBoost Classifier**
* Selected after comparison with Logistic Regression
* Captures non-linear relationships in match situations

---

## Performance

* ROC-AUC Score: ~0.88
* Cross-validated for generalization

---

## Model Explainability

SHAP (SHapley Additive Explanations) is used to:

* Explain individual predictions
* Identify most influential features
* Provide transparency into model decisions

---

## Key Insights

* Win probability decreases as required run rate increases
* Wickets remaining is the most critical factor in late innings
* Sudden drops in probability indicate key match turning points
* Even with fewer runs remaining, low wickets drastically reduce chances

---

## Streamlit App

The project includes an interactive web application where users can:

1. Enter match details
2. Get win probability instantly
3. View explanation of prediction

---

## How to Run

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn xgboost shap streamlit matplotlib
```

---

### 2. Run the app

```bash
python -m streamlit run app.py
```

---


---

## Project Structure

```
IPL/
├── app.py          # Streamlit app
├── model.pkl       # Trained model
├── README.md       # Project documentation
├── requirements.txt     
```

---

## Limitations

* Does not consider player-specific performance
* Ignores pitch conditions and external factors
* Based on historical match patterns

---

## Future Improvements

* Add player-level data
* Incorporate live match APIs
* Improve UI/UX design
* Add match progression visualization

---

## Conclusion

This project demonstrates how machine learning can be applied to real-time sports analytics by combining feature engineering, predictive modeling, and interpretability.

---

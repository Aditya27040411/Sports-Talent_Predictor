# 🏅 SportTalent AI

### AI-Powered Sports Aptitude Prediction for Children (Ages 5–10)

> Helping children discover the sports they are naturally suited for through machine learning and sports science.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🚀 Project Overview

Choosing the right sport early can significantly impact a child's long-term athletic development.

Most children enter sports through:

* Parent preference
* School availability
* Trial and error

This often results in:

* Early dropout
* Poor performance
* Missed athletic potential

**SportTalent AI** uses machine learning and research-based athletic profiles to recommend the sports a child is most likely to excel in based on measurable physical and cognitive attributes.

---

## 🎯 What This Project Does

Given a child's assessment scores, the model predicts:

🥇 Best-fit sport

🥈 Second-best sport

🥉 Third-best sport

along with confidence scores and explainable reasoning.

### Example Prediction

Input:

| Attribute   | Score |
| ----------- | ----- |
| Speed       | 8.2   |
| Strength    | 6.7   |
| Endurance   | 9.1   |
| Flexibility | 5.3   |
| Cognitive   | 7.4   |
| Reflex      | 6.8   |

Output:

```text
🏅 Top 3 Sport Recommendations

1. Long Distance Running (82%)
2. Football (11%)
3. Basketball (5%)
```

---

## 🧠 Features

✅ Sports aptitude prediction

✅ Random Forest, SVM and Ensemble models

✅ Explainable AI using SHAP

✅ Synthetic data generation pipeline

✅ GAN-based augmentation

✅ Jupyter notebook experimentation

✅ Unit-tested ML workflow

---

## 🏗️ System Architecture

```text
Child Assessment
        ↓
Feature Extraction
        ↓
Data Preprocessing
        ↓
Machine Learning Model
        ↓
Probability Ranking
        ↓
Top-3 Sport Recommendations
```

---

## 📊 Model Performance

| Model            | Accuracy | Macro F1 |
| ---------------- | -------- | -------- |
| Random Forest    | 91%      | 0.90     |
| SVM              | 88%      | 0.87     |
| Stacked Ensemble | 93%      | 0.92     |

### Key Observation

The stacked ensemble consistently outperformed individual models and provided the most balanced predictions across all sport categories.

---

## 🧬 Dataset

### Dataset Summary

| Metric   | Value                      |
| -------- | -------------------------- |
| Records  | 542                        |
| Sports   | 7                          |
| Features | 6                          |
| Classes  | Balanced                   |
| Source   | Synthetic + Research-Based |

### Supported Sports

🏀 Basketball

⚽ Football

🎾 Tennis

🤸 Gymnastics

🏃 Long Distance Running

🤼 Wrestling

♟️ Chess

---

## 🔬 Scientific Foundation

The sport profiles were constructed using findings from sports science literature covering:

* Physical performance requirements
* Energy system demands
* Reaction speed requirements
* Cognitive complexity
* Movement patterns

Gaussian noise and CTGAN augmentation were used to simulate realistic variation between athletes.

---

## 🛠️ Tech Stack

* Python
* Scikit-Learn
* Pandas
* NumPy
* Matplotlib
* SHAP
* CTGAN
* Jupyter Notebook
* PyTest

---

## 📁 Repository Structure

```text
src/
├── preprocess.py
├── train.py
├── evaluate.py
├── predict.py
└── gan_augment.py

tests/
notebooks/
results/
models/
data/
```

---

## 🛣️ Future Work

* Real-world pilot study (30–50 children)
* Mobile application
* Parent dashboard
* Coach recommendation engine
* Indian language support
* Wearable integration
* Muscle fiber profile integration

---

## ⚠️ Limitations

* Current dataset is synthetic
* Requires real-world validation
* Designed as a decision-support system, not a final verdict
* Environmental and socioeconomic factors are not yet modeled

---

## 👨‍💻 Author

**Aditya Paliwal**

B.Tech Computer Science

Machine Learning • Sports Analytics • Applied AI

---

### ⭐ If you found this project interesting, consider giving the repository a star.





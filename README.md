# Dropout Risk Prediction

This project provides a model to predict the likelihood of a user dropping out based on their app usage behavior.

## Setup Instructions

### 1. Clone or Download the Repository

-`main.ipynb` — Full training pipeline (feature engineering, model training, evaluation).

-`dropout_gb_model.pkl` — The saved trained model.

-`predict_dropout.py` — Script to interactively predict dropout.

-`dropout_risk_report.md` or `.pdf` — Risk analysis report.

-`requirements.txt` — Python dependencies.

---

2.**Install requirements**

    install dependencies:

```bash

pip install -r requirements.txt

```

3.**Run prediction script**

   From the terminal:

```bash

python predict_dropout.py

```

   You’ll be prompted to enter:

- Average sessions per week
- Average session duration (minutes)
- Goal check-in rate (between 0 and 1)

  The script will output:
- Whether the user is likely to **Stay** or **Drop out**
- Their **dropout risk score** (between 0 and 1)

---

4.**retrain the model**

If you want to modify or retrain the model:

- Open **main.ipynb** in Jupyter Notebook or any compatible environment (e.g., Google Colab).
- Run all cells sequentially.

## 👤 Author

**Zaynap Ahmad**

_Machine Learning Track – August 2025_

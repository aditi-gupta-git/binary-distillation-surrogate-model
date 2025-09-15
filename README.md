Here you go — full **README.md** in one block so you can copy at once:

```markdown
# AI/ML Surrogate Modelling for Binary Distillation

This project implements machine learning surrogate models to predict **Distillate Purity (xD)** and **Reboiler Duty (QR)** for an Ethanol–Water binary distillation column. Data was generated using the **DWSIM** process simulator, and multiple ML models were compared for accuracy, robustness, and generalization.

---

## 📂 Project Structure

```

AI\_Distillation\_Surrogate/
│── distill\_data.csv        # Dataset generated from DWSIM
│── EDA.ipynb               # Exploratory Data Analysis
│── ModelTraining.ipynb     # Model training & evaluation
│── train.py                # Training script (exported from notebook)
│── evaluate.py             # Evaluation & plotting script
│── plots/                  # Data visualisation & evaluation plots
│── Report.pdf              # Detailed report (3–5 pages)
│── requirements.txt        # Python dependencies
│── README.md               # Instructions (this file)

````

---

## ⚙️ Setup

1. Clone/Download the repository or unzip:
   ```bash
   unzip AI_Distillation_Surrogate.zip
   cd AI_Distillation_Surrogate
````

2. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

### Run Training

```bash
python train.py
```

* Trains multiple ML models (Linear, Polynomial, Random Forest, AdaBoost, SVR, XGBoost).
* Saves trained models and metrics.

### Run Evaluation

```bash
python evaluate.py
```

* Generates metrics, residual plots, and parity plots.
* Results are saved in the `plots/` folder.

---

## 📊 Models Implemented

* Linear Regression (baseline)
* Polynomial Regression
* Random Forest Regressor
* AdaBoost Regressor
* Support Vector Regressor
* XGBoost

✅ **Best Performer:** XGBoost (R² > 0.9, lowest RMSE for both xD & QR).
Ensemble methods (XGBoost, Random Forest) showed the strongest generalization and lowest error.

---

## 📑 Report

The full methodology and results are documented in **Report.pdf**, including:

* Flowsheet & variable ranges
* Data generation protocol (DWSIM)
* Data cleaning & preprocessing
* Model training & hyperparameter tuning
* Results, plots & diagnostics
* Conclusions and key insights

---

## 👤 Author

Prepared as part of the **Autumn Internship Screening Task**:
**AI/ML Surrogate Modelling for Binary Distillation**

```
```

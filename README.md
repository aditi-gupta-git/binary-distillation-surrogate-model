# AI/ML Surrogate Modelling for Binary Distillation
Machine Learning Project to develop ML surrogate models to approximate the performance of a binary distillation column. Process data is generated using DWSIM, an open-source process simulator, by systematically varying key operating parameters to predict Distillate Purity and Reboiler Duty.

---

## 📌 Project Overview
This project applies **AI/ML surrogate modelling** to replace computationally expensive process simulations (DWSIM) with lightweight regression models.  
Using the dataset generated from DWSIM simulations, multiple ML methods are trained and compared to capture the relationship between column operating conditions and output performance.

> 🎯 **Goal:** Predict separation efficiency (xD) and energy demand (QR) accurately, and identify which operating conditions impact them most.

---

## 📊 Dataset Information
- **Source:** Self-generated using **DWSIM** process simulator  
- **File Used:** `distill_data.csv`  
- **Size:** 373 rows × 8 columns  

### KEY INPUT FEATURES
| Column Name         | Description                                          |
|---------------------|------------------------------------------------------|
| `Reflux_Ratio` (R)  | Ratio of liquid returned to distillate               |
| `Boilup_Ratio` (B)  | Vapor generated in reboiler / distillate flow        |
| `Feed_MoleFraction` | Ethanol mole fraction in feed (0.2–0.95)             |
| `Feed_FlowRate` (F) | Total feed flowrate (±30% variation)                 |
| `Number_of_Stages`  | Theoretical trays in the column (15–25)              |
| `Feed_ThermalCond`  | Feed condition (0 = subcooled, 1 = saturated)        |

### OUTPUT VARIABLES
- `Distillate_MoleFraction (xD)` → Separation purity  
- `Reboiler_Duty (QR)` → Energy requirement in kW  

---

## 🔬 Methods Used

- **Data Cleaning & Preprocessing**
  - Removed 66 duplicate rows (final dataset: 307 samples × 8 features)  
  - Checked for missing values (none found)  
  - Standard scaling applied to continuous features  
  - Encoding performed for categorical feature (`Feed_ThermalCondition`)  
  - Verified unit consistency for flowrates (kmol/h) and duties (kW)  

- **Exploratory Data Analysis (EDA)**
  - **Univariate analysis:** Histograms & Boxplots for feature distributions and outlier detection  
  - **Bivariate analysis:** Scatterplots (xD & QR vs inputs) to identify correlations  
  - **Multivariate analysis:** Correlation Heatmaps & Pairplots to capture feature interactions  
  - **Key findings:**  
    - Distillate purity (xD) strongly correlates with **Reflux Ratio** & **Feed Mole Fraction**  
    - Reboiler duty (QR) strongly correlates with **Feed Flowrate**  

- **Model Building**
  - **Linear Regression** → weak baseline, high bias  
  - **Polynomial Regression** → improved fit but overfitting observed  
  - **Random Forest Regressor** → robust, accurate, interpretable feature importance  
  - **AdaBoost Regressor** → decent accuracy, moderate error  
  - **Support Vector Regressor (SVR)** → high accuracy, but computationally expensive  
  - **XGBoost** → best performer (R² > 0.9, lowest RMSE for both xD & QR)  

- **Hyperparameter Tuning**
  - Used **RandomizedSearchCV** for ensemble models (RF, AdaBoost, XGBoost)  
  - Tuned parameters: number of estimators, tree depth, learning rate, etc.  
  - Incorporated into **pipelines** with preprocessing steps  

- **Model Evaluation**
  - Metrics: **R²** (goodness of fit), **RMSE** (prediction error)  
  - Diagnostic plots:  
    - Residual plots → check variance & bias  
    - Parity plots → predicted vs actual (RF/XGBoost aligned closely with diagonal)  
    - Error distribution plots → ensembles showed narrow, centered errors  


---

## 🚀 Results & Insights
- **Best performer:** **XGBoost**  
  - Best performer on validation: XGBoost (R² ≈ 0.92, RMSE ≈ 35.8).  
  - Lowest RMSE, robust generalization  
- Ensemble methods (XGBoost, Random Forest) consistently outperformed simpler models.  
- Key predictors:
  - xD strongly influenced by **Reflux Ratio** & **Feed Mole Fraction**  
  - QR strongly influenced by **Boilup Ratio** & **Feed Flowrate**  

---

## 📈 Use-Case
- **Process Engineers:** Rapid estimation of distillation column performance without running full simulations.  
- **Optimization:** Enables faster design studies by integrating ML models into optimization loops.  
- **Energy–Purity Tradeoff:** Helps visualize how operating conditions impact both purity and energy demand.  

---

## ⚙️ Requirements
Install dependencies with:
```bash
pip install -r requirements.txt

## Instructions
- Extract files from **AI_Distillation_Surrogate.zip**
- Open zip file in VS Code
- **data** folder consisists of **distill_data.csv**
- **EDA** and **ModelTraining** consists of the code.
- **Report.docx** - Report should be opened using Word
- **README.md** can be accessed in the folder.
- **requirements.txt** attached to install dependencies
- **Data Visualisation** and **Evaluation** folders contains EDA plots and Parity and Residual Plots

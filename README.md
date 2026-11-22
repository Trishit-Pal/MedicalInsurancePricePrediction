# Medical Insurance Price Prediction

A beginner-friendly machine learning project that predicts yearly medical insurance charges from basic personal and health details.

## Live Demo

Try the web app here: `https://medical-insurance-price-predictionml.streamlit.app/` 

## Overview

Insurance companies estimate medical costs using factors like age, BMI, family size, and smoking habits. In this project, we explore the data, clean outliers, train five regression models, compare their scores, and save the final model for future predictions.

## Dataset

The dataset (`insurance.csv`) has **1,338 rows** and **7 columns** with no missing values:

| Column | Description | Example / Range |
| :--- | :--- | :--- |
| `age` | Age of the person | 18 to 64 |
| `sex` | Gender | `male`, `female` |
| `bmi` | Body Mass Index | 15.96 to 53.13 |
| `children` | Number of children covered | 0 to 5 |
| `smoker` | Whether the person smokes | `yes`, `no` |
| `region` | Residential area in the US | `northwest`, `northeast`, `southeast`, `southwest` |
| `charges` | Medical insurance cost in USD (target) | $1,121.87 to $63,770.43 |

## Workflow

1. **Data Exploration (EDA):** Checked data types, summary statistics, missing values, and plotted charts to see how each column affects `charges`.
2. **Data Cleaning:** Removed 1 duplicate row and capped extreme `bmi` outliers using the IQR rule (`feature-engine`).
3. **Encoding:** Converted text columns (`sex`, `smoker`, `region`) into numbers so machine learning models can read them.
4. **Model Training & Tuning:** Trained 5 regression models on an 80/20 train-test split and tuned tree models with `GridSearchCV`.
5. **Feature Selection:** Dropped `sex` and `region` because they contributed less than 1% to predictions, keeping `age`, `bmi`, `children`, and `smoker`.
6. **Saving the Model:** Saved the final XGBoost model to `insurancemodelf.pkl`.

## Model Results

All five models were compared using the **R² score** (`1.0` = perfect prediction, `0.0` = no better than guessing the average):

| Model | Train R² | Test R² | 5-Fold CV R² |
| :--- | :---: | :---: | :---: |
| **Gradient Boosting (Tuned)** | 0.871 | 0.902 | 0.861 |
| **XGBoost (Tuned)** | 0.881 | 0.900 | 0.858 |
| **Random Forest** | 0.974 | 0.881 | 0.837 |
| **Linear Regression** | 0.730 | 0.806 | 0.747 |
| **SVR (Unscaled)** | -0.102 | -0.134 | -0.104 |

**Final Saved Model (`insurancemodelf.pkl`):** Tuned `XGBRegressor` using 4 features (`age`, `bmi`, `children`, `smoker`) — **Test R²: 0.896**, **5-Fold CV R²: 0.859**.

## Key Takeaways

- **Smoking matters most:** `smoker` accounts for about **84%** of the model's feature importance. Smokers pay much higher charges than non-smokers.
- **BMI and age come next:** Higher BMI (especially among smokers) and older age steadily increase medical costs.
- **Simpler inputs work just as well:** Removing `sex` and `region` kept accuracy almost identical while making the final model simpler.

## Project Files

```text
MedicalInsurancePricePrediction/
├── insurance.csv         # Raw dataset
├── MLHandsOn.ipynb       # Full EDA, preprocessing, training, and evaluation notebook
├── insurancemodelf.pkl   # Saved final XGBoost model
├── app.py                # Streamlit web app
├── requirements.txt      # Python packages needed to run the project
└── README.md             # Project documentation
```

## How to Run Locally

1. **Clone this repository:**
   ```bash
   git clone <your-repo-url>
   cd MedicalInsurancePricePrediction
   ```

2. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Open the notebook:**
   ```bash
   jupyter notebook MLHandsOn.ipynb
   ```

4. **Or start the web app:**
   ```bash
   streamlit run app.py
   ```

## Deploy for Free (Streamlit Community Cloud)

1. Push this project to a **public** GitHub repository.
2. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub and click **Create app**.
3. Pick your repo, branch `main`, and main file `app.py`.
4. Click **Deploy**, then paste the live link into the **Live Demo** section above.

## Quick Prediction Example

Note that in this project `smoker` is encoded as `0` for **yes** and `1` for **no**:

```python
import pickle
import pandas as pd

model = pickle.load(open("insurancemodelf.pkl", "rb"))

sample = pd.DataFrame([{
    "age": 30,
    "bmi": 28.0,
    "children": 1,
    "smoker": 1,  # 0 = yes, 1 = no
}])

predicted_cost = model.predict(sample)[0]
print(f"Estimated insurance cost: ${predicted_cost:,.2f}")
```

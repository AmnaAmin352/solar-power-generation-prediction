# Solar Power Generation Prediction

A machine learning project that predicts solar power (AC) output using weather sensor data, and deploys the trained model as a live Streamlit web app.

## 📊 Dataset
- Source: Kaggle Solar Power Generation Data
- Combined data from 2 solar plants: generation data + weather sensor data (ambient temperature, module temperature, irradiation)
- Merged on timestamp for each plant

## 🛠️ Tools & Libraries
- Python, Pandas, NumPy
- Scikit-learn (Gradient Boosting Regressor)
- Matplotlib, Seaborn
- Streamlit (for deployment)

## 🔍 Project Workflow
1. Merged generation and weather data for both plants
2. Engineered cyclical time features (`HOUR_SIN`, `HOUR_COS`) to capture time-of-day patterns
3. Feature scaling
4. Trained a baseline Gradient Boosting Regressor
5. Hyperparameter tuning with GridSearchCV
6. Evaluated using RMSE, MAE, and R²
7. Saved the final model and built an interactive Streamlit prediction app

## 📈 Results

| Metric | Baseline | Tuned (Final) |
|---|---|---|
| R² | 0.841 | **0.844** |
| RMSE | 151.72 | 150.50 |
| MAE | 55.30 | 54.06 |

**Best hyperparameters:** `learning_rate=0.1, max_depth=3, n_estimators=200, subsample=1.0`

**Feature importance:** Irradiation was by far the most predictive feature (~95%), followed by Plant ID (~3.8%)  confirming that solar irradiance is the dominant driver of power output.

## 📁 Files
- `solar_power_generation_prediction.ipynb`  main notebook (data merging, EDA, feature engineering, model training & tuning)
- `app.py` — Streamlit app for live predictions using the trained model
- Dataset files (Plant 1 & 2 generation + weather sensor data)

## 🚀 How to Run

**Notebook:**
1. Clone this repository
2. `pip install pandas numpy scikit-learn matplotlib seaborn joblib`
3. Open the notebook in Jupyter and run all cells (this generates `gbr_solar_model.pkl` and `scaler.pkl`)

**Streamlit app:**
1. Make sure `gbr_solar_model.pkl` and `scaler.pkl` are in the same folder as `app.py`
2. `pip install streamlit joblib pandas numpy`
3. Run: `streamlit run app.py`

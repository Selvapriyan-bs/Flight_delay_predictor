
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                              AdaBoostRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# =============================================================
# 1️ DATA LOADING & INITIAL OVERVIEW
# =============================================================
print("📥 Loading dataset...")
df = pd.read_csv("B:/Project/AIML/flight_delay_no_history.csv")
print(f" Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Display top records
print(df.head(5))

# =============================================================
# 2️ BASIC CLEANING & DUPLICATES
# =============================================================
df.columns = df.columns.str.strip().str.replace(' ', '_')
df.drop_duplicates(inplace=True)
print(f" Duplicates removed. New shape: {df.shape}")

# =============================================================
# 3️ HANDLE MISSING VALUES SMARTLY
# =============================================================
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)
print(" Missing values imputed using median/mode.")

# =============================================================
# 4️ FEATURE ENGINEERING (DATE, TIME)
# =============================================================
if 'Scheduled_Departure' in df.columns:
    df['Scheduled_Departure'] = pd.to_datetime(df['Scheduled_Departure'], errors='coerce')
    df['Dep_Hour'] = df['Scheduled_Departure'].dt.hour
    df['Dep_Day'] = df['Scheduled_Departure'].dt.day
    df['Dep_Month'] = df['Scheduled_Departure'].dt.month
    df['Dep_Weekday'] = df['Scheduled_Departure'].dt.weekday
    df.drop(['Scheduled_Departure'], axis=1, inplace=True)
    print(" Date-time features extracted successfully!")

# =============================================================
# 5️ RARE CATEGORY HANDLING
# =============================================================
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < 0.02].index
    df[col] = df[col].replace(rare, 'Other')
print(" Rare labels grouped under 'Other'.")

# =============================================================
# 6️ OUTLIER HANDLING (Z-SCORE)
# =============================================================
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    z = np.abs(stats.zscore(df[col]))
    df = df[(z < 3)]
print(" Outliers handled using Z-Score filtering.")

# =============================================================
# 7️ SKEWNESS REDUCTION (LOG/POWER TRANSFORM)
# =============================================================
for col in num_cols:
    if df[col].skew() > 1:
        df[col] = np.log1p(df[col] - df[col].min() + 1)
print(" Skewed features normalized with log transform.")

# =============================================================
# 8️ CORRELATION & MULTICOLLINEARITY CONTROL
# =============================================================
corr_matrix = df[num_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
drop_corr = [column for column in upper.columns if any(upper[column] > 0.9)]
df.drop(columns=drop_corr, inplace=True)

X_vif = df[num_cols].dropna()._get_numeric_data()
vif_data = pd.DataFrame()
vif_data["feature"] = X_vif.columns
vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(len(X_vif.columns))]
drop_vif = vif_data[vif_data["VIF"] > 10]["feature"].tolist()
df.drop(columns=drop_vif, inplace=True)
print(f" Correlated features removed: {drop_corr + drop_vif}")

# =============================================================
# 9️ DEFINE FEATURES AND TARGET
# =============================================================
X = df.drop(['Flight_No', 'Delay'], axis=1, errors='ignore')
y = df['Delay']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

print(f"🎯 Features: {len(X.columns)}, Target: Delay")

# =============================================================
# 10️ SPLIT DATA
# =============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f" Data Split -> Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================
# 1️1️ DEFINE MULTIPLE PREPROCESSORS
# =============================================================
preprocessor_std = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('var', VarianceThreshold(0.0))
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

preprocessor_robust = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

preprocessor_power = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', PowerTransformer())
    ]), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

# =============================================================
# 1️2️ DEFINE MULTIPLE MODELS (> 4)
# =============================================================
models = {
    "RandomForest": (RandomForestRegressor(n_estimators=250, random_state=42, n_jobs=-1), preprocessor_robust),
    "GradientBoosting": (GradientBoostingRegressor(n_estimators=200, random_state=42), preprocessor_std),
    "AdaBoost": (AdaBoostRegressor(n_estimators=200, random_state=42), preprocessor_std),
    "Ridge": (Ridge(alpha=1.0), preprocessor_std),
    "Lasso": (Lasso(alpha=0.001), preprocessor_std),
    "SVR": (SVR(kernel='rbf', C=1.0, epsilon=0.2), preprocessor_power),
    "KNN": (KNeighborsRegressor(n_neighbors=7), preprocessor_power)
}

# # Optional: Use XGBoost if installed
# try:
#     from xgboost import XGBRegressor
#     models["XGBoost"] = (XGBRegressor(n_estimators=250, learning_rate=0.05, random_state=42, n_jobs=-1), preprocessor_robust)
# except:
#     print(" XGBoost not installed. Skipping that model.")

# =============================================================
# 1️3️ TRAIN EACH MODEL SEPARATELY
# =============================================================
results = []
predictions = pd.DataFrame(index=y_test.index)

for name, (model, preprocessor) in models.items():
    print(f"\n Training model: {name}")
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    predictions[name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, f"models/{name}_model.pkl")

    print(f" {name} done | R²={r2:.3f}, RMSE={rmse:.3f}")

# =============================================================
# 1️4️ META STACKING MODEL
# =============================================================
meta_X = predictions
meta_y = y_test

meta_model = GradientBoostingRegressor(n_estimators=200, random_state=42)
meta_model.fit(meta_X, meta_y)
final_pred = meta_model.predict(meta_X)

meta_mae = mean_absolute_error(meta_y, final_pred)
meta_rmse = np.sqrt(mean_squared_error(meta_y, final_pred))
meta_r2 = r2_score(meta_y, final_pred)

results.append({"Model": "Final_Stacked_Ensemble", "MAE": meta_mae, "RMSE": meta_rmse, "R2": meta_r2})
joblib.dump(meta_model, "models/final_stacked_ensemble.pkl")

# =============================================================
# 1️5️ PERFORMANCE SUMMARY
# =============================================================
results_df = pd.DataFrame(results).sort_values(by='R2', ascending=False)
print("\nMODEL PERFORMANCE COMPARISON")
print(results_df)

plt.figure(figsize=(12,6))
sns.barplot(x='Model', y='R2', data=results_df)
plt.title("Model R² Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

results_df.to_csv("models/model_performance_summary.csv", index=False)
print(" Results saved in models/model_performance_summary.csv")

# =============================================================
# 1️6️ FEATURE IMPORTANCE AGGREGATION
# =============================================================
try:
    feature_importance = {}
    for name, (model, preprocessor) in models.items():
        if hasattr(model, 'feature_importances_'):
            pipeline = joblib.load(f"models/{name}_model.pkl")
            feat_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
            importances = model.feature_importances_
            temp = pd.Series(importances, index=feat_names)
            feature_importance[name] = temp.sort_values(ascending=False).head(10)
    if feature_importance:
        print("\n Top 10 Important Features Across Models:")
        for k, v in feature_importance.items():
            print(f"\n{k}:\n{v}")
except:
    print(" Some models do not provide feature_importances_.")

# =============================================================
# 1️7️ FINAL EXPLANABILITY (Simplified SHAP-like)
# =============================================================
try:
    import shap
    print("\n Running SHAP Explainability on Final Meta Model...")
    explainer = shap.Explainer(meta_model, meta_X)
    shap_values = explainer(meta_X)
    shap.summary_plot(shap_values, meta_X)
except:
    print(" SHAP not installed, skipping explainability visualization.")

# =============================================================
# 1️8️ SAMPLE PREDICTION TEST
# =============================================================
print("\n Sample Prediction Check:")
sample = X_test.sample(1, random_state=42)
print(sample)
sample_preds = {}
for name, (model, preprocessor) in models.items():
    pipe = joblib.load(f"models/{name}_model.pkl")
    sample_preds[name] = pipe.predict(sample)[0]
sample_preds["Final_Stacked_Ensemble"] = meta_model.predict(pd.DataFrame([sample_preds]))[0]
print("\nPredictions from All Models:")
print(sample_preds)


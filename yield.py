import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ==========================================
# 1. LOAD THE DATASET
# ==========================================
# Note: 'yield_df.csv' is the combined dataset integrating yield, pesticides, rainfall, and temp files.
file_name = '/Users/ranjith/Desktop/RECOMMENDATION  ALALYSIS/yield_df.csv'
df = pd.read_csv(file_name)

# Drop redundant index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print('--- Dataset Info ---')
print(f'Dataset Shape: {df.shape}')
print(df.head(3))

# ==========================================
# 2. DATA PREPROCESSING & ENCODING
# ==========================================
target_column = 'hg/ha_yield'

X = df.drop(columns=[target_column])
y = df[target_column]

# Categorical columns to encode
categorical_cols = ['Area', 'Item']

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
feature_columns = X_encoded.columns

# Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# ==========================================
# 3. FEATURE SCALING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 4. DECISION TREE REGRESSOR TRAINING
# ==========================================
dt_regressor = DecisionTreeRegressor(max_depth=15, random_state=42)
dt_regressor.fit(X_train_scaled, y_train)

# ==========================================
# 5. MODEL EVALUATION & PERFORMANCE
# ==========================================
y_pred = dt_regressor.predict(X_test_scaled)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print('\n--- Model Performance Metrics ---')
print(f'R² Score: {r2:.4f}')
print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
print(f'Mean Absolute Error (MAE): {mae:.2f}')

# ==========================================
# 6. VISUALIZATIONS (BAR GRAPHS)
# ==========================================
# Figure 1: Top 10 Feature Importances Bar Graph
feature_importances = pd.Series(
    dt_regressor.feature_importances_, index=feature_columns
).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
feature_importances.head(10).plot(kind='barh', color='forestgreen')
plt.title(
    'Top 10 Feature Importances in Crop Yield Prediction',
    fontsize=14,
    fontweight='bold',
)
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Figure 2: Average Crop Yield by Top Items Bar Graph
plt.figure(figsize=(10, 5))
top_crops = df.groupby('Item')['hg/ha_yield'].mean().sort_values(ascending=False).head(10)
top_crops.plot(kind='bar', color='royalblue', edgecolor='black')
plt.title(
    'Top 10 Crops by Average Yield (hg/ha)', fontsize=14, fontweight='bold'
)
plt.xlabel('Crop Item', fontsize=12)
plt.ylabel('Average Yield (hg/ha)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ==========================================
# 7. SAMPLE PREDICTION
# ==========================================
sample_data = {
    'Year': 2013,
    'average_rain_fall_mm_per_year': 1485.0,
    'pesticides_tonnes': 121.0,
    'avg_temp': 16.37,
    'Area': 'Albania',
    'Item': 'Maize',
}

sample_df = pd.DataFrame([sample_data])
sample_df_encoded = pd.get_dummies(sample_df, columns=['Area', 'Item'])
sample_df_encoded = sample_df_encoded.reindex(
    columns=feature_columns, fill_value=0
)
sample_scaled = scaler.transform(sample_df_encoded)

predicted_yield = dt_regressor.predict(sample_scaled)[0]

print('\n--- Sample Prediction Result ---')
print(f"Input Sample: {sample_data}")
print(
    f'Predicted Crop Yield: 🌾 {predicted_yield:,.2f} hectograms/hectare'
)

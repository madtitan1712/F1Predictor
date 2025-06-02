import DataClean as cleandata
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

print("Starting F1 Race Winner Prediction Model Training...")

# Load and clean data
CleanedData = cleandata.getcleanedData()
print(f"Cleaned data shape: {CleanedData.shape}")

# Select features and target
features = ["Driver_enc", "Team_enc", "Track_enc", "Grid", "FastestLap", "AverageTime", "pitstops", "DNF"]
print(f"Using features: {features}")

x = CleanedData[features]
y = CleanedData["Win"]  # Predict if the driver won (1) or not (0)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Training set: {x_train.shape[0]} samples, Test set: {x_test.shape[0]} samples")

# Remove rows with missing labels (if any)
mask = y_train.notna()
x_train = x_train[mask]
y_train = y_train[mask]

# Initialize classifier
model = XGBClassifier(
    use_label_encoder=False, 
    eval_metric="logloss",
    learning_rate=0.05,
    n_estimators=200,
    max_depth=5
)

print("Training model...")
# Train model
model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    verbose=True
)

# Predict on test set
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

# Calculate accuracy and other metrics
accuracy = (y_pred == y_test).mean()
print(f"Model accuracy: {accuracy:.4f}")

# Feature importance
feature_importance = model.feature_importances_
features_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
for i, row in features_df.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(features_df['Feature'], features_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("Feature importance plot saved as 'feature_importance.png'")

# Save the model
with open("f1_prediction_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as 'f1_prediction_model.pkl'")
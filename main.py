import DataClean as cleandata
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

# Load and clean data
CleanedData = cleandata.getcleanedData()

# Select features and target
features = ["Driver_enc", "Team_enc", "Grid", "FastestLap", "AverageTime", "pitstops", "DNF"]
x = CleanedData[features]
y = CleanedData["Win"]  # Predict if the driver won (1) or not (0)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Remove rows with missing labels (if any)
mask = y_train.notna()
x_train = x_train[mask]
y_train = y_train[mask]

# Initialize classifier (disable label encoder warning)
model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")

# Train model
model.fit(x_train, y_train)

# Predict on test set
y_pred = model.predict(x_test)
y_pred_proba = model.predict_proba(x_test)[:, 1]

print("Predicted classes (Win=1):", y_pred)
print("Predicted winning probabilities:", y_pred_proba)

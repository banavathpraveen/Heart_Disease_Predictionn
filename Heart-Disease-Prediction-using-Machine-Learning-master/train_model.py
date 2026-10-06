import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
data = pd.read_csv("heart.csv")

# Split into features (X) and target (y)
X = data.drop('target', axis=1)
y = data['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------
# 1️⃣ Random Forest
# -------------------------------
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
joblib.dump(rf_model, "rf_model.pkl")
print("✅ rf_model.pkl saved")

# -------------------------------
# 2️⃣ AdaBoost
# -------------------------------
ab_model = AdaBoostClassifier()
ab_model.fit(X_train, y_train)
joblib.dump(ab_model, "ab_model.pkl")
print("✅ ab_model.pkl saved")

# -------------------------------
# 3️⃣ XGBoost
# -------------------------------
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, "xgb_model.pkl")
print("✅ xgb_model.pkl saved")

# -------------------------------
# 4️⃣ Neural Network
# -------------------------------
nn_model = Sequential([
    Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])
nn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
nn_model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0)
nn_model.save("nn_model.h5")
print("✅ nn_model.h5 saved")

print("\n🎉 All model files have been created successfully!")

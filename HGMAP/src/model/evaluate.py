from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import pandas as pd
import joblib
import config.constants as constants

#load model and data
stacked_model = joblib.load("stacked_model.pkl")
data = pd.read_csv("processed_microbiome_data.csv")
X_test = data.drop(columns=[constants.target_column])  # Replace with actual target column
y_test = data["target_column"]

#predictions
y_pred = stacked_model.predict(X_test)
y_prob = stacked_model.predict_proba(X_test)[:, 1]

#metrics
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob)}")

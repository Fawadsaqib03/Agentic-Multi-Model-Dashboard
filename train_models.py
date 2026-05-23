import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

print("Training Worker Models...")

# CHURN PREDICTION (Neural Network)
df_churn = pd.read_csv("data/Churn_Modelling.csv")
df_churn.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
df_churn = pd.get_dummies(df_churn, columns=['Geography', 'Gender'], drop_first=True)
X_ch = df_churn.drop('Exited', axis=1)
y_ch = df_churn['Exited']

scaler_ch = StandardScaler()
X_ch_sc = scaler_ch.fit_transform(X_ch)
nn_model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
nn_model.fit(X_ch_sc, y_ch)

with open("models/churn_model.pkl", "wb") as f:
    pickle.dump({"model": nn_model, "scaler": scaler_ch}, f)

# DIABETES PREDICTION (Logistic Regression)
df_diab = pd.read_csv("data/diabetes.csv")
X_di = df_diab.drop('Outcome', axis=1)
y_di = df_diab['Outcome']

scaler_di = StandardScaler()
X_di_sc = scaler_di.fit_transform(X_di)
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(X_di_sc, y_di)

with open("models/diabetes_model.pkl", "wb") as f:
    pickle.dump({"model": lr_model, "scaler": scaler_di}, f)

# SPAM DETECTION (SVM)
df_spam = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df_spam.columns = ['label', 'text']
y_sp = df_spam['label'].map({'ham': 0, 'spam': 1})
X_sp = df_spam['text']

vectorizer = TfidfVectorizer(max_features=3000)
X_sp_vec = vectorizer.fit_transform(X_sp)
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_sp_vec, y_sp)

with open("models/spam_model.pkl", "wb") as f:
    pickle.dump({"model": svm_model, "vectorizer": vectorizer}, f)

print("[DONE] All Worker Models Serialized to /models/*.pkl")
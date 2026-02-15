import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Placement Prediction App")

# -------------------------
# Load Models
# -------------------------
# models = {
#     "Decision Tree":  joblib.load("models/decision_tree.pkl"),
#     "Gaussian Naive Bayes":  joblib.load("models/gaussian_n_b.pkl"),
#     "K-Nearest Neighbor":  joblib.load("models/knn.pkl"),
#     "Logistic Regression": joblib.load("models/logistic.pkl"),
#     "Random Forest": joblib.load("models/random_forest.pkl"),
#     "XGBoost": joblib.load("models/xgboost.pkl")
# }

try:
    models =  {
        "Decision Tree":  joblib.load("models/decision_tree.pkl"),
        "Gaussian Naive Bayes":  joblib.load("models/gaussian_n_b.pkl"),
        "K-Nearest Neighbor":  joblib.load("models/knn.pkl"),
        "Logistic Regression": joblib.load("models/logistic.pkl"),
        "Random Forest": joblib.load("models/random_forest.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl")
    }
    st.success("Model Loaded Successfully")
except Exception as e:
    st.error(f"Model failed to load: {e}")
    st.stop()

# -------------------------
# a. Dataset Upload Option
# -------------------------
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(data.head())

    columns_to_drop = ['Student_ID', 'placement_status', 'salary_lpa']

    X = data.drop(columns=[col for col in columns_to_drop if col in data.columns])

    # X = data.drop(columns=['Student_ID', 'placement_status', 'salary_lpa'])

    # Assuming target column is named 'target'
    # X = data.drop("placement_status", axis=1)
    y = data['placement_status'].apply(lambda x: 1 if x == 'Placed' else 0)



    # -------------------------
    # b. Model Selection Dropdown
    # -------------------------
    selected_model_name = st.selectbox("Select Model", list(models.keys()))
    model = models[selected_model_name]

    # Predict
    y_pred = model.predict(X)

    if st.button("Evaluate Model"):

        # -------------------------
        # c. Display Evaluation Metrics
        # -------------------------
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        auc = roc_auc_score(y, model.predict_proba(X)[:,1])

        st.subheader("Evaluation Metrics")
        st.write(f"Accuracy: {acc:.3f}")
        st.write(f"Precision: {prec:.3f}")
        st.write(f"Recall: {rec:.3f}")
        st.write(f"F1 Score: {f1:.3f}")
        st.write(f"AUC: {auc:.3f}")

        # -------------------------
        # d. Confusion Matrix
        # -------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification Report
        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

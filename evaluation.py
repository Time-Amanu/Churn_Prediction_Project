from sklearn.metrics import roc_auc_score, precision_score, recall_score
import pandas as pd

def evaluate_models(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            'ROC-AUC': roc_auc_score(y_test, y_proba),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred)
        }
    return pd.DataFrame(results).T
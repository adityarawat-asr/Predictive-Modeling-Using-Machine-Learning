"""
Predictive Modeling Using Machine Learning
--------------------------------------------
Trains and compares Logistic/Decision Tree/Random Forest style models
(Decision Tree + Random Forest, as required by the task) on the
Breast Cancer Wisconsin dataset (built into scikit-learn), then
evaluates and visualizes performance with a confusion matrix and
an ROC curve, and writes a comparison table.

Outputs (written to ./outputs/):
    - confusion_matrix.png
    - roc_curve.png
    - feature_importance.png
    - results_table.csv
    - results_table.md
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, roc_auc_score
)

OUT = "outputs"

# ---------------------------------------------------------------
# 1. Load & split data
# ---------------------------------------------------------------
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------------------------------------------------------
# 2. Train models
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
roc_data = {}
best_name, best_model, best_acc = None, None, -1

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)

    results.append({
        "Model": name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "ROC-AUC": round(auc, 4),
    })

    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_data[name] = (fpr, tpr, auc)

    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(f"{OUT}/results_table.csv", index=False)

with open(f"{OUT}/results_table.md", "w") as f:
    f.write(results_df.to_markdown(index=False))

print(results_df.to_string(index=False))
print(f"\nBest model: {best_name} (accuracy={best_acc:.4f})")

# ---------------------------------------------------------------
# 3. Confusion matrix (best model)
# ---------------------------------------------------------------
best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)

fig, ax = plt.subplots(figsize=(5.5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(f"{OUT}/confusion_matrix.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. ROC curves (all models)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5.5))
for name, (fpr, tpr, auc) in roc_data.items():
    ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Model Comparison")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT}/roc_curve.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 5. Feature importance (Random Forest)
# ---------------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=data.feature_names)
top10 = importances.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(7, 5))
top10.sort_values().plot(kind="barh", ax=ax, color="#2E86AB")
ax.set_title("Top 10 Feature Importances — Random Forest")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT}/feature_importance.png", dpi=150)
plt.close()

print("\nAll outputs written to ./outputs/")

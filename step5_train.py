import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

df = pd.read_csv("data/landmarks.csv")
feat = [f"f{i}" for i in range(63)]

blk = (df.idx // 100) % 10
train = df[blk <= 7]
val = df[blk == 8]
test = df[blk == 9]
print(f"Samples  train: {len(train)}  val: {len(val)}  test: {len(test)}")


def mirror(X):
    X = X.copy()
    X[:, 0::3] *= -1  # flip x coordinates
    return X


Xtr = train[feat].values
ytr = train.label.values
Xtr = np.vstack([Xtr, mirror(Xtr)])
ytr = np.concatenate([ytr, ytr])

Xva, yva = val[feat].values, val.label.values
Xte, yte = test[feat].values, test.label.values

models = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, n_jobs=4, class_weight="balanced", random_state=42),
    "MLP": make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200,
                      early_stopping=True, random_state=42)),
}

best_name, best_model, best_val = None, None, -1
for name, m in models.items():
    print(f"\nTraining {name} ...")
    m.fit(Xtr, ytr)
    acc = accuracy_score(yva, m.predict(Xva))
    print(f"{name} validation accuracy: {acc * 100:.2f}%")
    if acc > best_val:
        best_name, best_model, best_val = name, m, acc

print(f"\nBest model: {best_name}")
pred = best_model.predict(Xte)
test_acc = accuracy_score(yte, pred)
print(f"TEST accuracy (held-out photos): {test_acc * 100:.2f}%")
print("\nPer-class report:")
print(classification_report(yte, pred, digits=3, zero_division=0))

classes = list(best_model.classes_)
cm = confusion_matrix(yte, pred, labels=classes)
cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)

plt.figure(figsize=(12, 10))
plt.imshow(cmn, cmap="Blues")
plt.xticks(range(len(classes)), classes, rotation=90)
plt.yticks(range(len(classes)), classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion matrix ({best_name}, test acc {test_acc * 100:.1f}%)")
plt.colorbar()
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)

pairs = []
for i in range(len(classes)):
    for j in range(len(classes)):
        if i != j and cm[i, j] > 0:
            pairs.append((cmn[i, j], classes[i], classes[j], cm[i, j]))
pairs.sort(reverse=True)
print("\nMost confused pairs (true -> predicted):")
for rate, a, b, n in pairs[:10]:
    print(f"  {a} -> {b}: {rate * 100:.1f}% ({n} samples)")

joblib.dump({"model": best_model, "classes": classes}, "models/letter_model.joblib")
print("\nSaved models/letter_model.joblib and results/confusion_matrix.png")

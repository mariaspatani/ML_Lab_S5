import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Binary classification
y = (y != 0).astype(int)
X = X[:, [2, 3]]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 6. Prediction and accuracy
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# 7. Decision boundary
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train)
ax = plt.gca()
xx, yy = np.meshgrid(np.linspace(-2, 2, 100),
                     np.linspace(-2, 2, 100))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[-1, 0, 1])
plt.scatter(model.support_vectors_[:, 0],
            model.support_vectors_[:, 1],
            s=150, facecolors='none', edgecolors='k')
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("SVM Decision Boundary")
plt.show()

# 8. Multiclass classification
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 9. Kernel comparison
models = {
    'Linear': SVC(kernel='linear'),
    'Polynomial': SVC(kernel='poly', degree=3),
    'RBF': SVC(kernel='rbf')
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    results[name] = acc
    print(name, "Accuracy:", acc)
    print(classification_report(y_test, pred))

# 10. Bar graph
plt.bar(results.keys(), results.values())
plt.xlabel("Kernels")
plt.ylabel("Accuracy")
plt.title("SVM Kernel Comparison")
plt.show()

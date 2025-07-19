# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load data
iris = load_iris()
X = iris.data
y = iris.target

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Actual:", y_test)
print("y_test:", y_test.tolist())
print("Class distribution:", np.bincount(y_test))
print("Predicted:", y_pred.tolist())
print("Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)

# Step 6: Plot confusion matrix using matplotlib heatmap
plt.figure(figsize=(7, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (KNN - Iris Dataset)")
plt.tight_layout()
plt.show()
🌟 K-Nearest Neighbors (KNN) Classifier using Iris Dataset
A beginner-friendly example of the KNN classification algorithm using the popular Iris flower dataset, including data preprocessing, evaluation metrics, and visualization.

📌 What is KNN?
K-Nearest Neighbors (KNN) is a simple, instance-based learning algorithm that classifies a data point based on the majority class of its k nearest neighbors in the feature space.

Supervised learning

Non-parametric, lazy learning

Uses Euclidean distance (by default)

🛠️ Tools Used
Python

scikit-learn (sklearn)

Pandas

Matplotlib & Seaborn (for visualization)

📊 Dataset
We use the built-in Iris Dataset from sklearn, which includes:

150 samples (flowers)

4 features: sepal length, sepal width, petal length, petal width

3 classes: Setosa, Versicolor, Virginica

✅ Steps Performed
Import libraries

Load dataset

Split into train and test sets

Standardize features using StandardScaler

Train KNN model

Predict & Evaluate

Visualize Confusion Matrix

💡 Code Example

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

# Step 3: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 4: Train model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# Step 5: Predict
y_pred = model.predict(X_test_scaled)

# Step 6: Evaluate
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", round(acc * 100, 2), "%")
print("Confusion Matrix:\n", cm)

# Step 7: Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("KNN Confusion Matrix (Iris Dataset)")
plt.tight_layout()
plt.show()

📈 Output (Example)

Accuracy: 100.0 %
Confusion Matrix:
[[19  0  0]
 [ 0 13  0]
 [ 0  0 13]]


📚 Learnings
KNN is highly sensitive to feature scaling.

Choosing the right value of k is crucial.

Perfect accuracy on test set often indicates well-separated classes (like Iris).

📌 To Try
Change k value (n_neighbors) and observe changes.

Try plotting decision boundaries using matplotlib (2D).

Use a different dataset (e.g., Wine, Breast Cancer from sklearn).

🔗 Related Topics
Euclidean Distance

Classification Metrics: Precision, Recall, F1

Hyperparameter Tuning (GridSearchCV)

📂 Project Structure

KNN-Classifier-IrisDataset/
│
├── knn_iris.py              # Python file with full code
├── README.md                # This file
├── requirements.txt         # Dependencies
└── images/
    └── confusion_matrix.png # (Optional) save plots here
    
✅ Requirements: scikit-learn, numpy, pandas, matplotlib, seaborn



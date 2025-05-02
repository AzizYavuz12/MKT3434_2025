# Machine Learning Course GUI –Submission (Assignment 1 & 2)

This Python GUI application, developed using **PyQt6**, provides an interactive environment for experimenting with classical and deep learning algorithms, as well as dimensionality reduction and clustering techniques. This project satisfies the requirements of **both Assignment 1 and Assignment 2** for the MKT3434 course.

## Project Purpose

The goal of this project is to give students an intuitive interface to:
- Load and preprocess datasets
- Train, evaluate, and visualize a wide range of machine learning models
- Apply advanced dimensionality reduction techniques for high-dimensional data
- Understand model performance through metrics and visual diagnostics
- Use unsupervised learning for clustering and projection

---

## Installation

Install all dependencies with:

```bash
pip install pyqt6 scikit-learn tensorflow matplotlib pandas numpy plotly umap-learn
```

✅ Assignment 1: Core ML Functionality

📌 Data Handling

### Mean Imputation for Missing Data
Replaces missing values with the column mean using `SimpleImputer` from scikit-learn. This replaces missing values with the column mean using scikit-learn's `SimpleImputer`.
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")
self.X_train = imputer.fit_transform(self.X_train)
```

- Load from built-in datasets or custom CSV files
- Select target column for supervised tasks
- Apply feature scaling:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler

### Feature Scaling
Normalizes feature values to improve model convergence. Normalize features using standard, min-max or robust methods to improve model performance.
```python
if scaling_method == "Standard Scaling":
    scaler = preprocessing.StandardScaler()
self.X_train = scaler.fit_transform(self.X_train)
```

- Handle missing values using:
  - Mean Imputation
  - Interpolation
  - Forward / Backward Fill

### Interpolation / Forward-Fill / Backward-Fill
Handles missing values using time-series-aware strategies. Handle missing time-series style data using pandas methods.
```python
self.X_train = self.X_train.interpolate(method='linear').fillna(method='bfill')
```

- Train / Validation / Test splits (e.g., 70-15-15)

### Custom Train/Validation/Test Splits
Splits the data into user-defined proportions for training, validation, and testing. Custom data partitioning to simulate real-world ML workflows.
```python
self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
    X_temp, y_temp, test_size=val_relative_size)
```

📌 Classical Models

- Regression Models:
  - Linear Regression
  - Support Vector Regression (SVR)

### Support Vector Regression (SVR)
A regression model that uses a margin of tolerance to fit the best line. A regression model with tunable kernel and margin-insensitive loss.
```python
model = SVR(C=1.0, epsilon=0.1, kernel='rbf')
model.fit(self.X_train, self.y_train)
```

- Classification Models:
  - Logistic Regression
  - Naive Bayes (with priors)
  - SVM
  - KNN, Decision Tree, Random Forest

### Random Forest Classifier
An ensemble learning method that uses multiple decision trees to increase accuracy and prevent overfitting. A robust ensemble model using multiple decision trees.
```python
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(self.X_train, self.y_train)
```

📌 Loss Function Configuration

### Customizable Loss Functions
Supports Mean Squared Error, Mean Absolute Error, and Huber loss for regression tasks. Select regression loss functions dynamically.
```python
if loss_type == "MSE":
    loss = mean_squared_error(y_true, y_pred)
elif loss_type == "Huber":
    delta = 1.0
    loss = np.mean(np.where(
        np.abs(y_true - y_pred) < delta,
        0.5 * (y_true - y_pred) ** 2,
        delta * (np.abs(y_true - y_pred) - 0.5 * delta)
    ))
```

📌 Naive Bayes Custom Priors

### Naive Bayes with Custom Priors
Allows specification of prior probabilities to influence classification outcomes. Enables students to specify prior class probabilities.
```python
priors = list(map(float, self.prior_input.text().strip().split(',')))
model = GaussianNB(var_smoothing=smoothing, priors=priors)
model.fit(self.X_train, self.y_train)
```

📌 Visual Output

### Prediction Results Visualization
Displays a scatter plot comparing actual and predicted values. Compare predicted vs. actual values with scatter plots.
```python
ax.scatter(self.y_test, y_pred)
ax.plot([self.y_test.min(), self.y_test.max()],
         [self.y_test.min(), self.y_test.max()], 'r--')
```

---

✅ Assignment 2: Dimensionality Reduction & Advanced Analysis

📌 PCA – Principal Component Analysis

### Principal Component Analysis (PCA)
PCA is a statistical method that transforms original features into a set of new uncorrelated components 
(principal components), ranked by the amount of variance they capture. It is especially useful for:
- Reducing computational cost
- Removing multicollinearity
- Visualizing high-dimensional data in 2D or 3D
Reduces dimensionality while preserving variance.
```python
from sklearn.decomposition import PCA

# Apply PCA with user-defined number of components
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_train)

# Explained variance ratio
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("First few components:
", X_pca[:5])
X_pca = pca.fit_transform(self.X_train)
```

📌 Covariance-Based Manual PCA

### Manual PCA from Covariance Matrix
This example manually computes the eigenvectors and eigenvalues of a covariance matrix to find the direction of maximum variance. 
It's a practical demonstration of how PCA works under the hood using linear algebra concepts.
Demonstrates eigen decomposition and 1D projection.
```python
import numpy as np

# Manual PCA using covariance matrix
cov = np.array([[5, 2], [2, 3]])
eigvals, eigvecs = np.linalg.eig(cov)
print("Eigenvalues:", eigvals)
print("Principal Component:", eigvecs[:, np.argmax(eigvals)])
eigvals, eigvecs = np.linalg.eig(cov)
principal_vector = eigvecs[:, np.argmax(eigvals)]
```

📌 LDA – Linear Discriminant Analysis

### Linear Discriminant Analysis (LDA)
LDA is a supervised dimensionality reduction technique that finds the linear combinations of features that best separate 
two or more classes. Unlike PCA, which is unsupervised, LDA uses class labels to enhance inter-class variance while minimizing intra-class variance.
Projects data in a way that maximizes class separability.
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Apply LDA for dimensionality reduction and class separation
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_train, y_train)
print("LDA Projection:
", X_lda[:5])
X_lda = lda.fit_transform(self.X_train, self.y_train)
```

📌 t-SNE

### t-SNE (t-Distributed Stochastic Neighbor Embedding)
t-SNE is a powerful non-linear dimensionality reduction algorithm used for visualization. It models pairwise similarities in high-dimensional space 
and tries to preserve these relationships in a low-dimensional map. It’s excellent for discovering clusters and structures in complex datasets, 
especially for visual exploration.
Projects high-dimensional data to 2D/3D using non-linear mapping.
```python
from sklearn.manifold import TSNE

# Run t-SNE with configurable dimensions and perplexity
tsne = TSNE(n_components=3, perplexity=30.0, random_state=42)
X_tsne = tsne.fit_transform(X_train)
print("t-SNE Output:
", X_tsne[:5])
X_tsne = tsne.fit_transform(self.X_train)
```

📌 UMAP

### UMAP (Uniform Manifold Approximation and Projection)
UMAP is a modern and scalable algorithm for dimensionality reduction. It preserves both local neighborhoods and broader structure, 
making it suitable for clustering, visualization, and pre-processing large datasets before training.
A faster alternative to t-SNE that preserves more global structure.
```python
from umap import UMAP

# Apply UMAP for 2D or 3D projection
reducer = UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_train)
print("UMAP Projection:
", X_umap[:5])
X_umap = reducer.fit_transform(self.X_train)
```

📌 KMeans Clustering

### KMeans Clustering & Evaluation
KMeans groups data into `k` clusters based on minimizing within-cluster distance. This section includes:
- Elbow Method: to choose optimal `k` by plotting inertia
- Silhouette Score: to assess how well-separated the clusters are
- PCA projection for visualization of clusters in 2D space
Fit clusters, evaluate inertia and silhouette score.
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Run KMeans clustering and evaluate using silhouette score
k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
labels = kmeans.fit_predict(X_train)
inertia = kmeans.inertia_
score = silhouette_score(X_train, labels)

print(f"KMeans Inertia: {inertia}")
print(f"Silhouette Score: {score:.3f}")
kmeans.fit(self.X_train)
labels = kmeans.predict(self.X_train)
score = silhouette_score(self.X_train, labels)
```

📌 Cross-Validation

### k-Fold Cross Validation
k-Fold CV is a statistical method to evaluate model generalization. It splits the training data into `k` parts, trains the model on `k-1` folds, 
and tests on the remaining one. The process repeats `k` times to calculate a stable average metric, such as accuracy or RMSE. 
This helps reduce bias from a single train/test split.
Evaluate model performance on multiple folds of the training set.
```python
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# Perform k-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in kf.split(X):
    model.fit(X[train_idx], y[train_idx])
    preds = model.predict(X[test_idx])
    acc = accuracy_score(y[test_idx], preds)
    accuracies.append(acc)

print("Fold Accuracies:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
for train_idx, test_idx in kf.split(self.X_train):
    model.fit(self.X_train[train_idx], self.y_train[train_idx])
    y_pred = model.predict(self.X_train[test_idx])
```

## Student Information

- **Student ID**: 2206A604  
- **Student Name & Surname**: Aziz Yavuz
- **Course**: MKT3434 – Machine Learning  
- **Instructor**: Ertugrul Bayraktar

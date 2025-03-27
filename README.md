
# Machine Learning Course GUI – Enhanced Version

A graphical interface built with **PyQt6**, enabling users to train and visualize classical machine learning and deep learning models interactively. Enhancements include model configuration, loss selection, visualization, and evaluation.

---

## Setup

### Installation
```bash
pip install pyqt6 scikit-learn tensorflow matplotlib pandas numpy
```
_Install required packages._

---

## GUI Usage Guide

### 1. Load Data
- Use built-in datasets or load a CSV.
- Select target column from a popup dialog.
- Choose optional scaling:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
- Adjust train-test split (e.g., 0.2).

### 2. Train Classical Models
- Available Models:
  - Linear Regression, Support Vector Regression (SVR)
  - Logistic Regression, Naive Bayes
  - Support Vector Machine (SVM), KNN
  - Decision Tree, Random Forest

- For regression:
  - Select a loss function: **MSE**, **MAE**, or **Huber**

- For classification:
  - Choose between **Cross-Entropy** and **Hinge** loss

- Naive Bayes offers custom prior configuration:
  - Example: `0.3, 0.7` for binary classes

- After training, pop-up shows calculated loss.
- Visual panel shows confusion matrix or scatter plot.

### 3. Deep Learning
- Click **Add Layer** to define a custom MLP architecture:
  - Dense, Dropout, Flatten, Conv2D (partial support)
- Set training options:
  - Batch size, Epochs, Learning rate
- Train model and view accuracy/loss over epochs

### 4. Dimensionality Reduction
- K-Means:
  - Set `n_clusters`, `max_iter`, `n_init`
- PCA:
  - Set `n_components`, enable `whiten` if desired

### 5. Reinforcement Learning (Planned)
- Basic UI setup with environments like `CartPole-v1`
- Algorithm options (non-functional): Q-Learning, SARSA, DQN

---

## Model Training Explained

### 1. Model Initialization Example
```python
if name == "Linear Regression":
    model = LinearRegression(
        fit_intercept=param_widgets["fit_intercept"].isChecked()
    )
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test)
```
_Creates and trains a Linear Regression model with optional intercept._

```python
elif name == "Support Vector Regression":
    model = SVR(
        C=param_widgets["C"].value(),
        epsilon=param_widgets["epsilon"].value(),
        kernel=param_widgets["kernel"].currentText()
    )
    model.fit(self.X_train, self.y_train)
    y_pred = model.predict(self.X_test)
```
_Builds an SVR model with user-defined kernel and parameters._

### 2. Regression Loss Calculation
```python
def calculate_regression_loss(y_true, y_pred):
    loss_type = self.loss_combo.currentText()
    if loss_type == "MSE":
        return mean_squared_error(y_true, y_pred)
    elif loss_type == "MAE":
        return mean_absolute_error(y_true, y_pred)
    elif loss_type == "Huber":
        delta = 1.0
        return np.mean(np.where(
            np.abs(y_true - y_pred) < delta,
            0.5 * (y_true - y_pred) ** 2,
            delta * (np.abs(y_true - y_pred) - 0.5 * delta)
        ))
```
_Computes regression loss based on selected metric (MSE, MAE, Huber)._ 

### 3. Classification Loss Calculation
```python
if loss_type == "Cross-Entropy":
    y_proba = model.predict_proba(self.X_test)
    loss = log_loss(self.y_test, y_proba)
elif loss_type == "Hinge":
    loss = hinge_loss(self.y_test, y_pred)
```
_Handles loss computation for classifiers using probabilities or margins._

### 4. Naive Bayes with Custom Priors
```python
if self.prior_combo.currentText() == "Custom":
    try:
        priors = list(map(float, self.prior_input.text().strip().split(',')))
    except:
        self.show_error("Invalid custom priors format. Example: 0.3, 0.7")
        return
model = GaussianNB(var_smoothing=smoothing, priors=priors)
model.fit(self.X_train, self.y_train)
y_pred = model.predict(self.X_test)
```
_Allows setting class priors manually for Naive Bayes classifier._

### 5. Model Evaluation and Visualization
```python
self.current_model = model
self.update_visualization(y_pred)
self.update_metrics(y_pred)
```
_Displays model results using graphs and metrics panel._

---

## Example Workflow
1. Load the Boston dataset
2. Select Vector Standard Regression
3. Choose Logistic Regression with `C=1.0`, `Epsion=0.1`
4. Select Train Regression
5. Click Train → See pop-up loss + metrics

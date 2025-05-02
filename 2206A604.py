import sys
import numpy as np
import pandas as pd
import plotly.express as px

from umap import UMAP
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QTabWidget, QPushButton, QLabel, 
                           QComboBox, QFileDialog, QSpinBox, QDoubleSpinBox,
                           QGroupBox, QScrollArea, QTextEdit, QStatusBar,
                           QProgressBar, QCheckBox, QGridLayout, QMessageBox,
                           QDialog, QLineEdit)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.metrics import silhouette_score
from sklearn import datasets, preprocessing, model_selection
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix,mean_absolute_error,log_loss, hinge_loss
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class MLCourseGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Machine Learning Course GUI")
        self.setGeometry(100, 100, 1400, 800)
        
        # Initialize main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Initialize data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.current_model = None
        
        # Neural network configuration
        self.layer_config = []
        
        # Create components
        self.create_data_section()
        self.create_tabs()
        self.setup_dimensionality_reduction_tools()
        self.create_visualization()
        self.create_status_bar()

    def run_pca(self):
        """Run PCA and plot explained variance"""
        try:
            # Convert X_train to numpy array if it's a DataFrame
            if isinstance(self.X_train, pd.DataFrame):
                X_train_array = self.X_train.values
            else:
                X_train_array = self.X_train

            # Check if data exists
            if X_train_array is None or len(X_train_array) == 0:
                self.show_error("No training data available for PCA.")
                return

            # Automatically apply standard scaling (important for PCA)
            scaler = preprocessing.StandardScaler()
            X_scaled = scaler.fit_transform(X_train_array)

            # Perform PCA
            n_components = self.pca_components_spin.value()
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X_scaled)

            # Plot the explained variance
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            self.canvas.draw()

            # Log to console
            print("PCA Successful:", pca.explained_variance_ratio_)

        except Exception as e:
            print("PCA Error:", str(e))  # For developer debugging
            self.show_error(f"Error running PCA: {str(e)}")

    
    def demo_covariance_projection(self):
        """Manual PCA example using given covariance matrix"""
        cov = np.array([[5, 2], [2, 3]])
        eigvals, eigvecs = np.linalg.eig(cov)
        principal_vector = eigvecs[:, np.argmax(eigvals)]

        self.show_message(f"Covariance Matrix:\n{cov}\n\nPrincipal Component:\n{principal_vector}")


    def run_lda(self):
        """Run LDA and plot class separation"""
        try:
            if self.y_train is None:  
                self.show_error("LDA requires target labels (y_train)!")
                return
            
            lda = LDA(n_components=2) 
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            self.figure.clear() 
            ax = self.figure.add_subplot(111)  
            scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=self.y_train, cmap='jet') 
            self.figure.colorbar(scatter) 
            ax.set_title('LDA Class Separation') 
            self.canvas.draw()

            # Explained variance ratio
            explained_var = lda.explained_variance_ratio_
            text = "LDA Explained Variance Ratio:\n" + "\n".join([f"Component {i+1}: {var:.4f}" for i, var in enumerate(explained_var)])
            self.show_message(text)

        except Exception as e:
            self.show_error(f"Error running LDA: {str(e)}")

    def run_tsne(self):
        """Run t-SNE and plot in 2D or 3D"""
        try:
            # Get selection from ComboBox
            selected_dim = self.tsne_dim_combo.currentText()
            n_components = 2 if selected_dim == "2D" else 3

            # Create t-SNE model
            tsne = TSNE(n_components=n_components, perplexity=self.tsne_perplexity_spin.value(), random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)

            self.figure.clear()

            # 2D Plot
            if n_components == 2:
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], 
                                    c=self.y_train if self.y_train is not None else 'b', cmap='viridis')
                self.figure.colorbar(scatter)
                ax.set_title('t-SNE Projection (2D)')

            # 3D Plot
            else:
                from mpl_toolkits.mplot3d import Axes3D  # Required for 3D plotting
                ax = self.figure.add_subplot(111, projection='3d')
                scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], 
                                    c=self.y_train if self.y_train is not None else 'b', cmap='viridis')
                self.figure.colorbar(scatter)
                ax.set_title('t-SNE Projection (3D)')

            self.canvas.draw()

        except Exception as e:
            self.show_error(f"Error running t-SNE: {str(e)}")

    def run_tsne_plotly(self):
        """Run t-SNE and show with Plotly"""
        try:
            selected_dim = self.tsne_dim_combo.currentText()
            n_components = 2 if selected_dim == "2D" else 3

            tsne = TSNE(n_components=n_components, perplexity=self.tsne_perplexity_spin.value(), random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)

            if self.y_train is not None:
                labels = self.y_train
            else:
                labels = [0] * len(X_tsne)

            if n_components == 2:
                fig = px.scatter(x=X_tsne[:, 0], y=X_tsne[:, 1], color=labels.astype(str),
                                labels={'x': 'TSNE-1', 'y': 'TSNE-2'},
                                title='t-SNE Projection (2D - Plotly)')
            else:
                fig = px.scatter_3d(x=X_tsne[:, 0], y=X_tsne[:, 1], z=X_tsne[:, 2],
                                    color=labels.astype(str),
                                    labels={'x': 'TSNE-1', 'y': 'TSNE-2', 'z': 'TSNE-3'},
                                    title='t-SNE Projection (3D - Plotly)')

            fig.show()

        except Exception as e:
            self.show_error(f"Error running Plotly t-SNE: {str(e)}")


    def run_umap(self):
        """Run UMAP and plot"""
        try:
            reducer = UMAP(n_components=2, random_state=42)
            X_umap = reducer.fit_transform(self.X_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=self.y_train if self.y_train is not None else 'b', cmap='Spectral')
            self.figure.colorbar(scatter)
            ax.set_title('UMAP Projection')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error running UMAP: {str(e)}")

    def run_umap_plotly(self):
        """Run UMAP and show with Plotly"""
        try:
            selected_dim = self.tsne_dim_combo.currentText()  # We reuse the t-SNE dimension selection
            n_components = 2 if selected_dim == "2D" else 3

            reducer = UMAP(n_components=n_components, random_state=42)
            X_umap = reducer.fit_transform(self.X_train)

            if self.y_train is not None:
                labels = self.y_train
            else:
                labels = [0] * len(X_umap)

            if n_components == 2:
                fig = px.scatter(x=X_umap[:, 0], y=X_umap[:, 1], color=labels.astype(str),
                                labels={'x': 'UMAP-1', 'y': 'UMAP-2'},
                                title='UMAP Projection (2D - Plotly)')
            else:
                fig = px.scatter_3d(x=X_umap[:, 0], y=X_umap[:, 1], z=X_umap[:, 2],
                                    color=labels.astype(str),
                                    labels={'x': 'UMAP-1', 'y': 'UMAP-2', 'z': 'UMAP-3'},
                                    title='UMAP Projection (3D - Plotly)')

            fig.show()

        except Exception as e:
            self.show_error(f"Error running Plotly UMAP: {str(e)}")


    def run_kmeans(self):
        """Run K-Means: Elbow method, PCA visualization, and Silhouette Score"""
        try:
            inertias = []
            Ks = range(1, 11)
            for k in Ks:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)

            k_selected = self.kmeans_k_spin.value()
            kmeans_final = KMeans(n_clusters=k_selected, random_state=42, n_init='auto')
            labels = kmeans_final.fit_predict(self.X_train)
            score = silhouette_score(self.X_train, labels)

            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(self.X_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10')
            self.figure.colorbar(scatter)
            ax.set_title(f"KMeans Clusters (k={k_selected}) - Silhouette Score: {score:.4f}")
            ax.set_xlabel("PCA Component 1")
            ax.set_ylabel("PCA Component 2")
            self.canvas.draw()

            elbow_text = f"Elbow Inertias (k=1 to 10):\n" + ", ".join(f"{val:.2f}" for val in inertias)
            self.show_message(elbow_text)

        except Exception as e:
            self.show_error(f"Error running KMeans: {str(e)}")


    def load_dataset(self):
        """Load selected dataset"""
        try:

            total = self.train_split_spin.value() + self.val_split_spin.value() + self.test_split_spin.value()
            if abs(total - 1.0) > 0.01:
                self.show_error("Train / Validation / Test ratios must sum to 1.0")
                return
            
            dataset_name = self.dataset_combo.currentText()
            
            if dataset_name == "Load Custom Dataset":
                return
            
            # Load selected dataset
            if dataset_name == "Iris Dataset":
                data = datasets.load_iris()
            elif dataset_name == "Breast Cancer Dataset":
                data = datasets.load_breast_cancer()
            elif dataset_name == "Digits Dataset":
                data = datasets.load_digits()
            elif dataset_name == "Boston Housing Dataset":
                data = datasets.load_boston()
            elif dataset_name == "MNIST Dataset":
                (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
                self.X_train, self.X_test = X_train, X_test
                self.y_train, self.y_test = y_train, y_test
                self.status_bar.showMessage(f"Loaded {dataset_name}")
                return
            
            # Split using custom Train/Val/Test proportions
            test_size = self.test_split_spin.value()
            val_size = self.val_split_spin.value()

            X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                data.data, data.target, test_size=test_size, random_state=42)

            val_relative_size = val_size / (1.0 - test_size)
            self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                X_temp, y_temp, test_size=val_relative_size, random_state=42)

            # Apply scaling if selected
            self.apply_scaling()
            
            self.status_bar.showMessage(f"Loaded {dataset_name}")
            
        except Exception as e:
            self.show_error(f"Error loading dataset: {str(e)}")
    
    def load_custom_data(self):
        """Load custom dataset from CSV file"""
        try:

            total = self.train_split_spin.value() + self.val_split_spin.value() + self.test_split_spin.value()
            if abs(total - 1.0) > 0.01:
                self.show_error("Train / Validation / Test ratios must sum to 1.0")
                return
            
            file_name, _ = QFileDialog.getOpenFileName(
                self,
                "Load Dataset",
                "",
                "CSV files (*.csv)"
            )
            
            if file_name:
                # Load data
                data = pd.read_csv(file_name)
                
                # Ask user to select target column
                target_col = self.select_target_column(data.columns)
                
                if target_col:
                    X = data.drop(target_col, axis=1)
                    y = data[target_col]

                    # Split using Train/Val/Test proportions
                    test_size = self.test_split_spin.value()
                    val_size = self.val_split_spin.value()

                    X_temp, self.X_test, y_temp, self.y_test = model_selection.train_test_split(
                        X, y, test_size=test_size, random_state=42)

                    val_relative_size = val_size / (1.0 - test_size)
                    self.X_train, self.X_val, self.y_train, self.y_val = model_selection.train_test_split(
                        X_temp, y_temp, test_size=val_relative_size, random_state=42)

                    
                    # Apply scaling if selected
                    self.apply_scaling()
                    
                    self.status_bar.showMessage(f"Loaded custom dataset: {file_name}")
                    
        except Exception as e:
            self.show_error(f"Error loading custom dataset: {str(e)}")
    
    def select_target_column(self, columns):
        """Dialog to select target column from dataset"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Target Column")
        layout = QVBoxLayout(dialog)
        
        combo = QComboBox()
        combo.addItems(columns)
        layout.addWidget(combo)
        
        btn = QPushButton("Select")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return combo.currentText()
        return None
    
    def apply_scaling(self):
        """Apply selected scaling method to the data"""
        scaling_method = self.scaling_combo.currentText()
        
        if scaling_method != "No Scaling":
            try:
                if scaling_method == "Standard Scaling":
                    scaler = preprocessing.StandardScaler()
                elif scaling_method == "Min-Max Scaling":
                    scaler = preprocessing.MinMaxScaler()
                elif scaling_method == "Robust Scaling":
                    scaler = preprocessing.RobustScaler()
                
                self.X_train = scaler.fit_transform(self.X_train)
                self.X_test = scaler.transform(self.X_test)
                
            except Exception as e:
                self.show_error(f"Error applying scaling: {str(e)}")
    def create_data_section(self):
        """Create the data loading and preprocessing section"""
        data_group = QGroupBox("Data Management")
        data_layout = QHBoxLayout()
        
        # Dataset selection
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems([
            "Load Custom Dataset",
            "Iris Dataset",
            "Breast Cancer Dataset",
            "Digits Dataset",
            "Boston Housing Dataset",
            "MNIST Dataset"
        ])
        self.dataset_combo.currentIndexChanged.connect(self.load_dataset)
        
        # Data loading button
        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_custom_data)
        
        # Preprocessing options
        self.scaling_combo = QComboBox()
        self.scaling_combo.addItems([
            "No Scaling",
            "Standard Scaling",
            "Min-Max Scaling",
            "Robust Scaling"
        ])
        
        # Train-test split options
        self.split_spin = QDoubleSpinBox()
        self.split_spin.setRange(0.1, 0.9)
        self.split_spin.setValue(0.2)
        self.split_spin.setSingleStep(0.1)
        
        # Add widgets to layout
        data_layout.addWidget(QLabel("Dataset:"))
        data_layout.addWidget(self.dataset_combo)
        data_layout.addWidget(self.load_btn)
        data_layout.addWidget(QLabel("Scaling:"))
        data_layout.addWidget(self.scaling_combo)
        data_layout.addWidget(QLabel("Test Split:"))
        data_layout.addWidget(self.split_spin)
        
        data_group.setLayout(data_layout)
        self.layout.addWidget(data_group)
        split_options = self.create_split_options()
        self.layout.addWidget(split_options)

    def create_split_options(self):
        """Create train/validation/test split and k-fold options"""
        split_group = QGroupBox("Data Split Options")
        layout = QVBoxLayout()

        # --- Train/Validation/Test Split ---
        split_layout = QHBoxLayout()

        self.train_split_spin = QDoubleSpinBox()
        self.train_split_spin.setRange(0.1, 0.9)
        self.train_split_spin.setSingleStep(0.05)
        self.train_split_spin.setValue(0.7)
        split_layout.addWidget(QLabel("Train %"))
        split_layout.addWidget(self.train_split_spin)

        self.val_split_spin = QDoubleSpinBox()
        self.val_split_spin.setRange(0.05, 0.9)
        self.val_split_spin.setSingleStep(0.05)
        self.val_split_spin.setValue(0.15)
        split_layout.addWidget(QLabel("Validation %"))
        split_layout.addWidget(self.val_split_spin)

        self.test_split_spin = QDoubleSpinBox()
        self.test_split_spin.setRange(0.05, 0.9)
        self.test_split_spin.setSingleStep(0.05)
        self.test_split_spin.setValue(0.15)
        split_layout.addWidget(QLabel("Test %"))
        split_layout.addWidget(self.test_split_spin)

        layout.addLayout(split_layout)

        # --- K-Fold Cross Validation ---
        self.kfold_checkbox = QCheckBox("Enable k-Fold Cross Validation")
        layout.addWidget(self.kfold_checkbox)

        kfold_layout = QHBoxLayout()
        self.kfold_spin = QSpinBox()
        self.kfold_spin.setRange(2, 20)
        self.kfold_spin.setValue(5)
        kfold_layout.addWidget(QLabel("k value:"))
        kfold_layout.addWidget(self.kfold_spin)

        layout.addLayout(kfold_layout)

        split_group.setLayout(layout)

        return split_group


    def create_tabs(self):
        """Create tabs for different ML topics"""
        self.tab_widget = QTabWidget()
        
        # Create individual tabs
        tabs = [
            ("Classical ML", self.create_classical_ml_tab),
            ("Deep Learning", self.create_deep_learning_tab),
            ("Dimensionality Reduction", self.create_dim_reduction_tab),
            ("Reinforcement Learning", self.create_rl_tab)
        ]
        
        for tab_name, create_func in tabs:
            scroll = QScrollArea()
            tab_widget = create_func()
            scroll.setWidget(tab_widget)
            scroll.setWidgetResizable(True)
            self.tab_widget.addTab(scroll, tab_name)
        
        self.layout.addWidget(self.tab_widget)
    
    def create_classical_ml_tab(self):
        """Create the classical machine learning algorithms tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Regression section
        regression_group = QGroupBox("Regression")
        regression_layout = QVBoxLayout()
        
        # Linear Regression
        lr_group = self.create_algorithm_group(
            "Linear Regression",
            {"fit_intercept": "checkbox",
             "normalize": "checkbox"}
        )
        
        regression_layout.addWidget(lr_group)

        # Support Vector Regression      
        svr_group = self.create_algorithm_group(
            "Support Vector Regression",
            {
                "C": "double",
                "epsilon": "double",
                "kernel": ["linear", "rbf", "poly"]
            }
        )
        regression_layout.addWidget(svr_group)
      
        # Logistic Regression
        logistic_group = self.create_algorithm_group(
            "Logistic Regression",
            {"C": "double",
             "max_iter": "int",
             "multi_class": ["ovr", "multinomial"]}
        )
        regression_layout.addWidget(logistic_group)
        layout.addWidget(regression_group, 0, 0)

        # Choosing loss function       
        loss_layout = QHBoxLayout()
        loss_label = QLabel("Loss Function:")
        self.loss_combo = QComboBox()
        self.loss_combo.addItems(["MSE", "MAE", "Huber"])
        loss_layout.addWidget(loss_label)
        loss_layout.addWidget(self.loss_combo)
        regression_layout.addLayout(loss_layout)
        regression_group.setLayout(regression_layout)

        # Classification section
        classification_group = QGroupBox("Classification")
        classification_layout = QVBoxLayout()
        
        # Classification Loss Selection
        clf_loss_layout = QHBoxLayout()
        clf_loss_label = QLabel("Loss Function:")
        self.classification_loss_combo = QComboBox()
        self.classification_loss_combo.addItems(["Cross-Entropy", "Hinge"])
        clf_loss_layout.addWidget(clf_loss_label)
        clf_loss_layout.addWidget(self.classification_loss_combo)
        classification_layout.addLayout(clf_loss_layout)

        # Naive Bayes
        nb_group = self.create_algorithm_group(
            "Naive Bayes",
            {"var_smoothing": "double"}
        )
        classification_layout.addWidget(nb_group)

        # Classification Naive Bayes and determining prior conditions
        prior_layout = QHBoxLayout()
        prior_label = QLabel("Priors:")
        self.prior_combo = QComboBox()
        self.prior_combo.addItems(["Uniform", "Custom"])
        prior_layout.addWidget(prior_label)
        prior_layout.addWidget(self.prior_combo)

        # Input field for custom prior
        self.prior_input = QLineEdit()
        self.prior_input.setPlaceholderText("e.g., 0.3, 0.7")
        prior_layout.addWidget(self.prior_input)

        classification_layout.addLayout(prior_layout)

        # SVM
        svm_group = self.create_algorithm_group(
            "Support Vector Machine",
            {"C": "double",
             "kernel": ["linear", "rbf", "poly"],
             "degree": "int"}
        )
        classification_layout.addWidget(svm_group)
        
        # Decision Trees
        dt_group = self.create_algorithm_group(
            "Decision Tree",
            {"max_depth": "int",
             "min_samples_split": "int",
             "criterion": ["gini", "entropy"]}
        )
        classification_layout.addWidget(dt_group)
        
        # Random Forest
        rf_group = self.create_algorithm_group(
            "Random Forest",
            {"n_estimators": "int",
             "max_depth": "int",
             "min_samples_split": "int"}
        )
        classification_layout.addWidget(rf_group)
        
        # KNN
        knn_group = self.create_algorithm_group(
            "K-Nearest Neighbors",
            {"n_neighbors": "int",
             "weights": ["uniform", "distance"],
             "metric": ["euclidean", "manhattan"]}
        )
        classification_layout.addWidget(knn_group)
        
        classification_group.setLayout(classification_layout)
        layout.addWidget(classification_group, 0, 1)
        
        return widget
    
    def create_dim_reduction_tab(self):
        """Create the dimensionality reduction tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # K-Means section
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()
        
        kmeans_params = self.create_algorithm_group(
            "K-Means Parameters",
            {"n_clusters": "int",
             "max_iter": "int",
             "n_init": "int"}
        )
        kmeans_layout.addWidget(kmeans_params)
        
        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 0, 0)
        
        # PCA section
        pca_group = QGroupBox("Principal Component Analysis")
        pca_layout = QVBoxLayout()
        
        pca_params = self.create_algorithm_group(
            "PCA Parameters",
            {"n_components": "int",
             "whiten": "checkbox"}
        )
        pca_layout.addWidget(pca_params)
        
        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 0, 1)
        
        return widget
    
    def create_rl_tab(self):
        """Create the reinforcement learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # Environment selection
        env_group = QGroupBox("Environment")
        env_layout = QVBoxLayout()
        
        self.env_combo = QComboBox()
        self.env_combo.addItems([
            "CartPole-v1",
            "MountainCar-v0",
            "Acrobot-v1"
        ])
        env_layout.addWidget(self.env_combo)
        
        env_group.setLayout(env_layout)
        layout.addWidget(env_group, 0, 0)
        
        # RL Algorithm selection
        algo_group = QGroupBox("RL Algorithm")
        algo_layout = QVBoxLayout()
        
        self.rl_algo_combo = QComboBox()
        self.rl_algo_combo.addItems([
            "Q-Learning",
            "SARSA",
            "DQN"
        ])
        algo_layout.addWidget(self.rl_algo_combo)
        
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group, 0, 1)
        
        return widget
    
    def create_visualization(self):
        """Create the visualization section"""
        viz_group = QGroupBox("Visualization")
        viz_layout = QHBoxLayout()
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        viz_layout.addWidget(self.canvas)
        
        # Metrics display
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        viz_layout.addWidget(self.metrics_text)
        
        viz_group.setLayout(viz_layout)
        self.layout.addWidget(viz_group)
    
    def create_status_bar(self):
        """Create the status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Add progress bar
        self.progress_bar = QProgressBar()
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_algorithm_group(self, name, params):
        """Helper method to create algorithm parameter groups"""
        group = QGroupBox(name)
        layout = QVBoxLayout()
        
        # Create parameter inputs
        param_widgets = {}
        for param_name, param_type in params.items():
            param_layout = QHBoxLayout()
            param_layout.addWidget(QLabel(f"{param_name}:"))
            
            if param_type == "int":
                widget = QSpinBox()
                widget.setRange(1, 1000)
            elif param_type == "double":
                widget = QDoubleSpinBox()
                widget.setRange(0.0001, 1000.0)
                widget.setSingleStep(0.1)
            elif param_type == "checkbox":
                widget = QCheckBox()
            elif isinstance(param_type, list):
                widget = QComboBox()
                widget.addItems(param_type)
            
            param_layout.addWidget(widget)
            param_widgets[param_name] = widget
            layout.addLayout(param_layout)
        
        # Add train button
        train_btn = QPushButton(f"Train {name}")
        train_btn.clicked.connect(lambda: self.train_model(name, param_widgets))
        layout.addWidget(train_btn)
        
        group.setLayout(layout)
        return group
    # Model Training
    def train_model(self, name, param_widgets):
        try:
            model = None
            y_pred = None
            #This part will work if the user choose the K-Fold
            if self.kfold_checkbox.isChecked():
                if self.kfold_checkbox.isChecked():
                    from sklearn.model_selection import KFold
                    k = self.kfold_spin.value()
                    kf = KFold(n_splits=k, shuffle=True, random_state=42)

                    metrics = []
                    for train_index, test_index in kf.split(self.X_train):
                        X_train_k, X_test_k = self.X_train[train_index], self.X_train[test_index]
                        y_train_k, y_test_k = self.y_train[train_index], self.y_train[test_index]

                        model = None
                        y_pred_k = None

                        if name == "Linear Regression":
                            model = LinearRegression(fit_intercept=param_widgets["fit_intercept"].isChecked())
                            model.fit(X_train_k, y_train_k)
                            y_pred_k = model.predict(X_test_k)
                            mse = mean_squared_error(y_test_k, y_pred_k)
                            rmse = np.sqrt(mse)
                            metrics.append(rmse)

                        elif name == "Support Vector Regression":
                            model = SVR(
                                C=param_widgets["C"].value(),
                                epsilon=param_widgets["epsilon"].value(),
                                kernel=param_widgets["kernel"].currentText()
                            )
                            model.fit(X_train_k, y_train_k)
                            y_pred_k = model.predict(X_test_k)
                            mse = mean_squared_error(y_test_k, y_pred_k)
                            rmse = np.sqrt(mse)
                            metrics.append(rmse)

                        elif name in ["Logistic Regression", "Support Vector Machine", "Decision Tree", "Random Forest"]:
                            if name == "Logistic Regression":
                                model = LogisticRegression(
                                    C=param_widgets["C"].value(),
                                    max_iter=param_widgets["max_iter"].value(),
                                    multi_class=param_widgets["multi_class"].currentText()
                                )
                            elif name == "Support Vector Machine":
                                model = SVC(
                                    C=param_widgets["C"].value(),
                                    kernel=param_widgets["kernel"].currentText(),
                                    degree=param_widgets["degree"].value()
                                )
                            elif name == "Decision Tree":
                                model = DecisionTreeClassifier(
                                    max_depth=param_widgets["max_depth"].value(),
                                    min_samples_split=param_widgets["min_samples_split"].value(),
                                    criterion=param_widgets["criterion"].currentText()
                                )
                            elif name == "Random Forest":
                                model = RandomForestClassifier(
                                    n_estimators=param_widgets["n_estimators"].value(),
                                    max_depth=param_widgets["max_depth"].value(),
                                    min_samples_split=param_widgets["min_samples_split"].value()
                                )
                            model.fit(X_train_k, y_train_k)
                            y_pred_k = model.predict(X_test_k)
                            acc = accuracy_score(y_test_k, y_pred_k)
                            metrics.append(acc)

                    mean_val = np.mean(metrics)
                    std_val = np.std(metrics)

                    if name in ["Linear Regression", "Support Vector Regression"]:
                        self.show_message(f"{name} - k-Fold RMSE: {mean_val:.4f} ± {std_val:.4f}")
                        self.metrics_text.setText(f"RMSE (k={k}) across folds:\n{metrics}\n\nMean ± Std:\n{mean_val:.4f} ± {std_val:.4f}")
                    else:
                        self.show_message(f"{name} - k-Fold Accuracy: {mean_val:.4f} ± {std_val:.4f}")
                        self.metrics_text.setText(f"Accuracy (k={k}) across folds:\n{metrics}\n\nMean ± Std:\n{mean_val:.4f} ± {std_val:.4f}")
                    return


                for train_index, test_index in kf.split(self.X_train):
                    X_train_k, X_test_k = self.X_train[train_index], self.X_train[test_index]
                    y_train_k, y_test_k = self.y_train[train_index], self.y_train[test_index]

                    if name == "Logistic Regression":
                        model = LogisticRegression(
                            C=param_widgets["C"].value(),
                            max_iter=param_widgets["max_iter"].value(),
                            multi_class=param_widgets["multi_class"].currentText()
                        )
                        model.fit(X_train_k, y_train_k)
                        y_pred_k = model.predict(X_test_k)
                        acc = accuracy_score(y_test_k, y_pred_k)
                        scores.append(acc)
                        self.status_bar.showMessage(f"Fold {fold} Accuracy: {acc:.4f}")
                        fold += 1
                    elif name == "Support Vector Machine":
                        model = SVC(
                            C=param_widgets["C"].value(),
                            kernel=param_widgets["kernel"].currentText(),
                            degree=param_widgets["degree"].value()
                        )
                        model.fit(X_train_k, y_train_k)
                        y_pred_k = model.predict(X_test_k)
                        acc = accuracy_score(y_test_k, y_pred_k)
                        scores.append(acc)

                    elif name == "Decision Tree":
                        model = DecisionTreeClassifier(
                            max_depth=param_widgets["max_depth"].value(),
                            min_samples_split=param_widgets["min_samples_split"].value(),
                            criterion=param_widgets["criterion"].currentText()
                        )
                        model.fit(X_train_k, y_train_k)
                        y_pred_k = model.predict(X_test_k)
                        acc = accuracy_score(y_test_k, y_pred_k)
                        scores.append(acc)

                    elif name == "Random Forest":
                        model = RandomForestClassifier(
                            n_estimators=param_widgets["n_estimators"].value(),
                            max_depth=param_widgets["max_depth"].value(),
                            min_samples_split=param_widgets["min_samples_split"].value()
                        )
                        model.fit(X_train_k, y_train_k)
                        y_pred_k = model.predict(X_test_k)
                        acc = accuracy_score(y_test_k, y_pred_k)
                        scores.append(acc)

                    elif name == "Support Vector Regression":
                        model = SVR(
                            C=param_widgets["C"].value(),
                            epsilon=param_widgets["epsilon"].value(),
                            kernel=param_widgets["kernel"].currentText()
                        )
                        model.fit(X_train_k, y_train_k)
                        y_pred_k = model.predict(X_test_k)
                        mse = mean_squared_error(y_test_k, y_pred_k)
                        rmse = np.sqrt(mse)
                        scores.append(rmse)


                mean_score = np.mean(scores)
                std_score = np.std(scores)
                if name in ["Linear Regression", "Support Vector Regression"]:
                    self.show_message(f"{name} - k-Fold RMSE: {mean_score:.4f} ± {std_score:.4f}")
                else:
                    self.show_message(f"{name} - k-Fold Accuracy: {mean_score:.4f} ± {std_score:.4f}")
                return

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
                return None

            if name == "Linear Regression":
                model = LinearRegression(
                    fit_intercept=param_widgets["fit_intercept"].isChecked()
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                loss = calculate_regression_loss(self.y_test, y_pred)
                self.show_message(f"Linear Regression Loss: {loss:.4f}")

            elif name == "Support Vector Regression":
                model = SVR(
                    C=param_widgets["C"].value(),
                    epsilon=param_widgets["epsilon"].value(),
                    kernel=param_widgets["kernel"].currentText()
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                loss = calculate_regression_loss(self.y_test, y_pred)
                self.show_message(f"SVR Loss: {loss:.4f}")

            elif name == "Logistic Regression":
                model = LogisticRegression(
                    C=param_widgets["C"].value(),
                    max_iter=param_widgets["max_iter"].value(),
                    multi_class=param_widgets["multi_class"].currentText()
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                loss_type = self.classification_loss_combo.currentText()
                if loss_type == "Cross-Entropy":
                    y_proba = model.predict_proba(self.X_test)
                    loss = log_loss(self.y_test, y_proba)
                elif loss_type == "Hinge":
                    loss = hinge_loss(self.y_test, y_pred)

                self.show_message(f"Logistic Regression Loss ({loss_type}): {loss:.4f}")

            elif name == "Support Vector Machine":
                model = SVC(
                    C=param_widgets["C"].value(),
                    kernel=param_widgets["kernel"].currentText(),
                    degree=param_widgets["degree"].value(),
                    probability=True
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)

                loss_type = self.classification_loss_combo.currentText()
                if loss_type == "Cross-Entropy":
                    y_proba = model.predict_proba(self.X_test)
                    loss = log_loss(self.y_test, y_proba)
                elif loss_type == "Hinge":
                    if len(np.unique(self.y_test)) > 2:
                        self.show_error("Hinge Loss is only suitable for binary classification.")
                        return
                    loss = hinge_loss(self.y_test, y_pred)

                self.show_message(f"SVM Loss ({loss_type}): {loss:.4f}")

            elif name == "Naive Bayes":
                smoothing = param_widgets["var_smoothing"].value()
                priors = None

                if self.prior_combo.currentText() == "Custom":
                    try:
                        priors = list(map(float, self.prior_input.text().strip().split(',')))
                    except:
                        self.show_error("Invalid custom priors format. Example: 0.3, 0.7")
                        return

                model = GaussianNB(var_smoothing=smoothing, priors=priors)
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                loss = log_loss(self.y_test, y_proba)

                self.show_message(f"Naive Bayes Loss (Cross-Entropy): {loss:.4f}")

            else:
                self.show_error(f"Model '{name}' is currently not supported.")
                return

            self.current_model = model
            self.update_visualization(y_pred)
            self.update_metrics(y_pred)

        except Exception as e:
            self.show_error(f"Error during training: {str(e)}")

    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)
    
    # Loss and Accuracy Values
    def show_message(self, message):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setText(message)
        msg.setWindowTitle("Model Result")
        msg.exec()   

    def create_deep_learning_tab(self):
        """Create the deep learning tab"""
        widget = QWidget()
        layout = QGridLayout(widget)
        
        # MLP section
        mlp_group = QGroupBox("Multi-Layer Perceptron")
        mlp_layout = QVBoxLayout()
        
        # Layer configuration
        self.layer_config = []
        layer_btn = QPushButton("Add Layer")
        layer_btn.clicked.connect(self.add_layer_dialog)
        mlp_layout.addWidget(layer_btn)
        
        # Training parameters
        training_params_group = self.create_training_params_group()
        mlp_layout.addWidget(training_params_group)
        
        # Train button
        train_btn = QPushButton("Train Neural Network")
        train_btn.clicked.connect(self.train_neural_network)
        mlp_layout.addWidget(train_btn)
        
        mlp_group.setLayout(mlp_layout)
        layout.addWidget(mlp_group, 0, 0)
        
        # CNN section
        cnn_group = QGroupBox("Convolutional Neural Network")
        cnn_layout = QVBoxLayout()
        
        # CNN architecture controls
        cnn_controls = self.create_cnn_controls()
        cnn_layout.addWidget(cnn_controls)
        
        cnn_group.setLayout(cnn_layout)
        layout.addWidget(cnn_group, 0, 1)
        
        # RNN section
        rnn_group = QGroupBox("Recurrent Neural Network")
        rnn_layout = QVBoxLayout()
        
        # RNN architecture controls
        rnn_controls = self.create_rnn_controls()
        rnn_layout.addWidget(rnn_controls)
        
        rnn_group.setLayout(rnn_layout)
        layout.addWidget(rnn_group, 1, 0)
        
        return widget
    
    def add_layer_dialog(self):
        """Open a dialog to add a neural network layer"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Neural Network Layer")
        layout = QVBoxLayout(dialog)
        
        # Layer type selection
        type_layout = QHBoxLayout()
        type_label = QLabel("Layer Type:")
        type_combo = QComboBox()
        type_combo.addItems(["Dense", "Conv2D", "MaxPooling2D", "Flatten", "Dropout"])
        type_layout.addWidget(type_label)
        type_layout.addWidget(type_combo)
        layout.addLayout(type_layout)
        
        # Parameters input
        params_group = QGroupBox("Layer Parameters")
        params_layout = QVBoxLayout()
        
        # Dynamic parameter inputs based on layer type
        self.layer_param_inputs = {}
        
        def update_params():
            # Clear existing parameter inputs
            for widget in list(self.layer_param_inputs.values()):
                params_layout.removeWidget(widget)
                widget.deleteLater()
            self.layer_param_inputs.clear()
            
            layer_type = type_combo.currentText()
            if layer_type == "Dense":
                units_label = QLabel("Units:")
                units_input = QSpinBox()
                units_input.setRange(1, 1000)
                units_input.setValue(32)
                self.layer_param_inputs["units"] = units_input
                
                activation_label = QLabel("Activation:")
                activation_combo = QComboBox()
                activation_combo.addItems(["relu", "sigmoid", "tanh", "softmax"])
                self.layer_param_inputs["activation"] = activation_combo
                
                params_layout.addWidget(units_label)
                params_layout.addWidget(units_input)
                params_layout.addWidget(activation_label)
                params_layout.addWidget(activation_combo)
            
            elif layer_type == "Conv2D":
                filters_label = QLabel("Filters:")
                filters_input = QSpinBox()
                filters_input.setRange(1, 1000)
                filters_input.setValue(32)
                self.layer_param_inputs["filters"] = filters_input
                
                kernel_label = QLabel("Kernel Size:")
                kernel_input = QLineEdit()
                kernel_input.setText("3, 3")
                self.layer_param_inputs["kernel_size"] = kernel_input
                
                params_layout.addWidget(filters_label)
                params_layout.addWidget(filters_input)
                params_layout.addWidget(kernel_label)
                params_layout.addWidget(kernel_input)
            
            elif layer_type == "Dropout":
                rate_label = QLabel("Dropout Rate:")
                rate_input = QDoubleSpinBox()
                rate_input.setRange(0.0, 1.0)
                rate_input.setValue(0.5)
                rate_input.setSingleStep(0.1)
                self.layer_param_inputs["rate"] = rate_input
                
                params_layout.addWidget(rate_label)
                params_layout.addWidget(rate_input)
        
        type_combo.currentIndexChanged.connect(update_params)
        update_params()  # Initial update
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Layer")
        cancel_btn = QPushButton("Cancel")
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        
        def add_layer():
            layer_type = type_combo.currentText()
            
            # Collect parameters
            layer_params = {}
            for param_name, widget in self.layer_param_inputs.items():
                if isinstance(widget, QSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QDoubleSpinBox):
                    layer_params[param_name] = widget.value()
                elif isinstance(widget, QComboBox):
                    layer_params[param_name] = widget.currentText()
                elif isinstance(widget, QLineEdit):
                    # Handle kernel size or other tuple-like inputs
                    if param_name == "kernel_size":
                        layer_params[param_name] = tuple(map(int, widget.text().split(',')))
            
            self.layer_config.append({
                "type": layer_type,
                "params": layer_params
            })
            
            dialog.accept()
        
        add_btn.clicked.connect(add_layer)
        cancel_btn.clicked.connect(dialog.reject)
        
        dialog.exec()
    
    def create_training_params_group(self):
        """Create group for neural network training parameters"""
        group = QGroupBox("Training Parameters")
        layout = QVBoxLayout()
        
        # Batch size
        batch_layout = QHBoxLayout()
        batch_layout.addWidget(QLabel("Batch Size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(32)
        batch_layout.addWidget(self.batch_size_spin)
        layout.addLayout(batch_layout)
        
        # Epochs
        epochs_layout = QHBoxLayout()
        epochs_layout.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(10)
        epochs_layout.addWidget(self.epochs_spin)
        layout.addLayout(epochs_layout)
        
        # Learning rate
        lr_layout = QHBoxLayout()
        lr_layout.addWidget(QLabel("Learning Rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(self.lr_spin)
        layout.addLayout(lr_layout)
        
        group.setLayout(layout)
        return group
    
    def create_cnn_controls(self):
        """Create controls for Convolutional Neural Network"""
        group = QGroupBox("CNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for CNN-specific controls
        label = QLabel("CNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def create_rnn_controls(self):
        """Create controls for Recurrent Neural Network"""
        group = QGroupBox("RNN Architecture")
        layout = QVBoxLayout()
        
        # Placeholder for RNN-specific controls
        label = QLabel("RNN Controls (To be implemented)")
        layout.addWidget(label)
        
        group.setLayout(layout)
        return group
    
    def train_neural_network(self):
        """Train the neural network with current configuration"""
        if not self.layer_config:
            self.show_error("Please add at least one layer to the network")
            return
        
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Prepare data for neural network
            if len(self.X_train.shape) == 1:
                X_train = self.X_train.reshape(-1, 1)
                X_test = self.X_test.reshape(-1, 1)
            else:
                X_train = self.X_train
                X_test = self.X_test
            
            # One-hot encode target for classification
            y_train = tf.keras.utils.to_categorical(self.y_train)
            y_test = tf.keras.utils.to_categorical(self.y_test)
            
            # Compile model
            optimizer = optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
            # Train model
            history = model.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_data=(X_test, y_test),
                                callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
            self.status_bar.showMessage("Neural Network Training Complete")
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
    
    def create_neural_network(self):
        """Create neural network based on current configuration"""
        model = models.Sequential()
        
        # Add layers based on configuration
        for layer_config in self.layer_config:
            layer_type = layer_config["type"]
            params = layer_config["params"]
            
            if layer_type == "Dense":
                model.add(layers.Dense(**params))
            elif layer_type == "Conv2D":
                # Add input shape for the first layer
                if len(model.layers) == 0:
                    params['input_shape'] = self.X_train.shape[1:]
                model.add(layers.Conv2D(**params))
            elif layer_type == "MaxPooling2D":
                model.add(layers.MaxPooling2D())
            elif layer_type == "Flatten":
                model.add(layers.Flatten())
            elif layer_type == "Dropout":
                model.add(layers.Dropout(**params))
        
        # Add output layer based on number of classes
        num_classes = len(np.unique(self.y_train))
        model.add(layers.Dense(num_classes, activation='softmax'))
                
        return model

        
    def train_neural_network(self):
        """Train the neural network"""
        try:
            # Create and compile model
            model = self.create_neural_network()
            
            # Get training parameters
            batch_size = self.batch_size_spin.value()
            epochs = self.epochs_spin.value()
            learning_rate = self.lr_spin.value()
            
            # Compile model
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            
            # Train model
            history = model.fit(self.X_train, self.y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              validation_data=(self.X_test, self.y_test),
                              callbacks=[self.create_progress_callback()])
            
            # Update visualization with training history
            self.plot_training_history(history)
            
        except Exception as e:
            self.show_error(f"Error training neural network: {str(e)}")
            
    def create_progress_callback(self):
        """Create callback for updating progress bar during training"""
        class ProgressCallback(tf.keras.callbacks.Callback):
            def __init__(self, progress_bar):
                super().__init__()
                self.progress_bar = progress_bar
                
            def on_epoch_end(self, epoch, logs=None):
                progress = int(((epoch + 1) / self.params['epochs']) * 100)
                self.progress_bar.setValue(progress)
                
        return ProgressCallback(self.progress_bar)
        
    def update_visualization(self, y_pred):
        """Update the visualization with current results"""
        self.figure.clear()
        
        # Create appropriate visualization based on data
        if len(np.unique(self.y_test)) > 10:  # Regression
            ax = self.figure.add_subplot(111)
            ax.scatter(self.y_test, y_pred)
            ax.plot([self.y_test.min(), self.y_test.max()],
                   [self.y_test.min(), self.y_test.max()],
                   'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            
        else:  # Classification
            if self.X_train.shape[1] > 2:  # Use PCA for visualization
                pca = PCA(n_components=2)
                X_test_2d = pca.fit_transform(self.X_test)
                
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
                
            else:  # Direct 2D visualization
                ax = self.figure.add_subplot(111)
                scatter = ax.scatter(self.X_test[:, 0], self.X_test[:, 1],
                                   c=y_pred, cmap='viridis')
                self.figure.colorbar(scatter)
        
        self.canvas.draw()
        
    def update_metrics(self, y_pred):
        """Update metrics display"""
        metrics_text = "Model Performance Metrics:\n\n"
        
        # Calculate appropriate metrics based on problem type
        if len(np.unique(self.y_test)) > 10:  # Regression
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = self.current_model.score(self.X_test, self.y_test)
            
            metrics_text += f"Mean Squared Error: {mse:.4f}\n"
            metrics_text += f"Root Mean Squared Error: {rmse:.4f}\n"
            metrics_text += f"R² Score: {r2:.4f}"
            
        else:  # Classification
            accuracy = accuracy_score(self.y_test, y_pred)
            conf_matrix = confusion_matrix(self.y_test, y_pred)
            
            metrics_text += f"Accuracy: {accuracy:.4f}\n\n"
            metrics_text += "Confusion Matrix:\n"
            metrics_text += str(conf_matrix)
        
        self.metrics_text.setText(metrics_text)
        
    def plot_training_history(self, history):
        """Plot neural network training history"""
        self.figure.clear()
        
        # Plot training & validation accuracy
        ax1 = self.figure.add_subplot(211)
        ax1.plot(history.history['accuracy'])
        ax1.plot(history.history['val_accuracy'])
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.legend(['Train', 'Test'])
        
        # Plot training & validation loss
        ax2 = self.figure.add_subplot(212)
        ax2.plot(history.history['loss'])
        ax2.plot(history.history['val_loss'])
        ax2.set_title('Model Loss')
        ax2.set_ylabel('Loss')
        ax2.set_xlabel('Epoch')
        ax2.legend(['Train', 'Test'])
        
        self.figure.tight_layout()
        self.canvas.draw()

    def setup_dimensionality_reduction_tools(self):
        """Set up extended Dimensionality Reduction tools"""
        tab = self.tab_widget.widget(2).widget()  # Dimensionality Reduction tab
        layout = tab.layout()

        # --- PCA Section ---
        pca_group = QGroupBox("PCA (Principal Component Analysis)")
        pca_layout = QVBoxLayout()

        self.pca_components_spin = QSpinBox()
        self.pca_components_spin.setRange(1, 100)
        self.pca_components_spin.setValue(2)
        pca_layout.addWidget(QLabel("Number of Components:"))
        pca_layout.addWidget(self.pca_components_spin)

        run_pca_btn = QPushButton("Run PCA")
        run_pca_btn.clicked.connect(self.run_pca)
        pca_layout.addWidget(run_pca_btn)

        # Button: PCA 1D Projection Example from covariance matrix
        demo_btn = QPushButton("PCA Projection Example (Σ matrix)")
        demo_btn.clicked.connect(self.demo_covariance_projection)
        pca_layout.addWidget(demo_btn)

        pca_group.setLayout(pca_layout)
        layout.addWidget(pca_group, 1, 0)

        # --- LDA Section ---
        lda_group = QGroupBox("LDA (Linear Discriminant Analysis)")
        lda_layout = QVBoxLayout()

        run_lda_btn = QPushButton("Run LDA")
        run_lda_btn.clicked.connect(self.run_lda)
        lda_layout.addWidget(run_lda_btn)

        lda_group.setLayout(lda_layout)
        layout.addWidget(lda_group, 1, 1)

        # --- t-SNE Section ---
        tsne_group = QGroupBox("t-SNE (t-Distributed Stochastic Neighbor Embedding)")
        tsne_layout = QVBoxLayout()

        plotly_btn = QPushButton("Run t-SNE (Interactive - Plotly)")
        plotly_btn.clicked.connect(self.run_tsne_plotly)
        tsne_layout.addWidget(plotly_btn)

        # Projection type (2D/3D)
        self.tsne_dim_combo = QComboBox()
        self.tsne_dim_combo.addItems(["2D", "3D"])
        tsne_layout.addWidget(QLabel("Projection Type:"))
        tsne_layout.addWidget(self.tsne_dim_combo)

        self.tsne_perplexity_spin = QDoubleSpinBox()
        self.tsne_perplexity_spin.setRange(5.0, 50.0)
        self.tsne_perplexity_spin.setValue(30.0)
        self.tsne_perplexity_spin.setSingleStep(1.0)
        tsne_layout.addWidget(QLabel("Perplexity:"))
        tsne_layout.addWidget(self.tsne_perplexity_spin)

        run_tsne_btn = QPushButton("Run t-SNE")
        run_tsne_btn.clicked.connect(self.run_tsne)
        tsne_layout.addWidget(run_tsne_btn)

        tsne_group.setLayout(tsne_layout)
        layout.addWidget(tsne_group, 2, 0)

        # --- UMAP Section ---
        umap_group = QGroupBox("UMAP (Uniform Manifold Approximation and Projection)")
        umap_layout = QVBoxLayout()

        run_umap_btn = QPushButton("Run UMAP")
        run_umap_btn.clicked.connect(self.run_umap)
        umap_layout.addWidget(run_umap_btn)

        run_umap_plotly_btn = QPushButton("Run UMAP (Interactive - Plotly)")
        run_umap_plotly_btn.clicked.connect(self.run_umap_plotly)
        umap_layout.addWidget(run_umap_plotly_btn)

        umap_group.setLayout(umap_layout)
        layout.addWidget(umap_group, 2, 2)

        # --- KMeans Section ---
        kmeans_group = QGroupBox("K-Means Clustering")
        kmeans_layout = QVBoxLayout()

        self.kmeans_k_spin = QSpinBox()
        self.kmeans_k_spin.setRange(1, 20)
        self.kmeans_k_spin.setValue(3)
        kmeans_layout.addWidget(QLabel("Number of Clusters (k):"))
        kmeans_layout.addWidget(self.kmeans_k_spin)

        run_kmeans_btn = QPushButton("Run KMeans")
        run_kmeans_btn.clicked.connect(self.run_kmeans)
        kmeans_layout.addWidget(run_kmeans_btn)

        kmeans_group.setLayout(kmeans_layout)
        layout.addWidget(kmeans_group, 2, 1)
   
    def run_pca(self):
        """Run PCA and plot explained variance"""
        try:
            pca = PCA(n_components=self.pca_components_spin.value())
            X_pca = pca.fit_transform(self.X_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
            ax.set_xlabel('Principal Components')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('PCA Explained Variance')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error running PCA: {str(e)}")

    def run_lda(self):
        """Run LDA and plot class separation"""
        try:
            if self.y_train is None:
                self.show_error("LDA requires target labels (y_train)!")
                return
            
            lda = LDA(n_components=2)
            X_lda = lda.fit_transform(self.X_train, self.y_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=self.y_train, cmap='jet')
            self.figure.colorbar(scatter)
            ax.set_title('LDA Class Separation')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error running LDA: {str(e)}")

    def run_tsne(self):
        """Run t-SNE and plot"""
        try:
            tsne = TSNE(n_components=2, perplexity=self.tsne_perplexity_spin.value(), random_state=42)
            X_tsne = tsne.fit_transform(self.X_train)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=self.y_train if self.y_train is not None else 'b', cmap='viridis')
            self.figure.colorbar(scatter)
            ax.set_title('t-SNE Projection')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error running t-SNE: {str(e)}")

    def run_kmeans(self):
        """Run K-Means and plot Elbow Method"""
        try:
            inertias = []
            Ks = range(1, 11)
            for k in Ks:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                kmeans.fit(self.X_train)
                inertias.append(kmeans.inertia_)

            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.plot(Ks, inertias, 'bo-')
            ax.set_xlabel('Number of Clusters (k)')
            ax.set_ylabel('Inertia')
            ax.set_title('KMeans Elbow Method')
            self.canvas.draw()
        except Exception as e:
            self.show_error(f"Error running KMeans: {str(e)}")
 
    def show_error(self, message):
        """Show error message dialog"""
        QMessageBox.critical(self, "Error", message)

def main():
    """Main function to start the application"""
    app = QApplication(sys.argv)
    window = MLCourseGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

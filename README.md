
#  Assignment 3: Deep Learning & Advanced Training Techniques

In this final extension, the GUI was enhanced to support **designing, training, and evaluating deep learning models interactively**. The system supports modern techniques such as CNNs, RNNs, transfer learning, gradient visualization, and advanced training control via the GUI.

---

##  Neural Network Architecture Design

Users can dynamically build neural network models by stacking different layer types via GUI components. Each layer can be configured (units, activation, dropout, etc.) and added/removed interactively.

###  Multi-Layer Perceptrons (MLPs)

MLPs consist of stacked fully-connected (Dense) layers. They are fundamental in deep learning for structured data.

```python
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
```

The GUI allows users to add these layers in sequence, choose activations (ReLU, Sigmoid, Tanh), and visualize the layer stack.

---

###  Convolutional Neural Networks (CNNs)

CNNs are specialized for image data, capturing spatial hierarchies through filters. Users can add Conv2D, MaxPooling2D, Flatten, and Dropout layers via the interface.

```python
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
```

The GUI automatically reshapes 1D inputs into 2D image tensors when the feature size is a perfect square (e.g., 784 → 28×28).

---

###  Recurrent Neural Networks (LSTM / GRU)

RNNs process sequential data like time-series or text. The interface allows users to insert LSTM or GRU layers with optional `return_sequences` and dropout.

```python
model.add(LSTM(64, return_sequences=True))
model.add(GRU(32))
```

Layers are listed visually and can be removed or reordered.

---

##  Training Customization

The training panel allows full control over optimization strategy and model regularization.

###  Optimizer Selection

Users can choose between **Adam**, **SGD**, or **RMSprop**, each suited for different convergence behaviors.

```python
if optimizer == "Adam":
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
```

---

###  Learning Rate Decay (Scheduling)

Learning rate schedules adjust how fast the model learns over time:

- **Step Decay**: halves learning rate at regular intervals.
- **Exponential Decay**: continuously decreases the rate exponentially.

```python
def step_decay(epoch):
    return initial_lr * (0.5 ** (epoch // 10))
```

The GUI dropdown lets users activate these schedules.

---

###  Regularization: Dropout & L2

To prevent overfitting, users can add:

- **Dropout**: Randomly zeroes neurons during training.
- **L2 Regularization**: Penalizes large weights using a lambda factor.

```python
Dropout(0.5)
Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

These settings are defined through GUI components for each layer.

---

###  Early Stopping

Monitors validation loss and halts training if no improvement is observed for several epochs.

```python
EarlyStopping(monitor='val_loss', patience=5)
```

This is configurable via a numeric spinbox in the GUI.

---

##  Visualization Tools

###  Training Curves

The GUI plots both **accuracy and loss curves** for training and validation sets.

```python
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
```

This helps users detect underfitting or overfitting visually.

---

###  Gradient Histogram

After each epoch, a histogram of **weight gradients** is displayed using matplotlib to analyze gradient magnitude distribution.

```python
with tf.GradientTape() as tape:
    loss = model.compiled_loss(y_true, y_pred)
grads = tape.gradient(loss, model.trainable_weights)
```

This gives insight into vanishing/exploding gradients during training.

---

##  Evaluation Metrics

After training, the system computes and displays test set metrics:

- **Accuracy**
- **F1 Score**
- (Optional: Confusion Matrix)

```python
accuracy_score(y_true, y_pred)
f1_score(y_true, y_pred, average='weighted')
```

These are shown in the GUI for quick interpretation.

---

##  Data Augmentation (for Images)

For image inputs, the GUI allows on-the-fly augmentation using:

- **Random rotation** (up to 30°)
- **Zoom** (up to 20%)
- **Horizontal flip**

```python
ImageDataGenerator(
    rotation_range=30,
    zoom_range=0.2,
    horizontal_flip=True
)
```

This increases model generalization during training.

---

##  Transfer Learning

Supports loading pre-trained **VGG16** or **ResNet50** models (without top layers), freezing the convolutional base, and adding custom dense layers for fine-tuning.

```python
base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])
```

Students can control dropout, dense units, and freeze option via GUI.

---

###  Model Saving & Loading

- Save full model as `.h5`
- Save only architecture as `.json`
- Load architecture and weights separately

```python
model.save("model.h5")
model.to_json()
model.load_weights("weights.h5")
```

These options allow checkpointing and resuming training across sessions.

---

##  Conclusion

This final assignment brings the GUI into full production-level capabilities by supporting:
- Custom deep neural networks,
- CNNs, RNNs, transfer learning,
- Gradient diagnostics,
- Advanced training schedules and optimizers,
- Visual tracking of performance and generalization.

All features are exposed through an intuitive GUI for educational exploration.

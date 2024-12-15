import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

filepath = "../alzheimer_masked_nn/alzheimers_disease_data.csv"
data = pd.read_csv(filepath)

# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from tensorflow.keras import layers, Model
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm


class AlzheimersDataPreprocessor:
    def __init__(self):
        self.scalers = {}

    def preprocess(self, data, is_training=True):
        """
        Preprocess the Alzheimer's dataset
        """
        # Create a copy to avoid modifying original data
        df = data.copy()

        # 1. Remove non-predictive columns
        df = df.drop(["PatientID", "DoctorInCharge"], axis=1)

        # 2. Separate features and target
        if "Diagnosis" in df.columns:
            y = df["Diagnosis"].values  # Convert to numpy array
            X = df.drop("Diagnosis", axis=1)
        else:
            X = df
            y = None

        # 3. Define column groups
        numerical_columns = [
            "Age",
            "BMI",
            "AlcoholConsumption",
            "PhysicalActivity",
            "DietQuality",
            "SleepQuality",
            "SystolicBP",
            "DiastolicBP",
            "CholesterolTotal",
            "CholesterolLDL",
            "CholesterolHDL",
            "CholesterolTriglycerides",
            "MMSE",
            "FunctionalAssessment",
            "ADL",
        ]

        # 4. Scale numerical features
        for col in numerical_columns:
            if is_training:
                self.scalers[col] = StandardScaler()
                X[col] = self.scalers[col].fit_transform(X[col].values.reshape(-1, 1))
            else:
                X[col] = self.scalers[col].transform(X[col].values.reshape(-1, 1))

        # 5. Create mask for missing values
        mask = ~X.isna()

        # 6. Fill missing values with 0 (they'll be masked anyway)
        X = X.fillna(0)

        # Convert to numpy arrays
        X = X.values
        mask = mask.values

        if y is not None:
            return X, mask, y
        return X, mask


def prepare_data(data, test_size=0.2, random_state=42):
    """
    Prepare the dataset for training
    """
    # Create preprocessor
    preprocessor = AlzheimersDataPreprocessor()

    # Split data
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["Diagnosis"]
    )

    # Preprocess training data
    X_train, masks_train, y_train = preprocessor.preprocess(
        train_data, is_training=True
    )

    # Preprocess test data using fitted preprocessor
    X_test, masks_test, y_test = preprocessor.preprocess(test_data, is_training=False)

    return (X_train, X_test, masks_train, masks_test, y_train, y_test, preprocessor)


class MissingValueNetwork:
    def __init__(self, input_dim, hidden_dims=[128, 64, 32]):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.model = self._build_model()

    def _build_model(self):
        # Main input for features
        main_input = layers.Input(shape=(self.input_dim,), name="main_input")
        mask_input = layers.Input(shape=(self.input_dim,), name="mask_input")

        # Branch 1: Process available values with more capacity
        masked_input = layers.Multiply()([main_input, mask_input])

        # Deeper network for main branch
        x1 = masked_input
        for i, dim in enumerate(self.hidden_dims):
            x1 = layers.Dense(dim, activation="relu")(x1)
            x1 = layers.BatchNormalization()(x1)
            x1 = layers.Dropout(0.3)(x1)
            # Add residual connection if dimensions match
            if i > 0 and self.hidden_dims[i] == self.hidden_dims[i - 1]:
                x1 = layers.Add()([x1, previous_x1])
            previous_x1 = x1

        # Branch 2: Process missing patterns
        x2 = mask_input
        for dim in self.hidden_dims:
            x2 = layers.Dense(dim // 2, activation="relu")(x2)
            x2 = layers.BatchNormalization()(x2)
            x2 = layers.Dropout(0.3)(x2)

        # Combine branches with attention
        combined = layers.Concatenate()([x1, x2])

        # Add attention mechanism
        attention = layers.Dense(combined.shape[-1], activation="tanh")(combined)
        attention = layers.Dense(1, activation="sigmoid")(attention)
        combined = layers.Multiply()([combined, attention])

        # Final dense layers with skip connections
        x = layers.Dense(64, activation="relu")(combined)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(32, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        # Output layer
        output = layers.Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[main_input, mask_input], outputs=output)

        # Use a different optimizer configuration
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True
        )

        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=[
                "accuracy",
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Precision(name="precision"),
                tf.keras.metrics.Recall(name="recall"),
                tf.keras.metrics.F1Score(name="f1", threshold=0.5),
            ],
        )

        return model

    def train(
        self,
        X_train,
        masks_train,
        y_train,
        validation_data,
        epochs=50,
        batch_size=32,
        class_weights=None,
    ):
        X_val, masks_val, y_val = validation_data

        # Ensure numpy arrays
        y_train = np.array(y_train)
        y_val = np.array(y_val)

        # Calculate more balanced class weights
        if class_weights is None:
            unique_classes = np.unique(y_train)
            n_samples = len(y_train)
            class_weights = {}
            for cls in unique_classes:
                class_weights[cls] = float(
                    n_samples / (len(unique_classes) * np.sum(y_train == cls))
                )

        print("Class weights:", class_weights)

        # Add learning rate scheduler
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=1
        )

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True, verbose=1
        )

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            "best_model.keras", monitor="val_loss", save_best_only=True, verbose=1
        )

        # Add custom callback for better progress tracking
        class MetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                print(f"\nEpoch {epoch + 1} metrics:")
                metrics_to_print = ["loss", "accuracy", "precision", "recall", "f1"]
                for metric in metrics_to_print:
                    if metric in logs:
                        print(f"{metric}: {logs[metric]:.4f}")
                    if f"val_{metric}" in logs:
                        print(f"val_{metric}: {logs[f'val_{metric}']:.4f}")

        history = self.model.fit(
            [X_train, masks_train],
            y_train,
            validation_data=([X_val, masks_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weights,
            callbacks=[early_stopping, checkpoint, lr_scheduler, MetricsCallback()],
            verbose=1,
        )

        return history

    def predict(self, X, masks, threshold=0.5):
        """
        Make predictions with threshold tuning
        """
        probs = self.model.predict([X, masks])
        return probs, (probs > threshold).astype(int)

    def find_best_threshold(self, X_val, masks_val, y_val):
        """
        Find the optimal decision threshold using validation data
        """
        probs = self.model.predict([X_val, masks_val])

        # Try different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        print("\nFinding best threshold:")
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            f1 = f1_score(y_val, preds)
            print(f"Threshold: {threshold:.2f}, F1: {f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold


def plot_training_history(history):
    """
    Plot training history
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history.history["accuracy"])
    ax1.plot(history.history["val_accuracy"])
    ax1.set_title("Model Accuracy")
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend(["Train", "Validation"])

    # Plot loss
    ax2.plot(history.history["loss"])
    ax2.plot(history.history["val_loss"])
    ax2.set_title("Model Loss")
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend(["Train", "Validation"])

    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Load data
    filepath = "../alzheimer_masked_nn/alzheimers_disease_data.csv"
    data = pd.read_csv(filepath)

    # Prepare data
    X_train, X_test, masks_train, masks_test, y_train, y_test, preprocessor = (
        prepare_data(data)
    )

    # Print data information
    print("Data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"masks_train: {masks_train.shape}")
    print(f"masks_test: {masks_test.shape}")
    print("\nClass distribution:")
    print("Training set:", np.bincount(y_train))
    print("Test set:", np.bincount(y_test))

    # Initialize model
    model = MissingValueNetwork(input_dim=X_train.shape[1])

    # Train the model
    history = model.train(
        X_train,
        masks_train,
        y_train,
        validation_data=(X_test, masks_test, y_test),
        epochs=100,
        batch_size=32,
    )

    # Plot training history
    plot_training_history(history)

    # Find the best threshold
    best_threshold = model.find_best_threshold(X_test, masks_test, y_test)
    print(f"\nBest threshold: {best_threshold}")

    # Make predictions with the optimal threshold
    probs, y_pred = model.predict(X_test, masks_test, threshold=best_threshold)

    # Print final evaluation
    print("\nFinal Classification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

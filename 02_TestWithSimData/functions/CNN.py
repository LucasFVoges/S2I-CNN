import numpy as np
import pandas as pd
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


if __name__ == "__main__":
    pass

def calculate_metrics(y_true, y_pred_prob):
    y_pred = (y_pred_prob > 0.5).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0)
    }
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=["Actual First", "Actual Second"], columns=["Pred First", "Pred Second"])
    return metrics, cm_df

def train_simple_cnn(images, labels, epochs=5, batch_size=16):
    # Convert list of PIL images to a single numpy array
    # Normalize images to 0-1 range
    X = np.array([np.array(img).astype('float32') / 255.0 for img in images])

    # If height is 1, X shape is (Samples, 1, Width, Channels) or (Samples, 1, Width)
    # We want to squeeze out that 1-pixel height dimension for a 1D CNN
    if len(X.shape) == 4: # (Samples, Height, Width, Channels)
        X = np.squeeze(X, axis=1)
    elif len(X.shape) == 3: # (Samples, Height, Width)
        X = np.squeeze(X, axis=1)
        X = np.expand_dims(X, axis=-1) # Add channel dim back
        
    # Convert labels to numeric (assuming 2 classes for now)
    y = np.array([0 if l == "First Class" else 1 for l in labels])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # random_state=42
    
    # Simple CNN Model
    model = models.Sequential([
        layers.Conv1D(8, 3, activation='relu', input_shape=X.shape[1:]),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=0)

    train_metrics, train_cm = calculate_metrics(y_train, model.predict(X_train, verbose=0))
    test_metrics, test_cm = calculate_metrics(y_test, model.predict(X_test, verbose=0))

    return model, history, (train_metrics, test_metrics, train_cm, test_cm)

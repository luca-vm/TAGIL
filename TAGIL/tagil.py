import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from load_tagil import *
import time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 1000
lr = 1.0
r = 0.95
dropout = 0.3
num_action = 18
regularization_factor=0.0
epsilon=1e-08
Durations = True
SW = True
Game = "bank_heist"
type = "TAGIL"
weighted = True
sample_weighting = 2


# Initialize WandB run
run = wandb.init(
    project="AGIL-FINAL",
    config={
        "batch_size": BATCH_SIZE,
        "epochs": num_epoch,
        "optimizer": "Adadelta",
        "learning_rate": lr,
        "rho": r,
        "dropout": dropout,
        "epsilon": epsilon,
        "buffer-size" : BUFF_SIZE,
        "regularization_factor": regularization_factor,
        "Durations": Durations,
        "Sample_weighing": SW,
        "Game": Game,
        "Model": type,
        "weighted": weighted,
        "sample_weighting": sample_weighting
    }
)




def create_saliency_model(SHAPE=(84, 84, 1), regularization_factor=0.01, dropout=0.3, num_action=18):
    gaze_heatmaps = L.Input(shape=(SHAPE[0], SHAPE[1], 1))
    g = gaze_heatmaps
    g = L.BatchNormalization()(g)

    imgs = L.Input(shape=SHAPE)
    x = imgs
    x = L.Multiply()([x, g])
    x_intermediate = x
    x = L.Conv2D(32, (8,8), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.Dropout(dropout)(x)

    x = L.Conv2D(64, (4,4), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)
    x = L.Dropout(dropout)(x)

    x = L.Conv2D(64, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = L.BatchNormalization()(x)
    x = L.Activation('relu')(x)

    # ============================ channel 2 ============================
    orig_x = imgs
    orig_x = L.Conv2D(32, (8,8), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x = L.BatchNormalization()(orig_x)
    orig_x = L.Activation('relu')(orig_x)
    orig_x = L.Dropout(dropout)(orig_x)

    orig_x = L.Conv2D(64, (4,4), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x = L.BatchNormalization()(orig_x)
    orig_x = L.Activation('relu')(orig_x)
    orig_x = L.Dropout(dropout)(orig_x)

    orig_x = L.Conv2D(64, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x = L.BatchNormalization()(orig_x)
    orig_x = L.Activation('relu')(orig_x)

    x = L.Average()([x, orig_x])
    x = L.Dropout(dropout)(x)
    x = L.Flatten()(x)

    # Add input for difficulty classification
    difficulty = L.Input(shape=(2,))
    
    # Concatenate flattened convolutional features with difficulty classification
    x = L.Concatenate()([x, difficulty])

    x = L.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x = L.Dropout(dropout)(x)
    logits = L.Dense(num_action, name="logits", kernel_regularizer=regularizers.l2(regularization_factor))(x)
    prob = L.Activation('softmax', name="prob")(logits)

    model = Model(inputs=[imgs, gaze_heatmaps, difficulty], outputs=prob)
    
    print("Model created")
    return model


# Define a custom callback for logging
class TrainingLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Retrieve metrics
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        train_accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')
        train_weighted_loss = logs.get('weighted_loss')
        val_weighted_loss = logs.get('val_weighted_loss')
        train_weighted_accuracy = logs.get('weighted_accuracy')
        val_weighted_accuracy = logs.get('val_weighted_accuracy')

        # Log to WandB
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_weighted_loss": train_weighted_loss,
            "val_weighted_loss": val_weighted_loss,
            "train_weighted_accuracy": train_weighted_accuracy,
            "val_weighted_accuracy": val_weighted_accuracy
        })

        # Print metrics
        print(f"\nEpoch {epoch + 1}")
        print(f"Regular Metrics:")
        print(f"  Train Loss: {train_loss}, Train Accuracy: {train_accuracy}")
        print(f"  Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        print(f"Weighted Metrics:")
        print(f"  Train Weighted Loss: {train_weighted_loss}, Train Weighted Accuracy: {train_weighted_accuracy}")
        print(f"  Val Weighted Loss: {val_weighted_loss}, Val Weighted Accuracy: {val_weighted_accuracy}")

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=40,
    verbose=1,
    mode='min',
    restore_best_weights=True
)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def prepare_test_data(test_dataset):
    images, heatmaps, difficulties, labels = [], [], [], []
    for (img, heatmap, difficulty), label in test_dataset:
        images.append(img)
        heatmaps.append(heatmap)
        difficulties.append(difficulty)
        labels.append(label)
    return np.array(images), np.array(heatmaps), np.array(difficulties), np.array(labels)

def predict(model, images, heatmaps, difficulties):
    return model.predict([images, heatmaps, difficulties])

def analyze_performance(y_true, y_pred, difficulties):
    easy_mask = difficulties[:, 0] == 1
    hard_mask = difficulties[:, 1] == 1

    y_true_easy = y_true[easy_mask]
    y_pred_easy = y_pred[easy_mask]
    y_true_hard = y_true[hard_mask]
    y_pred_hard = y_pred[hard_mask]

    cm_easy = confusion_matrix(y_true_easy, y_pred_easy)
    cm_hard = confusion_matrix(y_true_hard, y_pred_hard)

    print("Classification Report for Easy States:")
    print(classification_report(y_true_easy, y_pred_easy))
    print("\nClassification Report for Hard States:")
    print(classification_report(y_true_hard, y_pred_hard))

    return cm_easy, cm_hard

def print_confusion_matrix(cm, title):
    print(f"\n{title}")
    print(cm)
    print(f"Shape: {cm.shape}")
        
def main():
    train_tar_file = '/datasets/lvmayer/bank_heist/train/train.tar.bz2'
    train_npz_file = '/datasets/lvmayer/bank_heist/gaze_smote_train.npz'
    train_label_file = '/datasets/lvmayer/bank_heist/train/train.txt'
    
    val_tar_file = '/datasets/lvmayer/bank_heist/val/val.tar.bz2'
    val_npz_file = '/datasets/lvmayer/bank_heist/gaze_smote_val.npz'
    val_label_file = '/datasets/lvmayer/bank_heist/val/val.txt'
    
    test_tar_file = '/datasets/lvmayer/bank_heist/test/test.tar.bz2'
    test_npz_file = '/datasets/lvmayer/bank_heist/gaze_smote_test.npz'
    test_label_file = '/datasets/lvmayer/bank_heist/test/test.txt'
    
    
    t1 = time.time()
    
    train_dataset = Dataset(train_tar_file, train_npz_file, train_label_file, sample_weighting)
    val_dataset = Dataset(val_tar_file, val_npz_file, val_label_file, sample_weighting)
    
    train_tf_dataset = train_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    val_tf_dataset = val_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    
    # Calculate steps per epoch
    train_steps_per_epoch = train_dataset.get_steps_per_epoch(BATCH_SIZE)
    val_steps = val_dataset.get_steps_per_epoch(BATCH_SIZE)
    
    print(f"Time spent loading and preprocessing: {time.time() - t1:.1f}s")

    # Create and compile the model
    model = create_saliency_model()
    
    opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
    model.compile(
        loss=K.losses.sparse_categorical_crossentropy,
        optimizer=opt,
        metrics=['accuracy'],
        weighted_metrics=['accuracy']  # Add this line
    )
    print("Compiled")

    callbacks = [TrainingLogger(), WandbMetricsLogger(log_freq=5), early_stopping]
    
    print(f"Starting model training. Steps per epoch: {train_steps_per_epoch}")
    history = model.fit(
        train_tf_dataset,
        validation_data=val_tf_dataset,
        epochs=num_epoch,
        steps_per_epoch=train_steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks
    )

    del train_dataset
    del val_dataset
    
    
    run_name = run.name
    # Save the model
    model.save(f"/datasets/lvmayer/bank_heist/{run_name}.hdf5")
    
    # Load test data
    test_dataset = Dataset(test_tar_file, test_npz_file, test_label_file, sample_weighting)
    test_tf_dataset = test_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    test_steps = test_dataset.get_steps_per_epoch(BATCH_SIZE)

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_tf_dataset, steps=test_steps, verbose=2)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_accuracy})
    
    # Prepare the test data
    images, heatmaps, difficulties, true_labels = prepare_test_data(test_dataset)

    # Make predictions
    predictions = predict(model, images, heatmaps, difficulties)
    predicted_labels = np.argmax(predictions, axis=1)

    # Analyze performance
    cm_easy, cm_hard = analyze_performance(true_labels, predicted_labels, difficulties)

    # Print confusion matrices
    print_confusion_matrix(cm_easy, "Confusion Matrix for Easy States")
    print_confusion_matrix(cm_hard, "Confusion Matrix for Hard States")

    # Calculate overall accuracy for easy and hard states
    accuracy_easy = np.sum(np.diag(cm_easy)) / np.sum(cm_easy)
    accuracy_hard = np.sum(np.diag(cm_hard)) / np.sum(cm_hard)

    print(f"\nOverall accuracy for easy states: {accuracy_easy:.4f}")
    print(f"Overall accuracy for hard states: {accuracy_hard:.4f}")
    wandb.log({"accuracy_easy": accuracy_easy, "accuracy_hard": accuracy_hard})

    # Print distribution of easy and hard states
    total_samples = len(true_labels)
    easy_samples = np.sum(difficulties[:, 0])
    hard_samples = np.sum(difficulties[:, 1])
    
    print(f"\nDistribution of states:")
    print(f"Easy states: {easy_samples} ({easy_samples/total_samples:.2%})")
    print(f"Hard states: {hard_samples} ({hard_samples/total_samples:.2%})")
    wandb.log({"easy_samples": easy_samples, "hard_samples": hard_samples})

    # Finish the WandB run
    wandb.finish()

  

if __name__ == "__main__":
    main()
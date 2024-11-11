import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from load_agil import *
import time


# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 15
lr = 1.0
r = 0.95
dropout = 0.3
num_action = 18
regularization_factor=0.0
epsilon=1e-08
Game = "ms_pacman"
type = "AGIL"
stadardise = False

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
        "Game": Game,
        "Model": type,
        "Standardise": stadardise
    }
)



def create_saliency_model(SHAPE=(84, 84, 1), regularization_factor=regularization_factor, dropout= dropout):
     ###############################
    # Architecture of the network #
    ###############################

    gaze_heatmaps = L.Input(shape=(SHAPE[0],SHAPE[1],1))
    g=gaze_heatmaps
    g=L.BatchNormalization()(g)

    imgs=L.Input(shape=SHAPE)
    x=imgs
    x=L.Multiply()([x,g])
    x_intermediate=x
    x=L.Conv2D(32, (8,8), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)

    x=L.Conv2D(64, (4,4), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    x=L.Dropout(dropout)(x)

    x=L.Conv2D(64, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(x)
    x=L.BatchNormalization()(x)
    x=L.Activation('relu')(x)
    # ============================ channel 2 ============================
    orig_x=imgs
    orig_x=L.Conv2D(32, (8,8), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (4,4), strides=2, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)
    orig_x=L.Dropout(dropout)(orig_x)

    orig_x=L.Conv2D(64, (3,3), strides=1, padding='same', kernel_regularizer=regularizers.l2(regularization_factor))(orig_x)
    orig_x=L.BatchNormalization()(orig_x)
    orig_x=L.Activation('relu')(orig_x)

    x=L.Average()([x,orig_x])
    x=L.Dropout(dropout)(x)
    x=L.Flatten()(x)
    x=L.Dense(512, activation='relu')(x)
    x=L.Dropout(dropout)(x)
    logits=L.Dense(num_action, name="logits")(x)
    prob=L.Activation('softmax', name="prob")(logits)

    model=Model(inputs=[imgs, gaze_heatmaps], outputs=prob)
    
    
    print("model created")
    return model


# Define a custom callback for logging
class TrainingLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        wandb.log({
            "train_loss": loss,
            "val_loss": val_loss,
            "train_accuracy": accuracy,
            "val_accuracy": val_accuracy
        })
        
        print(f"Epoch {epoch + 1}: loss = {loss:.4f}, val_loss = {val_loss:.4f}, accuracy = {accuracy:.4f}, val_accuracy = {val_accuracy:.4f}")

early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )



        
def main():
    train_tar_file = '/datasets/lvmayer/ms_pacman/train/train.tar.bz2'
    train_npz_file = '/datasets/lvmayer/ms_pacman/gaze_train.npz'
    train_label_file = '/datasets/lvmayer/ms_pacman/train/train.txt'
    
    val_tar_file = '/datasets/lvmayer/ms_pacman/val/val.tar.bz2'
    val_npz_file = '/datasets/lvmayer/ms_pacman/gaze_val.npz'
    val_label_file = '/datasets/lvmayer/ms_pacman/val/val.txt'
    
    test_tar_file = '/datasets/lvmayer/ms_pacman/test/test.tar.bz2'
    test_npz_file = '/datasets/lvmayer/ms_pacman/gaze_test.npz'
    test_label_file = '/datasets/lvmayer/ms_pacman/test/test.txt'
    
   
    
    t1 = time.time()
    
    train_dataset = Dataset(train_tar_file, train_npz_file, train_label_file)
    val_dataset = Dataset(val_tar_file, val_npz_file, val_label_file)
    
    train_tf_dataset = train_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    val_tf_dataset = val_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    
    # Calculate steps per epoch
    train_steps_per_epoch = train_dataset.get_steps_per_epoch(BATCH_SIZE)
    val_steps = val_dataset.get_steps_per_epoch(BATCH_SIZE)
    
    print(f"Time spent loading and preprocessing: {time.time() - t1:.1f}s")

    # Create and compile the model
    model = create_saliency_model()
    
    opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
    model.compile(loss=K.losses.sparse_categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
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
    
    # Load test data
    test_dataset = Dataset(test_tar_file, test_npz_file, test_label_file)
    test_tf_dataset = test_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    test_steps = test_dataset.get_steps_per_epoch(BATCH_SIZE)

    test_loss, test_acc = model.evaluate(test_tf_dataset, steps=test_steps, verbose=2)
    print(f'Test Loss: {test_loss}, Test Accuracy: {test_acc}')
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

    # Finish the WandB run
    wandb.finish()

    run_name = run.name
    # Save the model
    model.save(f"/datasets/lvmayer/ms_pacman/{run_name}.hdf5")

if __name__ == "__main__":
    main()
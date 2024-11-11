
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras_cv.losses import FocalLoss
from sklearn.metrics import confusion_matrix
# from load_onlyinit import *
from load_tgaze import *
from smote import *
import time


# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 100
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.01
epsilon=1e-08
gaze_weight = 0.3
alpha = 0.25
gamma = 2.0
minority_increase = 10
Game = "berzerk"
type = "Temporal Gaze"

# Initialize WandB run
run = wandb.init(
    project="Gaze-Prediction-FINAL",
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
        "gaze_weight": gaze_weight,
        "alpha": alpha,
        "gamma": gamma,
        "minority_increase" : minority_increase,
        "Game": Game,
        "Model": type
    
    }
)

def nss_metric(y_true, y_pred):
    # Reshape the inputs to (batch_size, 84, 84)
    y_true = tf.reshape(y_true[:, :84*84], (-1, 84, 84))
    y_pred = tf.reshape(y_pred[:, :84*84], (-1, 84, 84))
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_pred_norm = (y_pred - tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)) / (tf.math.reduce_std(y_pred, axis=[1, 2], keepdims=True) + K.backend.epsilon())
    return tf.reduce_mean(tf.reduce_sum(y_pred_norm * y_true, axis=[1, 2]) / (tf.reduce_sum(y_true, axis=[1, 2]) + K.backend.epsilon()))

def auc_metric(y_true, y_pred):
    # Reshape the inputs to (batch_size, 84*84)
    y_true = tf.reshape(y_true[:, :84*84], (-1, 84*84))
    y_pred = tf.reshape(y_pred[:, :84*84], (-1, 84*84))
    
    def auc_tf(y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        
        # Sort predictions and corresponding true values
        sorted_indices = tf.argsort(y_pred, direction='DESCENDING')
        y_true_sorted = tf.gather(y_true, sorted_indices)
        
        # Calculate TPR and FPR
        tp_cumsum = tf.cumsum(y_true_sorted)
        fp_cumsum = tf.cumsum(1 - y_true_sorted)
        
        tp_rate = tp_cumsum / (tf.reduce_sum(y_true) + K.backend.epsilon())
        fp_rate = fp_cumsum / (tf.reduce_sum(1 - y_true) + K.backend.epsilon())
        
        # Calculate AUC using trapezoidal rule
        auc = tf.reduce_sum((fp_rate[1:] - fp_rate[:-1]) * (tp_rate[1:] + tp_rate[:-1]) / 2)
        
        return auc
    
    return tf.reduce_mean(tf.map_fn(lambda x: auc_tf(x[0], x[1]), (y_true, y_pred), fn_output_signature=tf.float32))

def cc_metric(y_true, y_pred):
    # Reshape the inputs to (batch_size, 84, 84)
    y_true = tf.reshape(y_true[:, :84*84], (-1, 84, 84))
    y_pred = tf.reshape(y_pred[:, :84*84], (-1, 84, 84))
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    y_true_mean = tf.reduce_mean(y_true, axis=[1, 2], keepdims=True)
    y_pred_mean = tf.reduce_mean(y_pred, axis=[1, 2], keepdims=True)
    
    y_true_std = tf.math.reduce_std(y_true, axis=[1, 2], keepdims=True)
    y_pred_std = tf.math.reduce_std(y_pred, axis=[1, 2], keepdims=True)
    
    covariance = tf.reduce_mean((y_true - y_true_mean) * (y_pred - y_pred_mean), axis=[1, 2])
    
    cc = covariance / (y_true_std * y_pred_std + K.backend.epsilon())
    return tf.reduce_mean(cc)

def my_softmax(x):
    reshaped_x = tf.reshape(x, (-1, 84 * 84))
    softmaxed_x = tf.nn.softmax(reshaped_x, axis=-1)
    output = tf.reshape(softmaxed_x, tf.shape(x))
    return output

def my_kld(y_true, y_pred):
    epsilon = 1e-10
    y_true = K.backend.cast(K.backend.clip(y_true, epsilon, 1), tf.float32)
    y_pred = K.backend.cast(K.backend.clip(y_pred, epsilon, 1), tf.float32)
    return K.backend.sum(y_true * K.backend.log(y_true / y_pred), axis=[1, 2, 3])


def duration_loss(y_true, y_pred):
    loss = FocalLoss(alpha=alpha, gamma=gamma)
    return loss(y_true, y_pred)

def combined_loss(gaze_weight=gaze_weight, duration_weight= 1 - gaze_weight):
    def loss(y_true, y_pred):
        gaze_true = y_true[:, :84*84]
        duration_true = y_true[:, 84*84:]
        
        gaze_pred = y_pred[:, :84*84]
        duration_pred = y_pred[:, 84*84:]

        gaze_true = tf.reshape(gaze_true, [-1, 84, 84, 1])
        gaze_pred = tf.reshape(gaze_pred, [-1, 84, 84, 1])

        gaze_loss = my_kld(gaze_true, gaze_pred)
        dur_loss = duration_loss(duration_true, duration_pred)
        
        return gaze_weight * gaze_loss + duration_weight * dur_loss
    return loss


def iou_metric(y_true, y_pred):
    # Extract only the gaze map portion (first 84*84 elements)
    gaze_true = y_true[:, :84*84]
    gaze_pred = y_pred[:, :84*84]
    
    # Reshape back to image dimensions
    gaze_true = tf.reshape(gaze_true, [-1, 84, 84, 1])
    gaze_pred = tf.reshape(gaze_pred, [-1, 84, 84, 1])
    
    # For y_true, keep the binary nature
    y_true_mask = tf.cast(gaze_true > 0.3, tf.float32)
    
    # For y_pred, use a dynamic threshold
    threshold = tf.reduce_mean(gaze_pred) + tf.math.reduce_std(gaze_pred)
    y_pred_mask = tf.cast(gaze_pred > threshold, tf.float32)
    
    # Calculate intersection and union
    intersection = tf.reduce_sum(y_true_mask * y_pred_mask, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true_mask, axis=[1, 2, 3]) + tf.reduce_sum(y_pred_mask, axis=[1, 2, 3]) - intersection
    
    # Calculate IOU
    iou = tf.where(union > 0, intersection / union, tf.zeros_like(intersection))
    
    return tf.reduce_mean(iou)

def create_saliency_model(input_shape=(84, 84, 4), regularization_factor=0.01, dropout=0.3):
    imgs = L.Input(shape=input_shape)
    opt = L.Input(shape=input_shape)
    sal = L.Input(shape=input_shape)
    
    def process_branch(input_tensor, name_prefix):
        x = input_tensor
        for i, (filters, kernel_size, strides) in enumerate([(32, 8, 4), (64, 4, 2), (64, 3, 1)]):
            x = L.Conv2D(filters, kernel_size, strides=strides, padding='valid', 
                         kernel_regularizer=regularizers.l2(regularization_factor),
                         name=f'{name_prefix}_conv{i+1}')(x)
            x = L.Activation('relu')(x)
            x = L.BatchNormalization()(x)
            x = L.Dropout(dropout)(x)
        
        for i, (filters, kernel_size, strides) in enumerate([(64, 3, 1), (32, 4, 2), (1, 8, 4)]):
            x = L.Conv2DTranspose(filters, kernel_size, strides=strides, padding='valid',
                                  kernel_regularizer=regularizers.l2(regularization_factor),
                                  name=f'{name_prefix}_deconv{i+1}')(x)
            if i < 2:
                x = L.Activation('relu')(x)
                x = L.BatchNormalization()(x)
                x = L.Dropout(dropout)(x)
        return x
    
    x = process_branch(imgs, 'imgs')
    opt_x = process_branch(opt, 'opt')
    sal_x = process_branch(sal, 'sal')
    
    # Gaze map prediction
    gaze_map = L.Average()([x, opt_x, sal_x])
    gaze_map = L.Activation(my_softmax)(gaze_map)
    gaze_map_flat = L.Flatten()(gaze_map)
    
    # Duration prediction branch
    duration_x = L.Concatenate()([L.Flatten()(x), L.Flatten()(opt_x), L.Flatten()(sal_x)])
    duration_x = L.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
    duration_x = L.Dropout(dropout)(duration_x)
    duration_x = L.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
    duration_x = L.Dropout(dropout)(duration_x)
    duration_pred = L.Dense(2, activation='softmax', kernel_regularizer=regularizers.l2(regularization_factor))(duration_x)
    
    # Combine outputs
    combined_output = L.Concatenate()([gaze_map_flat, duration_pred])
    
    model = Model(inputs=[imgs, opt, sal], outputs=combined_output)
    return model

class TrainingLogger(K.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        kld = logs.get('loss')
        val_kld = logs.get('val_loss')
        iou = logs.get('iou_metric')
        val_iou = logs.get('val_iou_metric')
        nss = logs.get('nss_metric')
        val_nss = logs.get('val_nss_metric')
        auc = logs.get('auc_metric')
        val_auc = logs.get('val_auc_metric')
        cc = logs.get('cc_metric')
        val_cc = logs.get('val_cc_metric')
        duration_loss_value = logs.get('duration_loss')
        val_duration_loss_value = logs.get('val_duration_loss')

        if val_kld is not None:
            wandb.log({
                "val_kld": val_kld,
                "train_kld": kld,
                "train_iou": iou,
                "val_iou": val_iou,
                "train_nss": nss,
                "val_nss": val_nss,
                "train_auc": auc,
                "val_auc": val_auc,
                "train_cc": cc,
                "val_cc": val_cc,
                "train_duration_loss": duration_loss_value,
                "val_duration_loss": val_duration_loss_value,
            })

        print(f"Epoch {epoch + 1}: KLD = {kld:.4f}, val_KLD = {val_kld:.4f}, IOU = {iou:.4f}, val_IOU = {val_iou:.4f}, "
              f"NSS = {nss:.4f}, val_NSS = {val_nss:.4f}, AUC = {auc:.4f}, val_AUC = {val_auc:.4f}, "
              f"CC = {cc:.4f}, val_CC = {val_cc:.4f}, Duration Loss = {duration_loss_value:.4f}, "
              f"val_Duration Loss = {val_duration_loss_value:.4f}")

        
early_stopping = EarlyStopping(
        monitor='val_duration_loss',
        patience=20,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

def calculate_and_print_confusion_matrix(model, test_dataset, test_steps):
    # Get predictions
    y_pred = model.predict(test_dataset, steps=test_steps)
    
    # Extract duration predictions (last 2 elements of each prediction)
    y_pred_duration = y_pred[:, -2:]
    y_pred_classes = np.argmax(y_pred_duration, axis=1)
    
    # Get true labels
    y_true = np.concatenate([y for x, y in test_dataset.take(test_steps)])
    y_true_duration = y_true[:, -2:]
    y_true_classes = np.argmax(y_true_duration, axis=1)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    
    # Print confusion matrix
    print("Confusion Matrix:")
    print(cm)
    
    # Calculate and print additional metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    
    # Log metrics to wandb
    wandb.log({
        # "confusion_matrix": wandb.Table(data=cm.tolist(), columns=["Predicted Negative", "Predicted Positive"]),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    })        

        
def main():
    
    # train_tar_file = './berzerk/496_RZ_3560871_Jul-19-13-28-35.tar.bz2'
    # train_opt_file = './berzerk/496_RZ_3560871_Jul-19-13-28-35_opt.tar.bz2'
    # train_sal_file = './berzerk/496_RZ_3560871_Jul-19-13-28-35_sal.tar.bz2'
    # train_label_file = './berzerk/496_RZ_3560871_Jul-19-13-28-35.txt'
    

    # train_tar_file = './berzerk/test.tar.bz2'
    # train_opt_file = './berzerk/test_opt.tar.bz2'
    # train_sal_file = './berzerk/test_sal.tar.bz2'
    # train_label_file = './berzerk/test.txt'
    
    # val_tar_file = './berzerk/val_test.tar.bz2'
    # val_opt_file = './berzerk/val_test_opt.tar.bz2'
    # val_sal_file = './berzerk/val_test_sal.tar.bz2'
    # val_label_file = './berzerk/val_test.txt'
    
    # test_tar_file = './berzerk/val_test.tar.bz2'
    # test_opt_file = './berzerk/val_test_opt.tar.bz2'
    # test_sal_file = './berzerk/val_test_sal.tar.bz2'
    # test_label_file = './berzerk/val_test.txt'
    
        
    
    # train_tar_file = './berzerk/train_data/train.tar.bz2'
    # train_opt_file = './berzerk/train_data/train_opt.tar.bz2'
    # train_sal_file = './berzerk/train_data/train_sal.tar.bz2'
    # train_label_file = './berzerk/train_data/train.txt'
    
    # # train_tar_file = './berzerk/val_data/val.tar.bz2'
    # # train_opt_file = './berzerk/val_data/val_opt.tar.bz2'
    # # train_sal_file = './berzerk/val_data/val_sal.tar.bz2'
    # # train_label_file = './berzerk/val_data/val.txt'
    
    # val_tar_file = './berzerk/val_data/val.tar.bz2'
    # val_opt_file = './berzerk/val_data/val_opt.tar.bz2'
    # val_sal_file = './berzerk/val_data/val_sal.tar.bz2'
    # val_label_file = './berzerk/val_data/val.txt'
    
    
    # test_tar_file = './berzerk/test_data/test.tar.bz2'
    # test_opt_file = './berzerk/test_data/test_opt.tar.bz2'
    # test_sal_file = './berzerk/test_data/test_sal.tar.bz2'
    # test_label_file = './berzerk/test_data/test.txt'
    
    train_tar_file = '/datasets/lvmayer/berzerk/train/train.tar.bz2'
    train_opt_file = '/datasets/lvmayer/berzerk/train/train_opt.tar.bz2'
    train_sal_file = '/datasets/lvmayer/berzerk/train/train_sal.tar.bz2'
    train_label_file = '/datasets/lvmayer/berzerk/train/train.txt'
    
    val_tar_file = '/datasets/lvmayer/berzerk/val/val.tar.bz2'
    val_opt_file = '/datasets/lvmayer/berzerk/val/val_opt.tar.bz2'
    val_sal_file = '/datasets/lvmayer/berzerk/val/val_sal.tar.bz2'
    val_label_file = '/datasets/lvmayer/berzerk/val/val.txt'
    
    
    test_tar_file = '/datasets/lvmayer/berzerk/test/test.tar.bz2'
    test_opt_file = '/datasets/lvmayer/berzerk/test/test_opt.tar.bz2'
    test_sal_file = '/datasets/lvmayer/berzerk/test/test_sal.tar.bz2'
    test_label_file = '/datasets/lvmayer/berzerk/test/test.txt'
    
    
    
    
    
    t1 = time.time()
    
    # train_dataset = Dataset(train_tar_file, train_opt_file, train_sal_file, train_label_file)
    train_dataset = TemporalSMOTEDataset(train_tar_file, train_opt_file, train_sal_file, train_label_file, minority_increase=minority_increase)
    # # Visualize the first synthetic sample
    # train_dataset.visualize_synthetic_sample()

    # # Visualize the 6th synthetic sample (if available)
    # train_dataset.visualize_synthetic_sample(5)
    
    
    val_dataset = Dataset(val_tar_file, val_opt_file, val_sal_file, val_label_file)
    
    
    

    train_tf_dataset = train_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    val_tf_dataset = val_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    
     # Calculate steps per epoch
    train_steps_per_epoch = train_dataset.get_steps_per_epoch(BATCH_SIZE)
    val_steps = val_dataset.get_steps_per_epoch(BATCH_SIZE)
    
    print(f"Time spent loading and preprocessing: {time.time() - t1:.1f}s")

    # Create and compile the model
    model = create_saliency_model()
    
    opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
    model.compile(loss=combined_loss(gaze_weight), optimizer=opt, 
                  metrics=['accuracy', iou_metric, nss_metric, auc_metric, cc_metric, duration_loss])
    print("Compiled")
    
    
   # Initialize callbacks
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
    model.save(f"/datasets/lvmayer/berzerk/{run_name}.hdf5")
    
    # Load test data
    test_dataset = Dataset(test_tar_file, test_opt_file, test_sal_file, test_label_file)
    test_tf_dataset = test_dataset.get_dataset(BATCH_SIZE, BUFF_SIZE)
    test_steps = test_dataset.get_steps_per_epoch(BATCH_SIZE)

    test_kld, test_acc, test_iou, test_nss, test_auc, test_cc, test_duration_loss = model.evaluate(test_tf_dataset, steps=test_steps, verbose=2)
    print(f'Test KLD: {test_kld}, Test IOU: {test_iou}, Test NSS: {test_nss}, Test AUC: {test_auc}, '
          f'Test CC: {test_cc}, Test Duration Loss: {test_duration_loss}')
    wandb.log({
        "test_kld": test_kld,
        "test_iou": test_iou,
        "test_nss": test_nss,
        "test_auc": test_auc,
        "test_cc": test_cc,
        "test_duration_loss": test_duration_loss
    })

    calculate_and_print_confusion_matrix(model, test_tf_dataset, test_steps)


    # Finish the WandB run
    wandb.finish()

   

  


if __name__ == "__main__":
   
    main()


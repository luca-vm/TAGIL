import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
from keras_cv.losses import FocalLoss
# from load_onlyinit import *
from load_t_inf import *
import time
from sklearn.metrics import confusion_matrix
import gc

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 1000
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.0
epsilon=1e-08
gaze_weight = 0.3
alpha = 0.25
gamma = 2.0
minority_increase = 40

# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

# Initialize WandB run
run = wandb.init(
    project="InfSmote",
    config={

    }
)




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


class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name 

    def init_model(self, gaze_model_file, input_shape=(84, 84, 4), regularization_factor=regularization_factor, dropout= dropout):
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
        
        self.model = Model(inputs=[imgs, opt, sal], outputs=combined_output)
        
        opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
        self.model.compile(loss=combined_loss(gaze_weight), optimizer=opt, metrics=['accuracy'])
        
      
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
        


  
    def predict_and_save(self, dataset, batch_size):
        print("Predicting results and saving incrementally...")
        
        npz_file = f"/datasets/lvmayer/{self.game_name}/gaze_smote_test.npz"
        temp_file = f"/datasets/lvmayer/{self.game_name}/temp_gaze_smote_test.npz"
        
        batch_count = 0
        total_predictions = 0
        
        for (stacked_imgs, stacked_opts, stacked_sals), _, frame_ids in dataset:
            # Process data in smaller sub-batches
            for i in range(0, len(stacked_imgs), batch_size):
                sub_batch = [
                    stacked_imgs[i:i+batch_size],
                    stacked_opts[i:i+batch_size],
                    stacked_sals[i:i+batch_size]
                ]
                
                batch_preds = self.model.predict(sub_batch, batch_size=batch_size)
                frame_ids_batch = frame_ids[i:i+batch_size]
                frame_ids_batch = [frame_id.numpy().decode('utf-8') for frame_id in frame_ids_batch]
                
                # Separate gaze heatmap and duration predictions
                gaze_preds = batch_preds[:, :84*84]
                duration_preds = batch_preds[:, 84*84:]
                
                # Reshape gaze predictions back to 2D heatmaps
                gaze_heatmaps = gaze_preds.reshape(-1, 84, 84)
                
                # Save predictions to temporary file
                np.savez_compressed(temp_file, 
                                    heatmap=gaze_heatmaps, 
                                    durations=duration_preds,
                                    frame_ids=frame_ids_batch)
                
                # Append temporary file to main file
                self._append_to_npz(temp_file, npz_file)
                
                total_predictions += len(batch_preds)
                batch_count += 1
                print(f"Processed and saved sub-batch {batch_count}, Total predictions: {total_predictions}")
                
                # Clear memory
                del batch_preds, gaze_preds, duration_preds, gaze_heatmaps, frame_ids_batch
                gc.collect()
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print("Prediction and incremental saving completed.")
        print(f"Output is saved in: {npz_file}")

    def _append_to_npz(self, source_file, target_file):
        with np.load(source_file, allow_pickle=True) as source_data:
            source_heatmap = source_data['heatmap']
            source_durations = source_data['durations']
            source_frame_ids = source_data['frame_ids']
        
        if not os.path.exists(target_file):
            np.savez_compressed(target_file, heatmap=source_heatmap, durations=source_durations, frame_ids=source_frame_ids)
        else:
            with np.load(target_file, allow_pickle=True) as target_data:
                target_heatmap = target_data['heatmap']
                target_durations = target_data['durations']
                target_frame_ids = target_data['frame_ids']
            
            # Ensure correct shapes
            if target_heatmap.size == 0:
                target_heatmap = target_heatmap.reshape(0, 84, 84)
            if target_durations.size == 0:
                target_durations = target_durations.reshape(0, 2)
            elif len(target_durations.shape) == 1:
                target_durations = target_durations.reshape(-1, 2)
            
            updated_heatmap = np.concatenate([target_heatmap, source_heatmap], axis=0)
            updated_durations = np.concatenate([target_durations, source_durations], axis=0)
            updated_frame_ids = np.concatenate([target_frame_ids, source_frame_ids])
            
            np.savez_compressed(target_file, heatmap=updated_heatmap, durations=updated_durations, frame_ids=updated_frame_ids)
        
        # Clear memory
        del source_heatmap, source_durations, source_frame_ids
        if 'target_heatmap' in locals():
            del target_heatmap, target_durations, target_frame_ids, updated_heatmap, updated_durations, updated_frame_ids
        gc.collect()

# test_tar_file = '/datasets/lvmayer/berzerk/train/train.tar.bz2'
# test_opt_file = '/datasets/lvmayer/berzerk/train/train_opt.tar.bz2'
# test_sal_file = '/datasets/lvmayer/berzerk/train/train_sal.tar.bz2'
# test_label_file = '/datasets/lvmayer/berzerk/train/train.txt'

# test_tar_file = '/datasets/lvmayer/berzerk/val/val.tar.bz2'
# test_opt_file = '/datasets/lvmayer/berzerk/val/val_opt.tar.bz2'
# test_sal_file = '/datasets/lvmayer/berzerk/val/val_sal.tar.bz2'
# test_label_file = '/datasets/lvmayer/berzerk/val/val.txt'

test_tar_file = '/datasets/lvmayer/berzerk/test/test.tar.bz2'
test_opt_file = '/datasets/lvmayer/berzerk/test/test_opt.tar.bz2'
test_sal_file = '/datasets/lvmayer/berzerk/test/test_sal.tar.bz2'
test_label_file = '/datasets/lvmayer/berzerk/test/test.txt'




name = 'berzerk'
# gaze_model = '/datasets/lvmayer/berzerk/ethereal-haze-16.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/gallant-sponge-21.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/unique-armadillo-27.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/bright-dream-47.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/scarlet-elevator-58.hdf5'
gaze_model = '/datasets/lvmayer/berzerk/electric-deluge-66.hdf5'
# file_name = (tar_file.split('.'))[1].split('/')[2])


# file_name = (tar_file.split('.'))[1].split('/')[2])

       

def run_prediction(test_dataset, gaze_model, name, batch_size):
    gp = Human_Gaze_Predictor(name)
    gp.init_model(gaze_model)

    test_tf_dataset = test_dataset.get_dataset(batch_size)

    gp.predict_and_save(test_tf_dataset, batch_size)

# Usage
batch_size = 20000# You can adjust this value based on your GPU memory
test_dataset = Dataset(test_tar_file, test_opt_file, test_sal_file, test_label_file)
run_prediction(test_dataset, gaze_model, name, batch_size)
import tensorflow as tf
import tensorflow.keras as K
import tensorflow.keras.layers as L
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
# from load_onlyinit import *
from load_inf import *
import gc

BATCH_SIZE = 128
BUFF_SIZE = 40000
num_epoch = 3
lr = 1.0
r = 0.95
dropout = 0.3
regularization_factor=0.01
epsilon=1e-08

# import os
# os.environ["WANDB_MODE"] = "disabled"


import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# WandB initialization
wandb.login(key='ed8dd6f4ee07699f2e9c1d9a3ffec3b84b45c4b6')

# Initialize WandB run
run = wandb.init(
    project="Inf",
    config={

    }
)



def my_softmax(x):
    # Reshape the input tensor to flatten the spatial dimensions (84x84)
    reshaped_x = tf.reshape(x, (-1, 84 * 84))
    
    # Apply the softmax along the last dimension
    softmaxed_x = tf.nn.softmax(reshaped_x, axis=-1)
    
    # Reshape it back to the original shape (None, 84, 84, 1)
    output = tf.reshape(softmaxed_x, tf.shape(x))
    
    return output


def my_kld(y_true, y_pred):
    """Compute the KL-divergence between two metrics."""
    epsilon = 1e-10
    y_true = tf.clip_by_value(y_true, epsilon, 1.0)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0)
    
    # KL divergence calculation based on the provided formula
    kld = tf.reduce_sum(y_true * tf.math.log(epsilon + y_true / (epsilon + y_pred)), axis=[1, 2, 3])
    
    return kld


class Human_Gaze_Predictor:
    def __init__(self, game_name):
        self.game_name = game_name

    def init_model(self, gaze_model_file, input_shape=(84, 84, 4), regularization_factor=regularization_factor, dropout= dropout):
        imgs = L.Input(shape=input_shape)
    
    
        x=imgs 
        conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = conv3(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)
        
        deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv1(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)

        deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv2(x)
        x=L.Activation('relu')(x)
        x=L.BatchNormalization()(x)
        x=L.Dropout(dropout)(x)         

        deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        x = deconv3(x)
        
        #==================== Branch 2 ============================
        opt = L.Input(shape=input_shape)
        
        opt_x=opt 
        opt_conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv1(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv2(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_conv3(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)
        
        opt_deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv1(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)

        opt_deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv2(opt_x)
        opt_x=L.Activation('relu')(opt_x)
        opt_x=L.BatchNormalization()(opt_x)
        opt_x=L.Dropout(dropout)(opt_x)         

        opt_deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        opt_x = opt_deconv3(opt_x)
        
        
        
        #==================== Branch 3 ============================
        sal = L.Input(shape=input_shape)
        
        sal_x=sal 
        sal_conv1=L.Conv2D(32, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv1(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_conv2=L.Conv2D(64, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv2(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_conv3=L.Conv2D(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_conv3(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)
        
        sal_deconv1 = L.Conv2DTranspose(64, (3,3), strides=1, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv1(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)

        sal_deconv2 = L.Conv2DTranspose(32, (4,4), strides=2, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv2(sal_x)
        sal_x=L.Activation('relu')(sal_x)
        sal_x=L.BatchNormalization()(sal_x)
        sal_x=L.Dropout(dropout)(sal_x)         

        sal_deconv3 = L.Conv2DTranspose(1, (8,8), strides=4, padding='valid', kernel_regularizer=regularizers.l2(regularization_factor))
        sal_x = sal_deconv3(sal_x)
        

        #=================== Avg ==================================
        x = L.Average()([x, opt_x, sal_x])
        outputs = L.Activation(my_softmax)(x)
        self.model=Model(inputs=[imgs, opt, sal], outputs=outputs)
        
        opt = K.optimizers.Adadelta(learning_rate=lr, rho=r, epsilon=epsilon)
        self.model.compile(loss=my_kld, optimizer=opt, metrics=['accuracy'])
        
      
        print("Loading model weights from %s" % gaze_model_file)
        self.model.load_weights(gaze_model_file)
        print("Loaded.")
  
    def predict_and_save(self, dataset, batch_size):
        print("Predicting results and saving incrementally...")
        
        npz_file = f"/datasets/lvmayer/{self.game_name}/gaze_train.npz"
        temp_file = f"/datasets/lvmayer/{self.game_name}/temp_gaze_train.npz"
        
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
                
                # Save predictions to temporary file
                np.savez_compressed(temp_file, 
                                    heatmap=batch_preds[:,:,:,0], 
                                    frame_ids=frame_ids_batch)
                
                # Append temporary file to main file
                self._append_to_npz(temp_file, npz_file)
                
                total_predictions += len(batch_preds)
                batch_count += 1
                print(f"Processed and saved sub-batch {batch_count}, Total predictions: {total_predictions}")
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        print("Prediction and incremental saving completed.")
        print(f"Output is saved in: {npz_file}")

    def _append_to_npz(self, source_file, target_file):
        with np.load(source_file, allow_pickle=True) as source_data:
            source_heatmap = source_data['heatmap']
            source_frame_ids = source_data['frame_ids']
        
        if not os.path.exists(target_file):
            np.savez_compressed(target_file, heatmap=source_heatmap, frame_ids=source_frame_ids)
        else:
            with np.load(target_file, allow_pickle=True) as target_data:
                target_heatmap = target_data['heatmap']
                target_frame_ids = target_data['frame_ids']
            
            updated_heatmap = np.concatenate([target_heatmap, source_heatmap], axis=0)
            updated_frame_ids = np.concatenate([target_frame_ids, source_frame_ids])
            
            np.savez_compressed(target_file, heatmap=updated_heatmap, frame_ids=updated_frame_ids)
            
         # Clear memory
        del source_heatmap, source_frame_ids
        if 'target_heatmap' in locals():
            del target_heatmap, target_frame_ids, updated_heatmap, updated_frame_ids
        gc.collect()


test_tar_file = '/datasets/lvmayer/berzerk/train/train.tar.bz2'
test_opt_file = '/datasets/lvmayer/berzerk/train/train_opt.tar.bz2'
test_sal_file = '/datasets/lvmayer/berzerk/train/train_sal.tar.bz2'
test_label_file = '/datasets/lvmayer/berzerk/train/train.txt'

# test_tar_file = '/datasets/lvmayer/berzerk/val/val.tar.bz2'
# test_opt_file = '/datasets/lvmayer/berzerk/val/val_opt.tar.bz2'
# test_sal_file = '/datasets/lvmayer/berzerk/val/val_sal.tar.bz2'
# test_label_file = '/datasets/lvmayer/berzerk/val/val.txt'

# test_tar_file = '/datasets/lvmayer/berzerk/test/test.tar.bz2'
# test_opt_file = '/datasets/lvmayer/berzerk/test/test_opt.tar.bz2'
# test_sal_file = '/datasets/lvmayer/berzerk/test/test_sal.tar.bz2'
# test_label_file = '/datasets/lvmayer/berzerk/test/test.txt'




name = 'berzerk'
# gaze_model = '/datasets/lvmayer/berzerk/avid-grass-10.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/polished-plasma-17.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/mild-sky-30.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/sunny-dream-41.hdf5'
# gaze_model = '/datasets/lvmayer/berzerk/firm-dew-55.hdf5'
gaze_model = '/datasets/lvmayer/berzerk/faithful-smoke-62.hdf5'
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
wandb.finish()
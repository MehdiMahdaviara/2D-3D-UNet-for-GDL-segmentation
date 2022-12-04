# Import libraries
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Input, concatenate, Conv2DTranspose
from keras.models import Model
import segmentation_models_3D as sm

# Main
def UNet(input_shape):
    inputs = Input(input_shape)
    # Encoding
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)

    #Decoding
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = Conv2D(3, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    return model

# Compiling
height = 384
width = 384
n_channels = 3

input_shape=(width, height, n_channels)
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.33, 0.33, 0.34]))
focal_loss = sm.losses.CategoricalFocalLoss()
dice_plus_focal_loss = dice_loss + (1*focal_loss)

model_unet = UNet(input_shape)
model_unet.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-4),
                   loss= dice_plus_focal_loss, #tf.keras.losses.CategoricalCrossentropy(),
                   metrics=['acc', sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)],
                   #run_eagerly = True
                   )
model_unet.summary()

# Training
callbacks = [
    tf.keras.callbacks.CSVLogger('train_log.csv', separator=",", append=False),
]

model_unet.fit(x_train, y_train,
               validation_data=(x_test, y_test), 
               epochs= 20,
               batch_size= 1,
               callbacks = callbacks)


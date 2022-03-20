import tensorflow as tf
from tensorflow.keras.layers import *

init = tf.keras.initializers.RandomNormal(stddev=0.02)

def Discriminator(image_shape):
    image = Input(shape=image_shape)
    target = Input(shape=image_shape)
    concat = Concatenate()([image, target])
    # First
    x = Conv2D(64, kernel_size=(4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(concat)
    x = LeakyReLU(alpha=0.2)(x)
    
    # Second
    x = Conv2D(128, kernel_size=(4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Third
    x = Conv2D(256, kernel_size=(4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)

    # Fourth
    x = Conv2D(512, kernel_size=(4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(1, kernel_size=(4,4), padding='same', kernel_initializer=init)(x)
    x = Activation('sigmoid')(x)

    model = tf.keras.models.Model([image, target], x)
    return model


def encoder(inputs, filters, batch_norm=True):
    x = Conv2D(filters, (4,4), strides=(2,2), padding='same',
                        kernel_initializer=init)(inputs)
    if batch_norm:
        x = BatchNormalization()(x, training=True)
    x = LeakyReLU(alpha=0.2)(x)
    return x

def decoder(inputs, skip, filters, dropout=True):
    x = Conv2DTranspose(filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(inputs)
    x = BatchNormalization()(x, training=True)
    if dropout:
        x = Dropout(0.5)(x, training=True)
    x = Concatenate()([x, skip])
    x = Activation('relu')(x)
    return x

def generator(input_shape):
    inputs = Input(shape=input_shape)

    x1 = encoder(inputs, 64, batch_norm=False)
    x2 = encoder(x1, 128)
    x3 = encoder(x2, 256)
    x4 = encoder(x3, 512)
    x5 = encoder(x4, 512)
    x6 = encoder(x5, 512)
    x7 = encoder(x6, 512)

    x = Conv2D(512, (4,4),strides=(2,2), padding='same', kernel_initializer=init)(x7)
    x = ReLU()(x)

    y1 = decoder(x, x7, 512)
    y2 = decoder(y1, x6, 512)
    y3 = decoder(y2, x5, 512)
    y4 = decoder(y3, x4, 512, dropout=False)
    y5 = decoder(y4, x3, 256, dropout=False)
    y6 = decoder(y5, x2, 128, dropout=False)
    y7 = decoder(y6, x1, 64, dropout=False)

    x_new = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same',
                    kernel_initializer=init)(y7)
    x_new = tf.keras.activations.tanh(x_new)

    model = tf.keras.models.Model(inputs, x_new)
    return model
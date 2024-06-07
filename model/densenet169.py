import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, Dense, GlobalAveragePooling2D

def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def conv_block(x, growth_rate):
    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, (1, 1), use_bias=False)(x1)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(growth_rate, (3, 3), padding='same', use_bias=False)(x1)
    x = layers.Concatenate()([x, x1])
    return x

def transition_block(x, reduction):
    filters = int(tf.keras.backend.int_shape(x)[-1] * reduction)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (1, 1), use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x

def DenseNet169(input_shape=(32, 32, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(inputs)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    
    x = dense_block(x, 6, 32)  # 6 layers
    x = transition_block(x, 0.5)
    x = dense_block(x, 12, 32)  # 12 layers
    x = transition_block(x, 0.5)
    x = dense_block(x, 32, 32)  # 32 layers
    x = transition_block(x, 0.5)
    x = dense_block(x, 32, 32)  # 32 layers
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs=x)
    return model
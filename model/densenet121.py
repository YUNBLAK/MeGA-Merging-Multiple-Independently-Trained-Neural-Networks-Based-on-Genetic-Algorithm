import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Concatenate, AveragePooling2D, Dense, GlobalAveragePooling2D

def dense_block(x, blocks, growth_rate):
    for i in range(blocks):
        x = conv_block(x, growth_rate)
    return x

def conv_block(x, growth_rate):
    # BatchNorm-ReLU-Conv2D pattern
    x1 = BatchNormalization()(x)
    x1 = Activation('relu')(x1)
    x1 = Conv2D(growth_rate, (3, 3), padding='same')(x1)
    
    # Concatenate input and new feature map
    x = Concatenate(axis=-1)([x, x1])
    return x

def transition_block(x, reduction):
    filters = int(tf.keras.backend.int_shape(x)[-1] * reduction)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, (1, 1), padding='same')(x)
    x = AveragePooling2D((2, 2), strides=2)(x)
    return x

def DenseNet121(input_shape=(32, 32, 3), num_classes=10, blocks=[6, 12, 24, 16], growth_rate=32):
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    
    # Dense Blocks
    for i, block in enumerate(blocks):
        x = dense_block(x, block, growth_rate)
        if i != len(blocks) - 1:
            x = transition_block(x, 0.5)
    
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    return model
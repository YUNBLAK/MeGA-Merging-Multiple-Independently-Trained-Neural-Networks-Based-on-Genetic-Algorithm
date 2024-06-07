import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, MaxPooling2D, Flatten, DepthwiseConv2D, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model

def vgg_block(x, filters, layers):
    global globalact
    for _ in range(layers):
        x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    return x

# Building VGG16
def VGG16(input_shape, num_classes=10):
    global globalact
    inputs = Input(shape=input_shape)
    
    # VGG16 Architecture
    x = vgg_block(inputs, 64, 2)
    x = BatchNormalization()(x)
    x = vgg_block(x, 128, 2)
    x = BatchNormalization()(x)
    x = vgg_block(x, 256, 3)
    x = BatchNormalization()(x)
    x = vgg_block(x, 512, 3)
    x = BatchNormalization()(x)
    x = vgg_block(x, 512, 3)
    x = BatchNormalization()(x)
    
    x = Flatten()(x)
    
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)
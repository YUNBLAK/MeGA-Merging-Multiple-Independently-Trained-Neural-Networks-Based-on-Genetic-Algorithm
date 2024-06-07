import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Dense, MaxPooling2D, Flatten, DepthwiseConv2D, Concatenate, AveragePooling2D
from tensorflow.keras.models import Model

def mobilenetv2_block(x, filters, strides):
    # Depthwise Convolution
    x = DepthwiseConv2D(3, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Pointwise Convolution
    x = Conv2D(filters, 1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    return x

def MobileNetV2(input_shape=(32, 32, 3), num_classes=10):
    input = Input(shape=input_shape)

    # Initial Conv Layer
    x = Conv2D(32, 3, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # MobileNetV2 Blocks
    x = mobilenetv2_block(x, 64, 1)
    x = mobilenetv2_block(x, 128, 2)
    x = mobilenetv2_block(x, 128, 1)
    x = mobilenetv2_block(x, 256, 2)
    x = mobilenetv2_block(x, 256, 1)
    x = mobilenetv2_block(x, 512, 2)
    for _ in range(5):
        x = mobilenetv2_block(x, 512, 1)
    x = mobilenetv2_block(x, 1024, 2)
    x = mobilenetv2_block(x, 1024, 1)
    
    x = GlobalAveragePooling2D()(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input, outputs=output)
    return model

def mobilenet_v1_block(input, filters, strides):
    """A basic MobileNet V1 block consists of a depthwise conv, batch norm, ReLU, followed by a pointwise conv, batch norm, and ReLU."""
    # Depthwise convolution
    x = DepthwiseConv2D(kernel_size=3, strides=strides, padding='same')(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Pointwise convolution
    x = Conv2D(filters, kernel_size=1, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    return x

def MobileNetV1(input_shape=(32, 32, 3), num_classes=10):
    input = Input(shape=input_shape)

    x = Conv2D(32, kernel_size=3, padding='same', use_bias=False)(input)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Define the number of filters
    filters_list = [64, 128, 128, 256, 256, 512] + [512]*5 + [1024, 1024]

    # Define the strides of the first layer of each block
    strides_list = [1, 2, 1, 2, 1, 2] + [1]*5 + [2, 1]

    for filters, strides in zip(filters_list, strides_list):
        x = mobilenet_v1_block(x, filters, strides)

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer for classification
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=output)
    return model
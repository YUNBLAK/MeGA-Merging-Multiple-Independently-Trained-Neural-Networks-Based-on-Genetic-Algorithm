import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, MaxPooling2D
from tensorflow.keras.models import Model

def res_block(x, filters, kernel_size=3, stride=1, use_relu=True):
    global globalact
    shortcut = x

    # First convolution layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    if use_relu:
        x = ReLU()(x)

    # Second convolution layer
    x = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    # Adjusting the shortcut for addition
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = tf.keras.layers.add([x, shortcut])

    if use_relu:
        x = ReLU()(x)

    return x

# Building ResNet-56
def ResNet56(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for _ in range(9):
        x = res_block(x, 16)

    x = res_block(x, 32, stride=2)
    for _ in range(8):
        x = res_block(x, 32)

    x = res_block(x, 64, stride=2)
    for _ in range(8):
        x = res_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

def ResNet110(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for _ in range(18):  # 18 blocks with 16 filters
        x = res_block(x, 16)

    x = res_block(x, 32, stride=2)
    for _ in range(17):  # 18 blocks with 32 filters, considering the first one with stride=2
        x = res_block(x, 32)

    x = res_block(x, 64, stride=2)
    for _ in range(17):  # 18 blocks with 64 filters, considering the first one with stride=2
        x = res_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)

def ResNet152(input_shape, num_classes=10):
    inputs = Input(shape=input_shape)
    
    x = Conv2D(16, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    # Residual blocks
    for _ in range(18):  # Adjusting the number of blocks for ResNet-156
        x = res_block(x, 16)

    x = res_block(x, 32, stride=2)
    for _ in range(36):  # Adjusting the number of blocks for ResNet-156
        x = res_block(x, 32)

    x = res_block(x, 64, stride=2)
    for _ in range(36):  # Adjusting the number of blocks for ResNet-156
        x = res_block(x, 64)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)
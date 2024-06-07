import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Input

def conv_bn_relu(x, filters, kernel_size, strides):
    x = Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def group_conv(x, filters, cardinality):
    """Grouped convolution."""
    d = filters // cardinality
    groups = []
    for j in range(cardinality):
        group = Conv2D(d, (3, 3), padding='same', use_bias=False)(x)
        groups.append(group)
    x = layers.Concatenate()(groups)
    return x

def grouped_convolution_block(x, filters, cardinality):
    group_list = []
    grouped_channels = int(filters / cardinality)
    for c in range(cardinality):
        group = Conv2D(grouped_channels, (3, 3), padding='same', use_bias=False)(x)
        group_list.append(group)
    x = layers.Concatenate()(group_list)
    return x

def resnext_block(x, filters, cardinality, strides):
    """ResNeXt block."""
    shortcut = x

    # 1x1 Convolution (dimension reduction)
    x = Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Grouped Convolution
    x = group_conv(x, filters, cardinality)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # 1x1 Convolution (dimension restoration)
    x = Conv2D(filters * 2, (1, 1), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Matching dimensions for shortcut connection
    if strides != 1 or shortcut.shape[-1] != filters * 2:
        shortcut = Conv2D(filters * 2, (1, 1), strides=strides, padding='same', use_bias=False)(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    return x



def bottleneck_block(x, filters, cardinality, strides):
    init = x
    x = conv_bn_relu(x, filters, (1, 1), strides)

    # grouped convolution
    x = grouped_convolution_block(x, filters, cardinality)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters * 4, (1, 1), padding='same', use_bias=False)(x)  # increase filter size
    x = BatchNormalization()(x)

    # match the dimensions for shortcut connection
    if strides != 1 or init.shape[-1] != filters * 4:
        init = Conv2D(filters * 4, (1, 1), strides=strides, padding='same', use_bias=False)(init)
        init = BatchNormalization()(init)

    x = Add()([x, init])
    x = Activation('relu')(x)
    return x

def create_resnext29(input_shape, num_classes, cardinality=32):
    """Build ResNeXt-29."""
    inputs = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # ResNeXt blocks
    x = resnext_block(x, 64, cardinality, strides=1)
    for _ in range(2): x = resnext_block(x, 64, cardinality, strides=1)
    
    x = resnext_block(x, 128, cardinality, strides=2)
    for _ in range(2): x = resnext_block(x, 128, cardinality, strides=1)

    x = resnext_block(x, 256, cardinality, strides=2)
    for _ in range(2): x = resnext_block(x, 256, cardinality, strides=1)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_resnext50(input_shape, num_classes, cardinality=32):
    inputs = Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64, (3, 3), strides=1)
    
    # ResNeXt blocks
    x = bottleneck_block(x, 128, cardinality, strides=1)
    for _ in range(2): 
        x = bottleneck_block(x, 128, cardinality, strides=1)
    
    x = bottleneck_block(x, 256, cardinality, strides=2)
    for _ in range(3): 
        x = bottleneck_block(x, 256, cardinality, strides=1)
    
    x = bottleneck_block(x, 512, cardinality, strides=2)
    for _ in range(5): 
        x = bottleneck_block(x, 512, cardinality, strides=1)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def create_resnext101(input_shape, num_classes, cardinality=32):
    inputs = Input(shape=input_shape)
    x = conv_bn_relu(inputs, 64, (3, 3), strides=1)

    # Stage 1
    x = bottleneck_block(x, 128, cardinality, strides=1, downsample=True)
    for _ in range(2): 
        x = bottleneck_block(x, 128, cardinality, strides=1)
    
    # Stage 2
    x = bottleneck_block(x, 256, cardinality, strides=2, downsample=True)
    for _ in range(3): 
        x = bottleneck_block(x, 256, cardinality, strides=1)
    
    # Stage 3
    x = bottleneck_block(x, 512, cardinality, strides=2, downsample=True)
    for _ in range(22):  # The main difference is here, 22 blocks in the third stage
        x = bottleneck_block(x, 512, cardinality, strides=1)
    
    # Stage 4
    x = bottleneck_block(x, 1024, cardinality, strides=2, downsample=True)
    for _ in range(2): 
        x = bottleneck_block(x, 1024, cardinality, strides=1)
    
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    return model
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Activation, Dropout, GlobalAveragePooling2D, Dense


def conv2d_bn(x, filters, kernel_size, strides=1, activation='relu', padding='same', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = tf.keras.layers.BatchNormalization(scale=False)(x)
    if activation:
        x = Activation(activation)(x)
    return x

def inception_resnet_block(x, scale, block_type, block_idx, activation='relu'):
    if block_type == 'block35':
        branch_0 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(x, 32, 1)
        branch_1 = conv2d_bn(branch_1, 32, 3)
        branch_2 = conv2d_bn(x, 32, 1)
        branch_2 = conv2d_bn(branch_2, 48, 3)
        branch_2 = conv2d_bn(branch_2, 64, 3)
        branches = [branch_0, branch_1, branch_2]
    # You can define other block types (block17, block8) as per requirement.

    mixed = concatenate(branches, axis=-1)
    up = conv2d_bn(mixed, tf.keras.backend.int_shape(x)[-1], 1, activation=None, padding='same')
    x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale, output_shape=tf.keras.backend.int_shape(x)[1:], arguments={'scale': scale})([x, up])
    if activation is not None:
        x = Activation(activation)(x)
    return x

def InceptionResNetV22(input_shape=(32, 32, 3), num_classes=10):
    inputs = Input(shape=input_shape)

    # Initial Convolution
    x = conv2d_bn(inputs, 32, 3, strides=2)
    x = conv2d_bn(x, 32, 3)
    x = conv2d_bn(x, 64, 3)
    x = MaxPooling2D(3, strides=2)(x)

    # Inception-ResNet blocks
    x = inception_resnet_block(x, scale=0.17, block_type='block35', block_idx=1)

    # Other blocks (block17, block8) could be added in a similar fashion
    
    # Final pooling and prediction
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(0.2)(x)
    x = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs=x)
    return model






# Inception-ResNet-A block
def inception_resnet_a(input):
    # Branch 1
    branch_1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)

    # Branch 2
    branch_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)
    branch_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)

    # Branch 3
    branch_3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)
    branch_3 = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch_3)

    merged = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3])
    filter_bottleneck = input.shape.as_list()[-1]
    linear = layers.Conv2D(filter_bottleneck, (1, 1), padding='same', activation=None)(merged)
    output = layers.add([input, linear])
    output = layers.Activation('relu')(output)
    return output

# Reduced Inception-ResNet V1 Model for CIFAR-10
def InceptionResNetV1(input_shape=(32, 32, 3), num_classes=10):
    input_img = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Inception-ResNet-A blocks
    for _ in range(5):  # Adjust the number of blocks as needed
        x = inception_resnet_a(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer for classification
    x = layers.Dense(num_classes, activation='softmax')(x)

    # Create model
    model = Model(input_img, x, name='inception_resnet_v1_cifar10')
    return model

# Inception-ResNet-A block for V2
def inception_resnet_a_v2(input):
    # Branch 1
    branch_1 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)

    # Branch 2
    branch_2 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)
    branch_2 = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)

    # Branch 3
    branch_3 = layers.Conv2D(32, (1, 1), padding='same', activation='relu')(input)
    branch_3 = layers.Conv2D(48, (3, 3), padding='same', activation='relu')(branch_3)
    branch_3 = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(branch_3)

    merged = layers.Concatenate(axis=-1)([branch_1, branch_2, branch_3])
    filter_bottleneck = input.shape.as_list()[-1]
    linear = layers.Conv2D(filter_bottleneck, (1, 1), padding='same', activation=None)(merged)
    output = layers.add([input, linear])
    output = layers.Activation('relu')(output)
    return output

# Reduced Inception-ResNet V2 Model for CIFAR-10
def InceptionResNetV2(input_shape=(32, 32, 3), num_classes=10):
    input_img = layers.Input(shape=input_shape)

    # Initial convolution
    x = layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    # Inception-ResNet-A blocks for V2
    for _ in range(5):  # Adjust the number of blocks as needed
        x = inception_resnet_a_v2(x)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layer for classification
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(input_img, x, name='inception_resnet_v2_cifar10')
    return model
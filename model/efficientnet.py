import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, DepthwiseConv2D, Dense, BatchNormalization, ReLU, GlobalAveragePooling2D, Reshape, Multiply, Add, Activation
from tensorflow.keras.models import Model

class MBConvBlock(Layer):
    def __init__(self, input_filters, output_filters, kernel_size, strides, expand_ratio, se_ratio, id_skip, drop_rate, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.expand_ratio = expand_ratio
        self.id_skip = id_skip  # skip connection and drop connect
        self.drop_rate = drop_rate
        self.se_ratio = se_ratio

        # Expansion phase (Inverted Residual)
        filters = input_filters * expand_ratio
        if expand_ratio != 1:
            self.expansion = tf.keras.Sequential([
                Conv2D(filters, 1, padding='same', use_bias=False, name=name + 'expand_conv'),
                BatchNormalization(name=name + 'expand_bn'),
                ReLU(max_value=6., name=name + 'expand_relu')
            ], name=name + 'expand')

        # Depthwise Convolution
        self.depthwise_conv = DepthwiseConv2D(kernel_size, strides=strides, padding='same', use_bias=False, name=name + 'dwconv')
        self.bn1 = BatchNormalization(name=name + 'bn1')
        self.relu = ReLU(max_value=6., name=name + 'relu')

        # Squeeze and Excitation phase
        if 0 < se_ratio <= 1:
            se_filters = max(1, int(input_filters * se_ratio))
            self.se = tf.keras.Sequential([
                GlobalAveragePooling2D(name=name + 'se_squeeze'),
                Reshape((1, 1, filters), name=name + 'se_reshape'),
                Dense(se_filters, activation='relu', name=name + 'se_reduce'),
                Dense(filters, activation='sigmoid', name=name + 'se_expand')
            ], name=name + 'se')

        # Output phase
        self.output_conv = Conv2D(output_filters, 1, padding='same', use_bias=False, name=name + 'project_conv')
        self.output_bn = BatchNormalization(name=name + 'project_bn')

    def call(self, inputs, training=None):
        x = inputs

        # Expansion and Depthwise Convolution
        if self.expand_ratio != 1:
            x = self.expansion(x, training=training)
        x = self.depthwise_conv(x)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        # Squeeze and Excitation
        if hasattr(self, 'se'):
            x = self.squeeze_and_excitation(x)

        # Output phase
        x = self.output_conv(x)
        x = self.output_bn(x, training=training)

        # Skip connection and drop connect
        if self.id_skip and self.strides == 1 and self.input_filters == self.output_filters:
            if self.drop_rate > 0 and training:
                x = tf.nn.dropout(x, rate=self.drop_rate)
            x = Add()([x, inputs])

        return x

    def squeeze_and_excitation(self, x):
        x_se = self.se(x)
        return Multiply()([x, x_se])

# Building the EfficientNetB0 model
def EfficientNetB0(input_shape=(224, 224, 3), width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2, drop_connect_rate=0.2):
    # Base model of EfficientNetB0
    inputs = tf.keras.Input(shape=input_shape)
    # [Rest of the model architecture]
    # Construct the model here similar to the MBConvBlock above.
    # You would add instances of MBConvBlock with appropriate parameters to form the EfficientNetB0 architecture.

    x = inputs
    # [Add the layers according to the architecture]

    model = Model(inputs=inputs, outputs=x, name="EfficientNetB0")
    return model

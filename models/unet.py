from keras.models import Model
from keras.layers import Input, concatenate, Dropout, Reshape, Permute, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.initializers import orthogonal, constant, he_normal
from keras.regularizers import l2
from keras import backend as K
from keras.layers.normalization import BatchNormalization

def relu6(x):
    """Relu 6
    """
    return K.relu(x, max_value=6.0)

def encode_block(inputs, filters, k=3, dropout_rate=0.2):
    '''
    编码模块
    '''
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv, pool

def mid_block(inputs, filters, k=3, dropout_rate=0.2):
    '''
    中间卷积模块
    '''
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(inputs)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    return conv

def decode_block(inputs, inputs_enc, filters, k=3, dropout_rate=0.2):
    '''
    解码模块
    '''
    up = UpSampling2D(size=(2, 2))(inputs)
    # up = Conv2DTranspose(filters, kernel_size=(2, 2), strides=(2, 2), kernel_initializer=orthogonal())(inputs)
    # up = BatchNormalization()(up)
    # up = Activation(relu6)(up)
    up = concatenate([up, inputs_enc], axis=-1)
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(up)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    conv = Conv2D(filters, (k, k), padding='same', kernel_initializer=orthogonal())(conv)
    conv = BatchNormalization()(conv)
    conv = Activation(relu6)(conv)
    return conv

def Unet(nClasses, input_height=256, input_width=256, nChannels=3):
    inputs = Input(shape=(input_height, input_width, nChannels))
    filters = [16, 32, 64, 128, 192, 256]
    # encode
    conv_list = []
    curr_inputs = inputs
    for i in range(6):
        conv, pool = encode_block(curr_inputs, filters[i])
        curr_inputs = pool 
        conv_list.append(conv)
    # mid
    curr_inputs = mid_block(curr_inputs, filters[-1])
    # decode
    for i in range(6):
        conv = decode_block(curr_inputs, conv_list[-1-i], filters[-1-i])
        curr_inputs = conv 

    conv = Conv2D(nClasses, (1, 1), padding='same',
                   kernel_initializer=orthogonal(), kernel_regularizer=l2(0.005))(curr_inputs)

    conv = Reshape((nClasses, input_height * input_width))(conv)
    conv = Permute((2, 1))(conv)

    out = Activation('softmax')(conv)

    model = Model(input=inputs, output=out)

    return model


if __name__ == '__main__':
    model = Unet(2)
    print(model.summary())
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from scipy.misc import imread, imresize, imsave

from convnetskeras.customlayers import convolution2Dgroup, crosschannelnormalization, \
    splittensor, Softmax4D, Recalc, ReRank, ExtractDim, SoftReRank, ActivityRegularizerOneDim, RecalcExpand
from convnetskeras.imagenet_tool import synset_to_id, id_to_synset,synset_to_dfs_ids
from keras.regularizers import l1l2, activity_l1l2
from keras import backend as K
def convnet(network, outdim=1000, weights_path=None, heatmap=False,
            trainable=None, l1=0, l2=0, usemil=False, usemymil=False, k=1.0, usemysoftmil=False, softmink=1.0, softmaxk=1.0,
            sparsemil=False, sparsemill1=0., sparsemill2=0., saveact=False):
    """
    Returns a keras model for a CNN.

    BEWARE !! : Since the different convnets have been trained in different settings, they don't take
    data of the same shape. You should change the arguments of preprocess_img_batch for each CNN :
    * For AlexNet, the data are of shape (227,227), and the colors in the RGB order (default)
    * For VGG16 and VGG19, the data are of shape (224,224), and the colors in the BGR order

    It can also be used to look at the hidden layers of the model.

    It can be used that way :
    >>> im = preprocess_img_batch(['cat.jpg'])

    >>> # Test pretrained model
    >>> model = convnet('vgg_16', 'weights/vgg16_weights.h5')
    >>> sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    >>> model.compile(optimizer=sgd, loss='categorical_crossentropy')
    >>> out = model.predict(im)

    Parameters
    --------------
    network: str
        The type of network chosen. For the moment, can be 'vgg_16' or 'vgg_19'

    weights_path: str
        Location of the pre-trained model. If not given, the model will be trained

    heatmap: bool
        Says wether the fully connected layers are transformed into Convolution2D layers,
        to produce a heatmap instead of a


    Returns
    ---------------
    model:
        The keras model for this convnet

    output_dict:
        Dict of feature layers, asked for in output_layers.
    """


    # Select the network
    if network == 'vgg_16':
        convnet_init = VGG_16
    elif network == 'vgg_19':
        convnet_init = VGG_19
    elif network == 'alexnet':
        convnet_init = AlexNet
    convnet = convnet_init(outdim, weights_path, heatmap=False, l1=l1, l2=l2, usemil=usemil, usemymil=usemymil, k=k, usemysoftmil=usemysoftmil, softmink=softmink, softmaxk=softmaxk,\
        sparsemil=sparsemil, sparsemill1=sparsemill1, sparsemill2=sparsemill2, saveact=saveact)

    if not heatmap:
        return convnet
    else:
        convnet_heatmap = convnet_init(outdim, heatmap=True, l1=l1, l2=l2)

        for layer in convnet_heatmap.layers:
            if layer.name.startswith("conv"):
                orig_layer = convnet.get_layer(layer.name)
                layer.set_weights(orig_layer.get_weights())
            elif layer.name.startswith("dense"):
                orig_layer = convnet.get_layer(layer.name)
                W,b = orig_layer.get_weights()
                n_filter,previous_filter,ax1,ax2 = layer.get_weights()[0].shape
                new_W = W.reshape((previous_filter,ax1,ax2,n_filter))
                new_W = new_W.transpose((3,0,1,2))
                new_W = new_W[:,:,::-1,::-1]
                layer.set_weights([new_W,b])
        return convnet_heatmap

    return model




def VGG_16(outdim=1000, weights_path=None, heatmap=False, l1=0, l2=0, usemil=False, usemymil=False, k=1.):
    model = Sequential()
    if heatmap:
        model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
    else:
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if heatmap:
        model.add(Convolution2D(4096,7,7,activation="relu",W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_1"))
        model.add(Convolution2D(4096,1,1,activation="relu",W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_2"))
        model.add(Convolution2D(outdim,1,1,W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_3"))
        model.add(Softmax4D(axis=1,name="softmax"))
    elif usemil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        prediction = Flatten(name='flatten')(prediction_1)
        dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(prediction)
        prediction = Activation("softmax",name="softmax2")(dense_3)
        #prediction = MaxPooling2D((6,6), name='output')(prediction_1)
        #prediction = Flatten(name='flatten')(prediction)
        #prediction = Recalc(axis=1, name='Recalcmil')(prediction)
    elif usemymil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        prediction = Flatten(name='flatten')(dense_3)
        prediction = ReRank(k=k, label=1, name='output')(prediction)
    else:
        model.add(Flatten(name="flatten"))
        model.add(Dense(4096, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_2'))
        model.add(Dropout(0.5))
        model.add(Dense(outdim, W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_3'))
        model.add(Activation("softmax",name="softmax"))

    if weights_path:
        model.load_weights(weights_path)
    return model




def VGG_19(outdim=1000, weights_path=None,heatmap=False,l1=0, l2=0, usemil=False, usemymil=False, k=1.):
    model = Sequential()

    if heatmap:
        model.add(ZeroPadding2D((1,1),input_shape=(3,None,None)))
    else:
        model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu',W_regularizer=l1l2(l1=l1factor, l2=l2factor), name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv3_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv3_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv4_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv4_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv5_3'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='conv5_4'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    if heatmap:
        model.add(Convolution2D(4096,7,7,activation="relu",W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_1"))
        model.add(Convolution2D(4096,1,1,activation="relu",W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_2"))
        model.add(Convolution2D(outdim,1,1,W_regularizer=l1l2(l1=l1factor, l2=l2factor),name="dense_3"))
        model.add(Softmax4D(axis=1,name="softmax"))
    elif usemil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        prediction = Flatten(name='flatten')(prediction_1)
        dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(prediction)
        prediction = Activation("softmax",name="softmax2")(dense_3)
        #prediction = MaxPooling2D((6,6), name='output')(prediction_1)
        #prediction = Flatten(name='flatten')(prediction)
        prediction = RecalcExpand(axis=1, name='Recalcmil')(prediction)
    elif usemymil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        prediction = Flatten(name='flatten')(dense_3)
        prediction = ReRank(k=k, label=1, name='output')(prediction)
    else:
        model.add(Flatten())
        model.add(Dense(4096, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_1'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_2'))
        model.add(Dropout(0.5))
        model.add(Dense(outdim, W_regularizer=l1l2(l1=l1factor, l2=l2factor),name='dense_3'))
        model.add(Activation("softmax"))

    if weights_path:
        model.load_weights(weights_path)

    return model



def AlexNet(outdim=1000, weights_path=None, heatmap=False, l1=0, l2=0, usemil=False, usemymil=False, k=1., usemysoftmil=False, softmink=1., softmaxk=1.,\
    sparsemil=False, sparsemill1=0., sparsemill2=0., saveact=False):
    l1factor = l1
    l2factor = l2
    if heatmap:
        inputs = Input(shape=(3,None,None))
    else:
        inputs = Input(shape=(3,227,227))

    conv_1 = Convolution2D(96, 11, 11,subsample=(4,4),activation='relu', W_regularizer=l1l2(l1=l1factor, l2=l2factor),
                           name='conv_1')(inputs)

    conv_2 = MaxPooling2D((3, 3), strides=(2,2))(conv_1)
    conv_2 = crosschannelnormalization(name="convpool_1")(conv_2)
    conv_2 = ZeroPadding2D((2,2))(conv_2)
    conv_2 = merge([
        Convolution2D(128,5,5,activation="relu",name='conv_2_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
            splittensor(ratio_split=2,id_split=i)(conv_2)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_2")

    conv_3 = MaxPooling2D((3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1,1))(conv_3)
    conv_3 = Convolution2D(384,3,3,activation='relu',name='conv_3', W_regularizer=l1l2(l1=l1factor, l2=l2factor))(conv_3)

    conv_4 = ZeroPadding2D((1,1))(conv_3)
    conv_4 = merge([
        Convolution2D(192,3,3,activation="relu",name='conv_4_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
            splittensor(ratio_split=2,id_split=i)(conv_4)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_4")

    conv_5 = ZeroPadding2D((1,1))(conv_4)
    conv_5 = merge([
        Convolution2D(128,3,3,activation="relu",name='conv_5_'+str(i+1), W_regularizer=l1l2(l1=l1factor, l2=l2factor))(
            splittensor(ratio_split=2,id_split=i)(conv_5)
        ) for i in range(2)], mode='concat',concat_axis=1,name="conv_5")

    dense_1 = MaxPooling2D((3, 3), strides=(2,2),name="convpool_5")(conv_5)

    if heatmap:
        dense_1 = Convolution2D(4096,6,6,activation="relu",name="dense_1",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(4096,1,1,activation="relu",name="dense_2",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(outdim, 1,1,name="dense_3",W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        prediction = Softmax4D(axis=1,name="softmax")(dense_3)
    elif usemil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        #prediction = Flatten(name='flatten')(prediction_1)
        #dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(prediction)
        #prediction = Activation("softmax",name="softmax2")(dense_3)
        
        prediction_1 = MaxPooling2D((6,6), name='output')(prediction_1)
        prediction = Flatten(name='flatten')(prediction_1)
        prediction = Recalc(axis=1, name='Recalcmil')(prediction)
    elif usemymil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        #prediction = ExtractDim(axis=1, name='extract')(prediction_1)
        prediction = Flatten(name='flatten')(dense_3)
        prediction = ReRank(k=k, label=1, name='output')(prediction)
    elif usemysoftmil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_3 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        #prediction = ExtractDim(axis=1, name='extract')(prediction_1)
        prediction = Flatten(name='flatten')(dense_3)
        prediction = SoftReRank(softmink=softmink, softmaxk=softmaxk, label=1, name='output')(prediction)
    elif sparsemil:
        dense_1 = Convolution2D(128,1,1,activation='relu',name='mil_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Convolution2D(128,1,1,activation='relu',name='mil_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        prediction_1 = Convolution2D(1,1,1,activation='sigmoid',name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor),\
            activity_regularizer=activity_l1l2(l1=sparsemill1, l2=sparsemill2))(dense_2)
#        prediction_1 = Softmax4D(axis=1, name='softmax')(prediction_1)
        #dense_3 = Convolution2D(outdim,1,1,name='mil_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        #prediction_1 = Softmax4D(axis=1, name='softmax')(dense_3)
        #prediction_1 = ActivityRegularizerOneDim(l1=sparsemill1, l2=sparsemill2)(prediction_1)
        #prediction = MaxPooling2D((6,6), name='output')(prediction_1)
#        prediction_1 = Convolution2D(1, 3, 3, activation='sigmoid', border_mode='same', name='smooth', \
#            W_regularizer=l1l2(l1=l1factor, l2=l2factor), activity_regularizer=activity_l1l2(l1=sparsemill1, l2=sparsemill2))(prediction_1)
        prediction = Flatten(name='flatten')(prediction_1)
        if saveact:
          model = Model(input=inputs, output=prediction)
          return model
        prediction = RecalcExpand(axis=1, name='Recalcmil')(prediction)
    else:
        dense_1 = Flatten(name="flatten")(dense_1)
        dense_1 = Dense(4096, activation='relu',name='dense_1',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_1)
        dense_2 = Dropout(0.5)(dense_1)
        dense_2 = Dense(4096, activation='relu',name='dense_2',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_2)
        dense_3 = Dropout(0.5)(dense_2)
        dense_3 = Dense(outdim,name='dense_3',W_regularizer=l1l2(l1=l1factor, l2=l2factor))(dense_3)
        prediction = Activation("softmax",name="softmax")(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights_path:
        model.load_weights(weights_path)

    return model

def preprocess_image_batch(image_paths, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img_list = []

    for im_path in image_paths:
        img = imread(im_path, mode='RGB')
        if img_size:
            img = imresize(img,img_size)

        img = img.astype('float32')
        # We permute the colors to get them in the BGR order
        if color_mode=="bgr":
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
        # We normalize the colors with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))

        if crop_size:
            img = img[:,(img_size[0]-crop_size[0])//2:(img_size[0]+crop_size[0])//2
                      ,(img_size[1]-crop_size[1])//2:(img_size[1]+crop_size[1])//2]

        img_list.append(img)

    img_batch = np.stack(img_list, axis=0)
    if not out is None:
        out.append(img_batch)
    else:
        return img_batch





if __name__ == "__main__":
    ### Here is a script to compute the heatmap of the dog synsets.
    ## We find the synsets corresponding to dogs on ImageNet website
    s = "n02084071"
    ids = synset_to_dfs_ids(s)
    # Most of the synsets are not in the subset of the synsets used in ImageNet recognition task.
    ids = np.array([id for id in ids if id != None])

    im = preprocess_image_batch(['examples/dog.jpg'],color_mode="bgr")

    # Test pretrained model
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=True)
    model.compile(optimizer=sgd, loss='mse')


    out = model.predict(im)
    heatmap = out[0,ids,:,:].sum(axis=0)

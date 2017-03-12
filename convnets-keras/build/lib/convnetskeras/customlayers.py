import numpy as np
from keras.layers.core import  Lambda, Merge
from keras.layers.convolutional import Convolution2D
from keras import backend as K
from keras.regularizers import l1l2, Regularizer
from keras.engine import Layer

def crosschannelnormalization(alpha = 1e-4, k=2, beta=0.75, n=5,**kwargs):
    """
    This is the function used for cross channel normalization in the original
    Alexnet
    """
    def f(X):
        b, ch, r, c = X.shape
        half = n // 2
        square = K.square(X)
        extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0,2,3,1))
                                              , (0,half))
        extra_channels = K.permute_dimensions(extra_channels, (0,3,1,2))
        scale = k
        for i in range(n):
            scale += alpha * extra_channels[:,i:i+ch,:,:]
        scale = scale ** beta
        return X / scale

    return Lambda(f, output_shape=lambda input_shape:input_shape,**kwargs)



def splittensor(axis=1, ratio_split=1, id_split=0,**kwargs):
    def f(X):
        div = X.shape[axis] // ratio_split

        if axis == 0:
            output =  X[id_split*div:(id_split+1)*div,:,:,:]
        elif axis == 1:
            output =  X[:, id_split*div:(id_split+1)*div, :, :]
        elif axis == 2:
            output = X[:,:,id_split*div:(id_split+1)*div,:]
        elif axis == 3:
            output == X[:,:,:,id_split*div:(id_split+1)*div]
        else:
            raise ValueError("This axis is not possible")

        return output

    def g(input_shape):
        output_shape=list(input_shape)
        output_shape[axis] = output_shape[axis] // ratio_split
        return tuple(output_shape)

    return Lambda(f,output_shape=lambda input_shape:g(input_shape),**kwargs)




def convolution2Dgroup(n_group, nb_filter, nb_row, nb_col, **kwargs):
    def f(input):
        return Merge([
            Convolution2D(nb_filter//n_group,nb_row,nb_col)(
                splittensor(axis=1,
                            ratio_split=n_group,
                            id_split=i)(input))
            for i in range(n_group)
        ],mode='concat',concat_axis=1)

    return f


class Softmax4D(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape
        #axis_index = self.axis % len(input_shape)
        #return tuple([input_shape[i] for i in range(len(input_shape)) \
        #              if i != axis_index ])

class Recalc(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Recalc, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        response = K.reshape(x[:,self.axis], (-1,1))
        return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        #return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape
        #axis_index = self.axis % len(input_shape)
        #return tuple([input_shape[i] for i in range(len(input_shape)) \
        #              if i != axis_index ])

class RecalcExpand(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(RecalcExpand, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        response = K.max(x, axis=-1, keepdims=True) #K.reshape(x, (-1,1))
        return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        #return e / s

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0], 2])

class ExtractDim(Layer):
    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(ExtractDim, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None): # batchsize*2*6*6
        return x[:,self.axis,:,:]

    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[0],1,input_shape[2],input_shape[3]])

class ReRank(Layer):
    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
    def __init__(self,k=1,label=1,**kwargs):
        # k is the factor we force to be 1
        self.k = k*1.0
        self.label = label
        super(ReRank, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        import theano.tensor as T
        newx = T.sort(x)
        #response = K.reverse(newx, axes=1)
        #response = K.sum(x> 0.5, axis=1) / self.k
        return newx
        #response = K.reshape(newx,[-1,1])
        #return K.concatenate([1-response, response], axis=self.label)
        #response = K.reshape(x[:,self.axis], (-1,1))
        #return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        #return e / s

    def get_output_shape_for(self, input_shape):
        #return tuple([input_shape[0],input_shape[1],2])
        return input_shape

class SoftReRank(Layer):
    # Rerank is difficult. It is equal to the number of points (>0.5) to be fixed number.
    def __init__(self,softmink=1, softmaxk=1,label=1,**kwargs):
        # k is the factor we force to be 1
        self.softmink=softmink
        self.softmaxk=softmaxk
        self.label = label
        super(SoftReRank, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        newx = K.sort(x)
        #response = K.reverse(newx, axes=1)
        #response = K.sum(x> 0.5, axis=1) / self.k
        return K.concatenate([newx[:,:self.softmink], newx[:,newx.shape[1]-self.softmaxk:]], axis=-1)
        #response = K.reshape(newx,[-1,1])
        #return K.concatenate([1-response, response], axis=self.label)
        #response = K.reshape(x[:,self.axis], (-1,1))
        #return K.concatenate([1-response, response], axis=self.axis)
        #e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        #s = K.sum(e, axis=self.axis, keepdims=True)
        #return e / s

    def get_output_shape_for(self, input_shape):
        #return tuple([input_shape[0],input_shape[1],2])
        return tuple([input_shape[0], self.softmink+self.softmaxk])

class ActivityRegularizerOneDim(Regularizer):
    def __init__(self, l1=0., l2=0.,**kwargs):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)
        self.uses_learning_phase = True
        super(ActivityRegularizerOneDim, self).__init__(**kwargs)
        #self.layer = None

    def set_layer(self, layer):
        if self.layer is not None:
            raise Exception('Regularizers cannot be reused')
        self.layer = layer

    def __call__(self, loss):
        #if self.layer is None:
        #    raise Exception('Need to call `set_layer` on '
        #                    'ActivityRegularizer instance '
        #                    'before calling the instance.')
        regularized_loss = loss
        for i in range(len(self.layer.inbound_nodes)):
            output = self.layer.get_output_at(i)
            if self.l1:
                regularized_loss += K.sum(self.l1 * K.abs(output[:,:,:,1]))
            if self.l2:
                regularized_loss += K.sum(self.l2 * K.square(output[:,:,:,1]))
        return K.in_train_phase(regularized_loss, loss)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'l1': float(self.l1),
                'l2': float(self.l2)}
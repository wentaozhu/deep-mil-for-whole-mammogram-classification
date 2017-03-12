"""
TrainExtension subclass for calculating ROC AUC scores on monitoring
dataset(s), reported via monitor channels.
"""

__author__ = "Steven Kearnes"
__copyright__ = "Copyright 2014, Stanford University"
__license__ = "3-clause BSD"

import numpy as np
try:
    from sklearn.metrics import roc_auc_score, roc_curve
except ImportError:
    roc_auc_score = None
import logging
import theano
from theano import gof, config
from theano import tensor as T
from keras.callbacks import Callback
import os
#from pylearn2.train_extensions import TrainExtension

class AUCEpoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.auc = 0
    self.X_val, self.y_val = validation_data
    self.filepath = filepath
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      #print(np.sum(y_pred[:,1]))
      #y_true = np.argmax(self.y_val, axis=1)
      #y_pred = np.argmax(y_pred, axis=1)
      #print(y_true.shape, y_pred.shape)
      if self.mymil:
        score = roc_auc_score(self.y_val.max(axis=1), y_pred.max(axis=1))  
      else: score = roc_auc_score(self.y_val[:,1], y_pred[:,1])
      print("interval evaluation - epoch: {:d} - auc: {:.2f}".format(epoch, score))
      if score > self.auc:
        self.auc = score
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'auc'):
            os.remove(f)
        self.model.save(self.filepath+'auc'+str(score)+'ep'+str(epoch)+'.hdf5')

class RocAucScoreOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='roc_auc', use_c_code=theano.config.cxx):
        super(RocAucScoreOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.vector(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        try:
            roc_auc = roc_auc_score(y_true, y_score)
        except ValueError:
            roc_auc = np.nan
        #rvalue = np.array((roc_auc, prec, reca, f1))
        #[0][0]
        output_storage[0][0] = theano._asarray(roc_auc, dtype=config.floatX)

class PrecisionEpoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.prec = 0
    self.X_val, self.y_val = validation_data
    self.filepath = filepath
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      if self.mymil:
        y_true = self.y_val.max(axis=1)
        y_score = y_pred.max(axis=1)>0.5
      else:
        y_true = np.argmax(self.y_val, axis=1)
        y_score = np.argmax(y_pred, axis=1)
      #print(type(y_true), y_true.shape, type(y_score), y_score.shape)
      #print(y_score, y_true)
      TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
      FP = np.sum(y_true[y_score==1]==0)*1. #/ (y_true.shape[0]-sum(y_true))
      prec = TP / (TP+FP+1e-6)
      print("interval evaluation - epoch: {:d} - prec: {:.2f}".format(epoch, prec))
      if prec > self.prec:
        self.prec = prec
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'prec'):
            os.remove(f)
        self.model.save(self.filepath+'prec'+str(prec)+'ep'+str(epoch)+'.hdf5')

class PrecisionOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='precision', use_c_code=theano.config.cxx):
        super(PrecisionOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.vector(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        print(y_true.shape)
        y_true = np.argmax(y_true, axis=1)
        y_score = np.argmax(y_score, axis=1)
        #print(type(y_true), y_true.shape, type(y_score), y_score.shape)
        try:
            TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
            FP = np.sum(y_true[y_score==1]==0)*1. #/ (y_true.shape[0]-sum(y_true))
            prec = TP / (TP+FP+1e-6)
        except ValueError:
            prec = np.nan
        #rvalue = np.array((roc_auc, prec, reca, f1))
        #[0][0]
        output_storage[0][0] = theano._asarray(prec, dtype=config.floatX)

class RecallEpoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.filepath = filepath
    self.reca = 0
    self.X_val, self.y_val = validation_data
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      if self.mymil:
        y_true = self.y_val.max(axis=1)
        y_score = y_pred.max(axis=1)>0.5
      else:
        y_true = np.argmax(self.y_val, axis=1)
        y_score = np.argmax(y_pred, axis=1)
      #print(type(y_true), y_true.shape, type(y_score), y_score.shape)
      TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
      FN = np.sum(y_true[y_score==0]==1)*1. #/ sum(y_true)
      reca = TP / (TP+FN+1e-6)
      print("interval evaluation - epoch: {:d} - reca: {:.2f}".format(epoch, reca))
      if reca > self.reca:
        self.reca = reca
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'reca'):
            os.remove(f)
        self.model.save(self.filepath+'reca'+str(reca)+'ep'+str(epoch)+'.hdf5')

class RecallOp(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='recall', use_c_code=theano.config.cxx):
        super(RecallOp, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.vector(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        y_true = np.argmax(y_true, axis=1)
        y_score = np.argmax(y_score, axis=1)
        try:
            TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
            FN = np.sum(y_true[y_score==0]==1)*1. #/ sum(y_true)
            reca = TP / (TP+FN+1e-6)
        except ValueError:
            reca = np.nan
        #rvalue = np.array((roc_auc, prec, reca, f1))
        #[0][0]
        output_storage[0][0] = theano._asarray(reca, dtype=config.floatX)

class F1Epoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.filepath = filepath
    self.f1 = 0
    self.X_val, self.y_val = validation_data
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      #print(y_pred.shape)
      if self.mymil:
        y_true = self.y_val.max(axis=1)
        y_score = y_pred.max(axis=1)>0.5
      else:
        y_true = np.argmax(self.y_val, axis=1)
        y_score = np.argmax(y_pred, axis=1)
      #print(type(y_true), y_true.shape, type(y_score), y_score.shape)
      TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
      FP = np.sum(y_true[y_score==1]==0)*1. #/ (y_true.shape[0]-sum(y_true))
      #TN = np.sum(truey[predy==0]==0)*1. / (truey.shape[0]-sum(truey))
      FN = np.sum(y_true[y_score==0]==1)*1. #/ sum(y_true)
      #prec = TP / (TP+FP+1e-6)
      #reca = TP / (TP+FN+1e-6)
      #f1 = 2*prec*reca / (prec+reca+1e-6)
      f1 = 2*TP / (2*TP + FP + FN+1e-6)
      print("interval evaluation - epoch: {:d} - f1: {:.2f}".format(epoch, f1))
      if f1 > self.f1:
        self.f1 = f1
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'f1'):
            os.remove(f)
        self.model.save(self.filepath+'f1'+str(f1)+'ep'+str(epoch)+'.hdf5')

class ACCEpoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.filepath = filepath
    self.acc = 0
    self.X_val, self.y_val = validation_data
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      #print(y_pred.shape)
      if self.mymil:
        y_true = self.y_val.max(axis=1)
        y_score = y_pred.max(axis=1)#>0.5
      else:
        y_true = self.y_val[:,1] #np.argmax(self.y_val, axis=1)
        y_score = y_pred[:,1] #np.argmax(y_pred, axis=1)
      sortindex = np.argsort(y_score)
      y_score = y_score[sortindex]
      y_true = y_true[sortindex]
      bestacc, bestthresh = np.mean(y_true == np.ones_like(y_true)), y_score[0]-0.001
      for thresh in y_score:
        acc = np.mean(y_true == (y_score>thresh))
        if acc > bestacc:
          bestacc, bestthresh = acc, thresh
      y_score = y_score>bestthresh
      #y_score = y_score >0.5
      acc = np.mean(y_true == y_score)
      assert(acc == bestacc)
      print("interval evaluation - epoch: {:d} - acc: {:.2f}".format(epoch, acc))
      if acc > self.acc:
        self.acc = acc
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'acc'):
            os.remove(f)
        self.model.save(self.filepath+'acc'+str(acc)+'ep'+str(epoch)+'.hdf5')

class LossEpoch(Callback):
  def __init__(self, filepath, validation_data=(), interval=1, mymil=False):
    super(Callback, self).__init__()
    self.interval = interval
    self.filepath = filepath
    self.loss = 1e6
    self.X_val, self.y_val = validation_data
    self.mymil = mymil
  def on_epoch_end(self, epoch, logs={}):
    if epoch % self.interval == 0:
      y_pred = self.model.predict(self.X_val, verbose=0)
      #print(y_pred.shape)
      if self.mymil:
        y_true = self.y_val.max(axis=1)
        y_score = y_pred.max(axis=1)>0.5
      else:
        y_true = np.argmax(self.y_val, axis=1)
        y_score = y_pred[np.arange(len(y_true)), y_true] #y_pred[:, y_true] #np.argmax(y_pred, axis=1)
      loss = -np.mean(np.log(y_score+1e-6)) #-np.mean(y_true*np.log(y_score+1e-6) + (1-y_true)*np.log(1-y_score+1e-6))
      print('')
      print("interval evaluation - epoch: {:d} - loss: {:.2f}".format(epoch, loss))
      if loss < self.loss:
        self.loss = loss
        for f in os.listdir('./'):
          if f.startswith(self.filepath+'loss'):
            os.remove(f)
        self.model.save(self.filepath+'loss'+str(loss)+'ep'+str(epoch)+'.hdf5')

class F1Op(gof.Op):
    """
    Theano Op wrapping sklearn.metrics.roc_auc_score.

    Parameters
    ----------
    name : str, optional (default 'roc_auc')
        Name of this Op.
    use_c_code : WRITEME
    """
    def __init__(self, name='f1', use_c_code=theano.config.cxx):
        super(F1Op, self).__init__(use_c_code)
        self.name = name

    def make_node(self, y_true, y_score):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        y_true : tensor_like
            Target class labels.
        y_score : tensor_like
            Predicted class labels or probabilities for positive class.
        """
        y_true = T.as_tensor_variable(y_true)
        y_score = T.as_tensor_variable(y_score)
        output = [T.vector(name=self.name, dtype=config.floatX)]
        return gof.Apply(self, [y_true, y_score], output)

    def perform(self, node, inputs, output_storage):
        """
        Calculate ROC AUC score.

        Parameters
        ----------
        node : Apply instance
            Symbolic inputs and outputs.
        inputs : list
            Sequence of inputs.
        output_storage : list
            List of mutable 1-element lists.
        """
        if roc_auc_score is None:
            raise RuntimeError("Could not import from sklearn.")
        y_true, y_score = inputs
        y_true = np.argmax(y_true, axis=1)
        y_score = np.argmax(y_score, axis=1)
        try:
            TP = np.sum(y_true[y_score==1]==1)*1. #/ sum(y_true)
            FP = np.sum(y_true[y_score==1]==0)*1. #/ (y_true.shape[0]-sum(y_true))
            #TN = np.sum(truey[predy==0]==0)*1. / (truey.shape[0]-sum(truey))
            FN = np.sum(y_true[y_score==0]==1)*1. #/ sum(y_true)
            #prec = TP / (TP+FP+1e-6)
            #reca = TP / (TP+FN+1e-6)
            #f1 = 2*prec*reca / (prec+reca+1e-6)
            f1 = 2*TP / (2*TP +FP +FN)
        except ValueError:
            f1 = np.nan
        #rvalue = np.array((roc_auc, prec, reca, f1))
        #[0][0]
        output_storage[0][0] = theano._asarray(f1, dtype=config.floatX)
'''class RocAucChannel(TrainExtension):
    """
    Adds a ROC AUC channel to the monitor for each monitoring dataset.

    This monitor will return nan unless both classes are represented in
    y_true. For this reason, it is recommended to set monitoring_batches
    to 1, especially when using unbalanced datasets.

    Parameters
    ----------
    channel_name_suffix : str, optional (default 'roc_auc')
        Channel name suffix.
    positive_class_index : int, optional (default 1)
        Index of positive class in predicted values.
    negative_class_index : int or None, optional (default None)
        Index of negative class in predicted values for calculation of
        one vs. one performance. If None, uses all examples not in the
        positive class (one vs. the rest).
    """
    def __init__(self, channel_name_suffix='roc_auc', positive_class_index=1,
                 negative_class_index=None):
        self.channel_name_suffix = channel_name_suffix
        self.positive_class_index = positive_class_index
        self.negative_class_index = negative_class_index

    def setup(self, model, dataset, algorithm):
        """
        Add ROC AUC channels for monitoring dataset(s) to model.monitor.

        Parameters
        ----------
        model : object
            The model being trained.
        dataset : object
            Training dataset.
        algorithm : object
            Training algorithm.
        """
        m_space, m_source = model.get_monitoring_data_specs()
        state, target = m_space.make_theano_batch()

        y = T.argmax(target, axis=1)
        y_hat = model.fprop(state)[:, self.positive_class_index]

        # one vs. the rest
        if self.negative_class_index is None:
            y = T.eq(y, self.positive_class_index)

        # one vs. one
        else:
            pos = T.eq(y, self.positive_class_index)
            neg = T.eq(y, self.negative_class_index)
            keep = T.add(pos, neg).nonzero()
            y = T.eq(y[keep], self.positive_class_index)
            y_hat = y_hat[keep]

        roc_auc = RocAucScoreOp(self.channel_name_suffix)(y, y_hat)
        roc_auc = T.cast(roc_auc, config.floatX)
        for dataset_name, dataset in algorithm.monitoring_dataset.items():
            if dataset_name:
                channel_name = '{0}_{1}'.format(dataset_name,
                                                self.channel_name_suffix)
            else:
                channel_name = self.channel_name_suffix
            model.monitor.add_channel(name=channel_name,
                                      ipt=(state, target),
                                      val=roc_auc,
                                      data_specs=(m_space, m_source),
                                      dataset=dataset)'''
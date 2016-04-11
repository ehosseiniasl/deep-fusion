#!/usr/bin/env python
# encoding: utf-8
"""
COPYRIGHT

2016-03-11 - Created by Ehsan Hosseini-Asl

Deep Fusion Network

"""

import argparse
import numpy as np
import os
import Queue
import threading
from PIL import Image
import pickle
import cPickle
import random
import sys
import time
import urllib
import theano
import theano.tensor as T
from theano.tensor import nnet
from theano.tensor.signal import downsample
import ipdb
from itertools import izip
import math
import scipy.io as sio
from sklearn import preprocessing
FLOAT_PRECISION = np.float32

def adadelta_updates(parameters, gradients, rho, eps):

    gradients_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION),) for p in parameters ]
    deltas_sq = [ theano.shared(np.zeros(p.get_value().shape, dtype=FLOAT_PRECISION)) for p in parameters ]

    gradients_sq_new = [ rho*g_sq + (1-rho)*(g**2) for g_sq,g in izip(gradients_sq,gradients) ]

    deltas = [ (T.sqrt(d_sq+eps)/T.sqrt(g_sq+eps))*grad for d_sq,g_sq,grad in izip(deltas_sq,gradients_sq_new,gradients) ]

    deltas_sq_new = [ rho*d_sq + (1-rho)*(d**2) for d_sq,d in izip(deltas_sq,deltas) ]

    gradient_sq_updates = zip(gradients_sq,gradients_sq_new)
    deltas_sq_updates = zip(deltas_sq,deltas_sq_new)
    parameters_updates = [ (p,p - d) for p,d in izip(parameters,deltas) ]
    return gradient_sq_updates + deltas_sq_updates + parameters_updates



class HiddenLayer(object):

    def __init__(self, rng, input, n_in, n_out, share_with=None, activation=None):

        self.input = input

        if share_with:
            self.W = share_with.W
            self.b = share_with.b

            self.W_delta = share_with.W_delta
            self.b_delta = share_with.b_delta
        else:
            W_values = np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == nnet.sigmoid:
                W_values *= 4

            self.W = theano.shared(value=W_values, name='W', borrow=True)

            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, name='b', borrow=True)

            self.W_delta = theano.shared(
                    np.zeros((n_in, n_out), dtype=theano.config.floatX),
                    borrow=True
                )

            self.b_delta = theano.shared(value=b_values, borrow=True)

        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

        lin_output = T.dot(self.input, self.W) + self.b

        # ipdb.set_trace()
        if activation == 'tanh':
            self.output = T.tanh(lin_output)
        elif activation == 'sigmoid':
            self.output = T.nnet.sigmoid(lin_output)
        elif activation == 'relu':
            # self.output = T.maximum(lin_output, 0)
            self.output = 0.5 * (lin_output + abs(lin_output)) + 1e-9
        elif activation == 'cube_root':
            self.output = lin_output - ((1/3.)*self.output)**3
        else:
            self.output = lin_output

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class softmaxLayer(object):
    def __init__(self, input, n_in, n_out):

        self.W = theano.shared(
            value=np.zeros(
                (n_in,n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )

        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        self.W_delta = theano.shared(
                np.zeros((n_in,n_out), dtype=theano.config.floatX),
                borrow=True
            )

        self.b_delta = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX),
            name='b',
            borrow=True)

        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.params = [self.W, self.b]

        self.deltas = [self.W_delta, self.b_delta]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def get_state(self):
        return self.W.get_value(), self.b.get_value()

    def set_state(self, state):
        self.W.set_value(state[0], borrow=True)
        self.b.set_value(state[1], borrow=True)

class MLP(object):
    def __init__(self, hidden_size=None, act_func=None, L2_reg=None):
        rng = np.random.RandomState(None)
        input = T.fmatrix('input')
        label = T.lvector('labels')

        self.layers = []
        for i, h_size in enumerate(hidden_size):
            if i == 0:
                self.layers.append(HiddenLayer(rng,
                                      input,
                                      hidden_size[i],
                                      hidden_size[i+1],
                                      activation=act_func
                                               )
                                   )

            elif i < len(hidden_size)-2:
                self.layers.append(HiddenLayer(rng,
                                      self.layers[i-1].output,
                                      hidden_size[i],
                                      hidden_size[i+1],
                                      activation=act_func)
                                        )
            else:
                self.layers.append(softmaxLayer(
                    input=self.layers[i-1].output,
                    n_in=hidden_size[i],
                    n_out=hidden_size[i+1])
                )
                break

        self.params = sum([layer.params for layer in self.layers], [])

        L2 = [(layer.W**2).sum() for layer in self.layers]
        L2 = sum(L2)
        output_layer = self.layers[-1]
        self.cost = output_layer.negative_log_likelihood(label) + L2_reg * L2
        self.grads = T.grad(self.cost, self.params)

        self.updates = adadelta_updates(parameters=self.params,
                                        gradients=self.grads,
                                        rho=0.95,
                                        eps=1e-6)

        self.error = output_layer.errors(label)
        self.y_pred = output_layer.y_pred
        self.prob = output_layer.p_y_given_x.max(axis=1)
        self.true_prob = output_layer.p_y_given_x[T.arange(label.shape[0]), label]
        self.p_y_given_x = output_layer.p_y_given_x
        self.train = theano.function(
            inputs=[input, label],
            outputs=(self.error, self.cost, self.y_pred, self.prob),
            updates=self.updates
        )


        self.forward = theano.function(
            inputs=[input, label],
            outputs=(self.error, self.y_pred, self.prob, self.true_prob, self.p_y_given_x)
        )
        self.forward_hidden = theano.function(
            inputs=[input],
            outputs=(self.layers[-2].output)
        )

    def save(self, filename):
        f = open(filename, 'w')
        for l in self.layers:
            pickle.dump(l.get_state(), f, -1)
        f.close()

    def load(self, filename):
        f = open(filename)
        for l in self.layers:
            l.set_state(pickle.load(f))
        f.close()
        print 'model loaded from', filename

def do_train_classifiers(classifier_model=None, input1= None, target=None, batch_size=1, filename=None):
    loss = 0
    # loss_history = []
    epoch = 0
    progress_report = 10
    save_interval = 10
    last_save = time.time()
    # batch_size = 1
    try:
        print 'training classifier'
        print 'epoch\tloss\terror'
        while True:
            for i in xrange(input1.shape[0]/batch_size):
                classifier_input = input1
                error, cost, pred, prob = classifier_model.train(classifier_input[i*batch_size:(i+1)*batch_size].reshape(batch_size,classifier_input.shape[1]), target[i*batch_size:(i+1)*batch_size].reshape(batch_size,))
                error, y_pred, prob, true_prob, p_y_given_x = classifier_model.forward(classifier_input, target)
                # print cost
                loss +=cost
                epoch += 1
                if epoch % progress_report == 0:
                    loss /= progress_report
                    print '%d\t%.6f\t%.2f' % (epoch, loss, error)
                    sys.stdout.flush()
                    loss = 0
                if time.time() - last_save >= save_interval:
                    # loss_history.append(loss)
                    # filename = 'mlp_model2.pkl'
                    classifier_model.save(filename)
                    print 'model saved to', filename
                    return
                    # last_save = time.time()
    except KeyboardInterrupt:
        # filename = 'mlp_model2.pkl'
        classifier_model.save(filename)
        print 'model saved to', filename

def do_train_top_classifier(classifier_models=None, top_model=None, inputs= None, target=None, batch_size=1, filename=None):
    loss = 0
    # loss_history = []
    epoch = 0
    progress_report = 10
    save_interval = 100
    last_save = time.time()
    # batch_size = 1

    input_list = []
    for i, input in enumerate(inputs):
        # _, _, _, _, input = classifier_models[i].forward(inputs[i], target)
        input = classifier_models[i].forward_hidden(inputs[i])
        input_list.append(input)
    # ipdb.set_trace()
    input_tuple = tuple(input_list)
    classifier_input = np.concatenate(input_tuple, axis=1)

    try:
        print 'training top classifier'
        print 'epoch\tloss\terror'
        while True:
            for i in xrange(inputs[0].shape[0]/batch_size):
                error, cost, pred, prob = top_model.train(classifier_input[i*batch_size:(i+1)*batch_size].reshape(batch_size,classifier_input.shape[1]), target[i*batch_size:(i+1)*batch_size].reshape(batch_size,))
                error, y_pred, prob, true_prob, p_y_given_x = top_model.forward(classifier_input, target)
                # print cost
                loss +=cost
                epoch += 1
                if epoch % progress_report == 0:
                    loss /= progress_report
                    print '%d\t%.6f\t%.2f' % (epoch, loss, error)
                    sys.stdout.flush()
                    loss = 0
                if time.time() - last_save >= save_interval:
                    # loss_history.append(loss)
                    # filename = 'mlp_model2.pkl'
                    top_model.save(filename)
                    print 'model saved to', filename
                    break
                    # last_save = time.time()
    except KeyboardInterrupt:
        # filename = 'mlp_model2.pkl'
        top_model.save(filename)
        print 'model saved to', filename

def do_train_top_classifier_crossvalidate(classifier_models=None, top_model=None, inputs= None, target=None,
                                          batch_size=1, filename=None):



    input_list = []
    for i, input in enumerate(inputs):
        input = classifier_models[i].forward_hidden(inputs[i])
        input_list.append(input)
    input_tuple = tuple(input_list)
    data = np.concatenate(input_tuple, axis=1)
    test_error = np.empty((8,), dtype=FLOAT_PRECISION)
    test_y_pred = np.empty((32,), dtype=FLOAT_PRECISION)
    test_p_y_given_x = np.empty((32, 2), dtype=FLOAT_PRECISION)
    try:
        print 'training top classifier'
        print 'epoch\tloss\terror'

        for k in range(8):
            loss = 0
            epoch = 0
            progress_report = 10
            save_interval = 100
            last_save = time.time()
            print 'train fold%d' % k

            train_fold = np.empty((data.shape[0]-4, data.shape[1]), dtype=FLOAT_PRECISION)
            target_train_fold = np.empty(data.shape[0]-4, dtype=np.int64)
            test_fold = np.empty((4, data.shape[1]), dtype=FLOAT_PRECISION)
            target_test_fold = np.empty(4, dtype=np.int64)
            cc = 0
            for ii in range(8):
                try:
                    if ii != k:
                        train_fold[cc*4:(cc+1)*4] = data[ii*4:(ii+1)*4]
                        target_train_fold[cc*4:(cc+1)*4] = target[ii*4:(ii+1)*4]
                        cc += 1
                except:
                    ipdb.set_trace()
            test_fold = data[k*4:(k+1)*4]
            target_test_fold = target[k*4:(k+1)*4]
            batch_size = train_fold.shape[0]

            for i in xrange(train_fold.shape[0]/batch_size):
                error, cost, pred, prob = top_model.train(train_fold[i*batch_size:(i+1)*batch_size].reshape(batch_size,
                                                                                                   train_fold.shape[1]),
                                                          target_train_fold[i*batch_size:(i+1)*batch_size].reshape(batch_size,))
                loss +=cost
                epoch += 1
                if epoch % progress_report == 0:
                    loss /= progress_report
                    print '%d\t%.6f\t%.2f' % (epoch, loss, error)
                    sys.stdout.flush()
                    loss = 0
                if time.time() - last_save >= save_interval:
                    top_model.save(filename)
                    print 'model saved to', filename
                    break

            error, y_pred, prob, true_prob, p_y_given_x = top_model.forward(test_fold, target_test_fold)
            print 'fold%d error:%.2f' % (k, error)
            test_error[k] = error
            test_y_pred[k*4:(k+1)*4] = y_pred
            test_p_y_given_x[k*4:(k+1)*4, :] = p_y_given_x
    except KeyboardInterrupt:
        top_model.save(filename)
        print 'model saved to', filename

    f = open('test_kfold.pkl', 'w')
    pickle.dump((test_error, test_y_pred, test_p_y_given_x), f, -1)
    f.close()



def ProcessCommandLine():
    parser = argparse.ArgumentParser(description='train DCCA for MLP')
    parser.add_argument('-m', '--model',
                        help='start with this model')
    parser.add_argument('-set', '--dataset', type=int, default=None,
                        help='training on dataset')
    parser.add_argument('-l2', '--l2reg', type=float, default=0.00,
                        help='l2 regularization')
    parser.add_argument('-tr', '--do_train', action='store_true',
                         help='do training')
    parser.add_argument('-trt', '--do_train_top', action='store_true',
                         help='do training top model')
    parser.add_argument('-ts', '--do_test', action='store_true',
                        help='do testing')
    parser.add_argument('-rd', '--do_reduce', action='store_true',
                        help='reduce features')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-9,
                        help='learning rate step')
    parser.add_argument('-act', '--activation', type=str, default='sigmoid',
                        help='activation function')
    parser.add_argument('-batch', type=int, default=1,
                        help='batch size')
    args = parser.parse_args()
    return args.model, args.dataset, args.l2reg, args.do_train, args.do_train_top, args.do_test, args.do_reduce, args.learning_rate, args.activation, args.batch

def print_results():
    f = open('test_kfold.pkl', 'r')
    res = pickle.load(f)
    f.close()
    f = open('kfold_prediction.txt', 'w')
    print >>f, '\terror'
    for k in range(8):
        print >>f, 'fold %d: %.2f' % (k, res[0][k])
    print >>f, '%d-fold accuracy: %.4f' % (res[0].shape, 1.-np.mean(res[0]))
    print >>f, '\n\nclass\tprob'
    for i, item in enumerate(res[1]):
        print >>f, '%d\t%.2f' % (item, res[2][i, int(item)])
    f.close()

def main():
    model, dataset, l2reg, do_train, do_train_top, do_test, do_reduce, learning_rate, activation, batchsize = ProcessCommandLine()
    print 'learning rate:', learning_rate
    print 'L2 regularization:', l2reg
    print 'activation:', activation
    print 'training on dataset#', dataset
    print 'batch size:', batchsize
    if do_reduce:
        print 'reduce features'

    sys_dir = os.getcwd()

    data_dir = sys_dir+'/data/restoffeaturesforthedatabase/'
    if do_train:
        for filename in ['FL', 'FR', 'OL', 'OR', 'PL', 'PR', 'TL', 'TR']:
            for featurename in range(1, 9):
                featurename = str(featurename)
                files = os.listdir(data_dir+filename+'/'+featurename+'/')
                for file in files:
                    if 'traindata' in file:
                        data_x = sio.loadmat(data_dir+filename+'/'+featurename+'/'+file)
                        data_x = np.asarray(data_x[data_x.keys()[0]], dtype=FLOAT_PRECISION)
                    else:
                        data_y = sio.loadmat(data_dir+filename+'/'+featurename+'/'+file)
                        data_y = np.asarray(data_y[data_y.keys()[0]], dtype=np.int64)
                n_features, n_samples =  data_x.shape
                # ipdb.set_trace()
                if n_features>=1000:
                    print 'reduce features'
                    tmp_x = np.empty((n_features/100, n_samples), dtype=FLOAT_PRECISION)
                    # ipdb.set_trace()
                    for idx, i in enumerate(xrange(tmp_x.shape[0])):
                        tmp_x[idx, :] = data_x[i, :]
                    data_x = tmp_x

                idx = np.arange(n_samples)
                np.random.shuffle(idx)
                data_x_shuffled = np.empty(data_x.shape, dtype=FLOAT_PRECISION)
                data_y_shuffled = np.empty(data_y.shape, dtype=np.int64)
                for i, t in enumerate(idx):
                    data_x_shuffled[:, i] = data_x[:, t]
                    data_y_shuffled[:, i] = data_y[:, t]
                data_x = data_x_shuffled
                data_y = data_y_shuffled.reshape(data_y.shape[1],)
                data_y[data_y==1] = 0
                data_y[data_y==2] = 1

                print '\nmlp for %s/%s' % (filename, featurename)
                model_name = '%s_%s.pkl' % (filename, featurename)
                n_output = np.unique(data_y).shape[0]
                classifier = MLP(hidden_size=[data_x.shape[0], 20, 10, n_output], L2_reg=l2reg)
                do_train_classifiers(classifier_model=classifier,
                                     input1=data_x.T,
                                     target=data_y,
                                     batch_size=n_samples,
                                     filename=model_name)
    elif do_train_top:
        model_dir = 'models_64/'
        classifier_models = []
        inputs_x = []
        data_x = sio.loadmat(data_dir+'FL/1/traindata_brain_EucFL_bins_Norm.mat')
        n_features, n_samples = data_x[data_x.keys()[0]].shape
        # ipdb.set_trace()
        shuffle_idx = np.arange(n_samples)
        np.random.shuffle(shuffle_idx)

        for filename in ['FL', 'FR', 'OL', 'OR', 'PL', 'PR', 'TL', 'TR']:
            for featurename in range(1, 9):
                featurename = str(featurename)
                files = os.listdir(data_dir+filename+'/'+featurename+'/')
                for file in files:
                    if 'traindata' in file:
                        data_x = sio.loadmat(data_dir+filename+'/'+featurename+'/'+file)
                        data_x = np.asarray(data_x[data_x.keys()[0]], dtype=FLOAT_PRECISION)
                    else:
                        data_y = sio.loadmat(data_dir+filename+'/'+featurename+'/'+file)
                        data_y = np.asarray(data_y[data_y.keys()[0]], dtype=np.int64)

                n_features, n_samples = data_x.shape
                if n_features>=1000:
                    print 'reduce features'
                    tmp_x = np.empty((n_features/100, n_samples), dtype=FLOAT_PRECISION)
                    # ipdb.set_trace()
                    for idx, i in enumerate(xrange(tmp_x.shape[0])):
                        tmp_x[idx, :] = data_x[i, :]
                    data_x = tmp_x
                # ipdb.set_trace()

                data_x_shuffled = np.empty(data_x.shape, dtype=FLOAT_PRECISION)
                data_y_shuffled = np.empty(data_y.shape, dtype=np.int64)
                for i, t in enumerate(shuffle_idx):
                    data_x_shuffled[:, i] = data_x[:, t]
                    data_y_shuffled[:, i] = data_y[:, t]
                data_x = data_x_shuffled
                data_y = data_y_shuffled.reshape(data_y.shape[1],)
                data_y[data_y==1] = 0
                data_y[data_y==2] = 1
                n_output = np.unique(data_y).shape[0]
                inputs_x.append(data_x.T)
                classifier = MLP(hidden_size=[data_x.shape[0], 20, 10, n_output], L2_reg=l2reg)
                model_name = '%s_%s.pkl' % (filename, featurename)
                classifier.load(model_dir+model_name)
                classifier_models.append(classifier)


        top_classifier = MLP(hidden_size=[10*64, 20, 10, 2], L2_reg=l2reg)
        do_train_top_classifier_crossvalidate(classifier_models=classifier_models,
                                top_model=top_classifier,
                                inputs=inputs_x,
                                target=data_y,
                                filename='top_classifier64.pkl')
        print_results()

if __name__ == '__main__':
    sys.exit(main())

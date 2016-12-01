#!/usr/bin/env python
# -*- coding: utf-8 -*-

import lasagne
import theano
import theano.tensor as T
import numpy as np
from tqdm import tqdm
from time import time
import sys, os
from sklearn.externals import joblib

from config import Config
from tools import TimePrint
from data_io import get_texture_iter, save_tensor


##
# define shortcuts for lasagne functions
relu        = lasagne.nonlinearities.rectify
lrelu       = lasagne.nonlinearities.LeakyRectify(0.2)
tanh        = lasagne.nonlinearities.tanh
sigmoid     = lasagne.nonlinearities.sigmoid
conv        = lambda incoming, num_filters, filter_size, W, b, nonlinearity: \
                lasagne.layers.Conv2DLayer(incoming, num_filters, filter_size, stride=(2,2), pad='same', W=W, b=b, flip_filters=True, nonlinearity=nonlinearity)
tconv       = lambda incoming, num_filters, filter_size, W, nonlinearity: lasagne.layers.TransposedConv2DLayer(incoming, num_filters, filter_size, stride=(2,2), crop='same', W=W, nonlinearity=nonlinearity)
batchnorm   = lasagne.layers.batch_norm

# bias and weight initializations
w_init      = lasagne.init.Normal(std=0.02)
b_init      = lasagne.init.Constant(val=0.0)
g_init      = lasagne.init.Normal(mean=1.,std=0.02)


def sharedX(X, dtype=theano.config.floatX, name=None):
    return theano.shared(np.asarray(X, dtype=dtype), name=name)


##
# network code
class SGAN(object):

    def __init__(self, name=None):
        '''
        @static configuration class
        @param name     load stored sgan model
        '''
        self.config = Config
        if name is not None:
            print "loading parameters from file:",name

            vals =joblib.load(name)
            
            
            self.dis_W = [sharedX(p) for p in vals["dis_W"]]
            self.dis_g = [sharedX(p) for p in vals["dis_g"]]
            self.dis_b = [sharedX(p) for p in vals["dis_b"]]
    
            self.gen_W = [sharedX(p) for p in vals["gen_W"]]
            self.gen_g = [sharedX(p) for p in vals["gen_g"]]
            self.gen_b = [sharedX(p) for p in vals["gen_b"]]
            
            ##now overwrite the static config with the correct values
            self.config.gen_ks = []
            self.config.gen_fn = []
            l = len(vals["gen_W"])
            for i in range(l):
                if i==0:
                    self.config.nz = vals["gen_W"][i].shape[0]
                else:
                    self.config.gen_fn +=[vals["gen_W"][i].shape[0]]
                self.config.gen_ks += [(vals["gen_W"][i].shape[2],vals["gen_W"][i].shape[3])]
            self.config.nc = vals["gen_W"][i].shape[1]
            self.config.gen_fn +=[self.config.nc]

            self.config.dis_ks = []
            self.config.dis_fn = []
            l = len(vals["dis_W"])
            for i in range(l):
                self.config.dis_fn +=[vals["dis_W"][i].shape[1]]   
                self.config.dis_ks += [(vals["gen_W"][i].shape[2],vals["gen_W"][i].shape[3])]             

            self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
            self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)

        else:
            self._setup_gen_params(self.config.gen_ks, self.config.gen_fn)
            self._setup_dis_params(self.config.dis_ks, self.config.dis_fn)
            ##
            # sample the initial weights and biases
            self._sample_initials()

        self._build_sgan()


    def save(self,name):
        print "saving SGAN parameters in file: ", name
        vals = {}
        vals["config"] = self.config
        vals["dis_W"] = [p.get_value() for p in self.dis_W]
        vals["dis_g"] = [p.get_value() for p in self.dis_g]
        vals["dis_b"] = [p.get_value() for p in self.dis_b]

        vals["gen_W"] = [p.get_value() for p in self.gen_W]
        vals["gen_g"] = [p.get_value() for p in self.gen_g]
        vals["gen_b"] = [p.get_value() for p in self.gen_b]
        
        joblib.dump(vals,name,True)


    def _setup_gen_params(self, gen_ks, gen_fn):
        '''
        set up the parameters, i.e. filter sizes per layer and depth, of the generator
        '''
        ## 
        # setup generator parameters and sanity checks
        if gen_ks==None:
            self.gen_ks = [(5,5)] * 5   # set to standard 5-layer net
        else:
            self.gen_ks = gen_ks

       
        self.gen_depth = len(self.gen_ks)

        if gen_fn!=None:
            assert len(gen_fn)==len(self.gen_ks), 'Layer number of filter numbers and sizes does not match.'
            self.gen_fn = gen_fn
        else:
            self.gen_fn = [64] * self.gen_depth
    

    def _setup_dis_params(self, dis_ks, dis_fn):
        '''
        set up the parameters, i.e. filter sizes per layer and depth, of the discriminator
        '''
        ##
        # setup discriminator parameters
        if dis_ks==None:
            self.dis_ks = [(5,5)] * 5   # set to standard 5-layer net
        else:
            self.dis_ks = dis_ks

        self.dis_depth = len(dis_ks)

        if dis_fn!=None:
            assert len(dis_fn)==len(self.dis_ks), 'Layer number of filter numbers and sizes does not match.'
            self.dis_fn = dis_fn
        else:
            self.dis_fn = [64] * self.dis_depth


    def _sample_initials(self):
        '''
        sample the initial weights and biases and push them back to internal lists
        '''
        self.dis_W = []
        self.dis_b = []
        self.dis_g = []


        self.dis_W.append( sharedX( w_init.sample( (self.dis_fn[0], self.config.nc, self.dis_ks[0][0], self.dis_ks[0][1]) )) )
        for l in range(self.dis_depth-1):
            self.dis_W.append( sharedX( w_init.sample( (self.dis_fn[l+1], self.dis_fn[l], self.dis_ks[l+1][0], self.dis_ks[l+1][1]) ) ) )
            self.dis_b.append( sharedX( b_init.sample( (self.dis_fn[l+1]) ) ) )
            self.dis_g.append( sharedX( g_init.sample( (self.dis_fn[l+1]) ) ) )
    
        self.gen_b = []
        self.gen_g = []
        for l in range(self.gen_depth-1):
            self.gen_b += [sharedX( b_init.sample( (self.gen_fn[l]) ) ) ]
            self.gen_g += [sharedX( g_init.sample( (self.gen_fn[l]) ) ) ]

        self.gen_W = []
        
        last = self.config.nz
        for l in range(self.gen_depth-1):
            self.gen_W +=[sharedX( w_init.sample((last,self.gen_fn[l], self.gen_ks[l][0],self.gen_ks[l][1])))]
            last=self.gen_fn[l]

        self.gen_W +=[sharedX( w_init.sample((last,self.gen_fn[-1], self.gen_ks[-1][0],self.gen_ks[-1][1])))]


    def _spatial_generator(self, inlayer):
        '''
        creates a SGAN generator network
        @param  inlayer     Lasagne layer
        '''
        layers  = [inlayer]
        for l in range(self.gen_depth-1):
            layers.append( batchnorm(tconv(layers[l], self.gen_fn[l], self.gen_ks[l],self.gen_W[l], nonlinearity=relu),gamma=self.gen_g[l],beta=self.gen_b[l]) )
        output  = tconv(layers[-1], self.gen_fn[-1], self.gen_ks[-1],self.gen_W[-1] , nonlinearity=tanh)

        return output

    def _spatial_discriminator(self, inlayer):
        '''
        creates a SGAN discriminator network
        @param  inlayer     Lasagne layer
        '''
        layers  = [inlayer]
        layers.append( conv(layers[-1], self.dis_fn[0], self.dis_ks[0], self.dis_W[0], None, nonlinearity=lrelu) )
        for l in range(1,self.dis_depth-1):
            layers.append( batchnorm(conv(layers[-1], self.dis_fn[l], self.dis_ks[l], self.dis_W[l],None,nonlinearity=lrelu),gamma=self.dis_g[l-1],beta=self.dis_b[l-1]) )
        output = conv(layers[-1], self.dis_fn[-1], self.dis_ks[-1], self.dis_W[-1], None, nonlinearity=sigmoid)

        return output


    def _build_sgan(self):
        ##
        # network
        Z               = lasagne.layers.InputLayer((None,self.config.nz,None,None))   # leave batch_size and shape unspecified for now
        X               = lasagne.layers.InputLayer((self.config.batch_size,self.config.nc,self.config.npx,self.config.npx))   # leave batch_size and shape unspecified for now

        gen_X           = self._spatial_generator(Z)
        d_real          = self._spatial_discriminator(X)
        d_fake          = self._spatial_discriminator(gen_X)

        prediction_gen  = lasagne.layers.get_output(gen_X)
        prediction_real = lasagne.layers.get_output(d_real)
        prediction_fake = lasagne.layers.get_output(d_fake)

        params_g        = lasagne.layers.get_all_params(gen_X, trainable=True)
        params_d        = lasagne.layers.get_all_params(d_real, trainable=True)

        ##
        # objectives
        l2_gen          = lasagne.regularization.regularize_network_params(gen_X, lasagne.regularization.l2)
        l2_dis          = lasagne.regularization.regularize_network_params(d_real, lasagne.regularization.l2)

        obj_d= -T.mean(T.log(1-prediction_fake)) - T.mean( T.log(prediction_real)) + self.config.l2_fac * l2_dis
        obj_g= -T.mean(T.log(prediction_fake)) + self.config.l2_fac * l2_gen

        ##
        # updates
        updates_d       = lasagne.updates.adam(obj_d, params_d, self.config.lr, self.config.b1)
        updates_g       = lasagne.updates.adam(obj_g, params_g, self.config.lr, self.config.b1)

        # ##
        # # theano functions
        TimePrint("Compiling the network...\n")
        self.train_d    = theano.function([X.input_var, Z.input_var], obj_d, updates=updates_d, allow_input_downcast=True)
        TimePrint("Discriminator done.")
        self.train_g    = theano.function([Z.input_var], obj_g, updates=updates_g, allow_input_downcast=True)
        TimePrint("Generator done.")
        self.generate   = theano.function([Z.input_var], prediction_gen, allow_input_downcast=True)
        TimePrint("generate function done.")


if __name__=="__main__":

    c           = Config

    if c.load_name   == None:
        sgan        = SGAN()
    else:
        sgan        = SGAN(name='models/' + c.load_name)
        
    c.print_info()

    ##
    # 
    z_sample        = np.random.uniform(-1.,1., (1, c.nz, c.zx_sample, c.zx_sample) )    # the sample for which to plot samples

    epoch           = 0
    tot_iter        = 0


    while True:
        epoch       += 1
        print("Epoch %d" % epoch)

        Gcost = []
        Dcost = []

        iters = c.epoch_iters / c.batch_size
        for it, samples in enumerate(tqdm(c.data_iter, total=iters)):
            if it >= iters:
                break
            tot_iter+=1

            Znp = np.random.uniform(-1.,1., (c.batch_size, c.nz, c.zx, c.zx) )

            if tot_iter % (c.k+1) == 0:
                cost = sgan.train_g(Znp)
                Gcost.append(cost)
            else:
                cost = sgan.train_d(samples,Znp)
                Dcost.append(cost)

        print "Gcost=", np.mean(Gcost), "  Dcost=", np.mean(Dcost)

        data = sgan.generate(z_sample)

        save_tensor(data[0], 'samples/%s_epoch%d.jpg' % (c.save_name,epoch))
        sgan.save('models/%s_epoch%d.sgan'%(c.save_name,epoch))


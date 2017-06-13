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

from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams(seed=234)

##given parameters from config, calculate the Z noise tensor
## the tensor has channels of type Global, Local, Periodic dimensions
## @param zx spatial size, now implemented only in square shapes
## @param batch_size how many instances in mini-batch
## @param zx_quilt if not None, will set  some parts of the global dims to random values in different spatial regions (tiles), else all global dim. are equal to the same vector spatially
def sample_noise_tensor(batch_size,zx,zx_quilt=None):
    Z = np.zeros((batch_size,Config.nz,zx,zx))
    Z[:,Config.nz_global:Config.nz_global+Config.nz_local] = np.random.uniform(-1.,1., (batch_size, Config.nz_local, zx, zx) )
    
    if zx_quilt is None:
        Z[:,:Config.nz_global] = np.random.uniform(-1.,1., (batch_size, Config.nz_global, 1, 1) )
    else:
        for i in range(zx/zx_quilt):
            for j in range(zx/zx_quilt):
                Z[:,:Config.nz_global,i*zx_quilt:(i+1)*zx_quilt, j*zx_quilt:(j+1)*zx_quilt]    =np.random.uniform(-1.,1., (batch_size, Config.nz_global, 1, 1) )
    
    
    if Config.nz_periodic > 0:
        for i,pixel in zip(range(1,Config.nz_periodic+1),np.linspace(30,130,Config.nz_periodic)):
            band =  np.pi*(0.5*i / float(Config.nz_periodic) +0.5   )##initial values for numerical stability            
            ##just horizontal and vertical coordinate indices
            for h in range(zx):
                Z[:, -i*2,:, h] = h * band 
            for w in range(zx):
                Z[:, -i * 2 + 1, w] = w * band 
    return Z
    
class PeriodicLayer(lasagne.layers.Layer):

    def __init__(self,incoming,config,wave_params):
        print "initing layer"
        self.config = config       
        self.wave_params = wave_params
        print "done"
        self.input_layer= incoming
        self.input_shape = incoming.output_shape
        self.get_output_kwargs = []
        self.params = {}
        for p in wave_params:
            self.params[p] = set('trainable')

    ##the frequency gating MLP
    def _wave_calculation(self,Z):
        if self.config.nz_periodic ==0:
            return Z
        nPeriodic = self.config.nz_periodic

        if self.config.nz_global > 0:  # #MLP or directly a weight vector in case of no Global dims
            h = T.tensordot(Z[:, :self.config.nz_global], self.wave_params[0], [1, 0]).dimshuffle(0, 3, 1, 2) + self.wave_params[1].dimshuffle('x', 0, 'x', 'x')
            band0 = (T.tensordot(relu(h),self.wave_params[2], [1, 0]).dimshuffle(0, 3, 1, 2)) + self.wave_params[3].dimshuffle('x', 0, 'x', 'x')  # #moved relu inside
        else:
            band0 = self.wave_params[0].dimshuffle('x', 0, 'x', 'x')

        band1 = Z[:, -nPeriodic * 2::2] * band0[:, :nPeriodic] + Z[:, -nPeriodic * 2 + 1::2] * band0[:, nPeriodic:2 * nPeriodic]
        band2 = Z[:, -nPeriodic * 2::2] * band0[:, 2 * nPeriodic:3 * nPeriodic] + Z[:, -nPeriodic * 2 + 1::2] * band0[:, 3 * nPeriodic:]
        band = T.concatenate([band1 , band2], axis=1)       
        ##random phase added here, use random stream generator
        band += srng.uniform((Z.shape[0],nPeriodic * 2)).dimshuffle(0,1, 'x', 'x') *np.pi*2
        return T.concatenate([Z[:, :-2 * nPeriodic], T.sin(band)], axis=1)

    def get_output_for(self, input, **kwargs):
        return self._wave_calculation(input)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],input_shape[1]+self.config.nz_periodic*2,input_shape[2],input_shape[3])     


periodic = lambda incoming,wave_params: PeriodicLayer(incoming,Config,wave_params)

##
# network code
class PSGAN(object):

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

            self.wave_params = [sharedX(p) for p in vals["wave_params"]]
            
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
            
            ##setup the initial MLP frequency gating weights
            self._setup_wave_params()

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

        vals["wave_params"] = [p.get_value() for p in self.wave_params]
        
        joblib.dump(vals,name,True)

    
    def _setup_wave_params(self):
        '''
        set up the parameters of the periodic dimensions, i.e. the weigts of the gating MLP
        '''

        if self.config.nz_periodic:
            nPeriodic = self.config.nz_periodic
            nperiodK = self.config.nz_periodic_MLPnodes
            if self.config.nz_global >0 and nperiodK>0:##K is hidden nodes layer; MLP depending on global dimensions
                lin1 =  sharedX( g_init.sample( (self.config.nz_global,nperiodK)))
                bias1 = sharedX( g_init.sample( (nperiodK)))
                lin2 =  sharedX( g_init.sample( (nperiodK,nPeriodic * 2*2)))
                bias2 = sharedX( g_init.sample( (nPeriodic * 2*nAffine)))
                self.wave_params = [lin1,bias1,lin2,bias2]
            else:##in case no global dimensions learn global wave numbers
                bias2 = sharedX( g_init.sample( (nPeriodic * 2*2)))
                self.wave_params = [bias2]
            a = np.zeros(nPeriodic * 2*2)              
            a[:nPeriodic]=1#x
            a[nPeriodic:2*nPeriodic]=0#y
            a[2*nPeriodic:3*nPeriodic]=0#x
            a[3*nPeriodic:]=1#y
            self.wave_params[-1].set_value(np.float32(a)) 
        else:
            self.wave_params = []

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
        layers.append(periodic(inlayer,self.wave_params))##TODO what is the inputshape of that guy?
        for l in range(self.gen_depth-1):
            layers.append( batchnorm(tconv(layers[-1], self.gen_fn[l], self.gen_ks[l],self.gen_W[l], nonlinearity=relu),gamma=self.gen_g[l],beta=self.gen_b[l]) )
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
        psgan        = PSGAN()
    else:
        psgan        = PSGAN(name='models/' + c.load_name)
        
    c.print_info()

    ##
    # sample used just for visualisation
    z_sample        = sample_noise_tensor(1,c.zx_sample,c.zx_sample_quilt)
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

            # random samples for training
            Znp = sample_noise_tensor(c.batch_size,c.zx) 

            if tot_iter % (c.k+1) == 0:
                cost = psgan.train_g(Znp)
                Gcost.append(cost)
            else:
                cost = psgan.train_d(samples,Znp)
                Dcost.append(cost)

        print "Gcost=", np.mean(Gcost), "  Dcost=", np.mean(Dcost)

        slist = []
        for img in samples:
            slist +=[img]
        img = np.concatenate(slist,axis=2)        
        save_tensor(img, 'samples/minibatchTrue_%s_epoch%d.jpg' % (c.save_name,epoch))

        samples =  psgan.generate(Znp)
        slist = []
        for img in samples:
            slist +=[img]
        img = np.concatenate(slist,axis=2)        
        save_tensor(img, 'samples/minibatchGen_%s_epoch%d.jpg' % (c.save_name,epoch))

        data = psgan.generate(z_sample)

        save_tensor(data[0], 'samples/largesample%s_epoch%d.jpg' % (c.save_name,epoch))
        psgan.save('models/%s_epoch%d.sgan'%(c.save_name,epoch))


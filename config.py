import os
from tools import create_dir
from data_io import get_texture_iter

create_dir('samples')               # create, if necessary, for the output samples 
create_dir('models') 
home        = os.path.expanduser("~")


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1)*2**depth + 1


class Config(object):
    '''
    wraps all configuration parameters in 'static' variables -- these are not serialized!!
    '''
    
    ##optimization constants
    lr          = 0.0002                # learning rate of adam
    b1          = 0.5                   # momentum term of adam
    l2_fac      = 1e-8                  # L2 weight regularization factor
    epoch_count = 100                    #how many epochs to do globally    
    k           = 1                     # number of D updates vs G updates
    batch_size  = 25
    epoch_iters =batch_size * 1000      #steps inside one epoch 
                 
    ##constructor to define serializable variables, serialized when dumping model
    def __init__(self):    
        ##
        # sampling parameters    
        self.nz_local = 30    
        self.nz_global = 60                 # num of global Z dimensions
        self.nz_periodic = 3                    # num of global Z dimensions
        self.nz_periodic_MLPnodes = 50          # the MLP gate for the neural network
        self.nz          = self.nz_local+self.nz_global+self.nz_periodic*2                   # num of dim for Z at each field position, sum of local, global, periodic dimensions
        self.periodic_affine = False            # if True planar waves sum x,y sinusoids, else axes aligned sinusoids x or y 
        self.zx          = 6                    # number of spatial dimensions in Z
        self.zx_sample   = 32                   # size of the spatial dimension in Z for producing the samples    
        self.zx_sample_quilt = self.zx_sample/4      # how many tiles in the global dimension quilt for output sampling

        ##
        # network parameters
        self.nc          = 3                     # number of channels in input X (i.e. r,g,b)
        self.gen_ks      = ([(5,5)] * 5)[::-1]   # kernel sizes on each layer - should be odd numbers for zero-padding stuff
        self.dis_ks      = [(5,5)] * 5           # kernel sizes on each layer - should be odd numbers for zero-padding stuff
        self.gen_ls      = len(self.gen_ks)           # num of layers in the generative network
        self.dis_ls      = len(self.dis_ks)           # num of layers in the discriminative network
        self.gen_fn      = [self.nc]+[2**(n+6) for n in range(self.gen_ls-1)]  # generative number of filters
        self.gen_fn      = self.gen_fn[::-1]
        self.dis_fn      = [2**(n+6) for n in range(self.dis_ls-1)]+[1]   # discriminative number of filters        
        self.npx         = zx_to_npx(self.zx, self.gen_ls) # num of pixels width/height of images in X
        
        ##input texture folder
        self.sub_name    = "honey"#'hex1'#
        self.texture_dir = home + "/DILOG/dcgan_code-master/texture_gan/%s/" % self.sub_name
        self.save_name   = self.sub_name+ "_filters%d_npx%d_%dgL_%ddL_%dGlobal_%dPeriodic_%sAffine_%dLocal" % (self.dis_fn[0],self.npx,self.gen_ls, self.dis_ls,self.nz_global,self.nz_periodic,self.periodic_affine ,self.nz_local)
        self.load_name   = None                  # if None, initializing network from scratch
           
    ## gives back the correct data iterator given class variables -- this way we avoid the python restriction not to pickle iterator objects
    def data_iter(self):
        return get_texture_iter(self.texture_dir, npx=self.npx, mirror=False, batch_size=self.batch_size)

    def print_info(self):
        ##
        # output some information
        print "Learning and generating samples from zx ", self.zx, ", which yields images of size npx ", zx_to_npx(self.zx, self.gen_ls) 
        print "Producing samples from zx_sample ", self.zx_sample, ", which yields images of size npx ", zx_to_npx(self.zx_sample, self.gen_ls) 
        print "Saving samples and model data to file ",self.save_name


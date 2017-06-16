import os
from tools import create_dir
from data_io import get_texture_iter


create_dir('samples')               # create, if necessary, for the output samples 
create_dir('models') 


def zx_to_npx(zx, depth):
    '''
    calculates the size of the output image given a stack of 'same' padded
    convolutional layers with size depth, and the size of the input field zx
    '''
    # note: in theano we'd have zx*2**depth
    return (zx - 1)*2**depth + 1


class Config(object):
    '''
    wraps all configuration parameters in 'static' variables
    '''
    
    ##
    # sampling parameters    
    nz_local = 30    
    nz_global = 60                      # num of global Z dimensions
    nz_periodic = 3                    # num of global Z dimensions
    nz_periodic_MLPnodes = 50          # the MLP gate for the neural network
    nz          = nz_local+nz_global+nz_periodic*2                   # num of dim for Z at each field position, sum of local, global, periodic dimensions
    periodic_affine = False            # if True planar waves sum x,y sinusoids, else axes aligned sinusoids x or y 
    zx          = 6                    # number of spatial dimensions in Z
    zx_sample   = 32                   # size of the spatial dimension in Z for producing the samples    
    zx_sample_quilt = zx_sample/4      # how many tiles in the global dimension quilt for output sampling

    ##
    # network parameters
    nc          = 3                     # number of channels in input X (i.e. r,g,b)
    gen_ks      = ([(5,5)] * 5)[::-1]   # kernel sizes on each layer - should be odd numbers for zero-padding stuff
    dis_ks      = [(5,5)] * 5           # kernel sizes on each layer - should be odd numbers for zero-padding stuff
    gen_ls      = len(gen_ks)           # num of layers in the generative network
    dis_ls      = len(dis_ks)           # num of layers in the discriminative network
    gen_fn      = [nc]+[2**(n+6) for n in range(gen_ls-1)]  # generative number of filters
    gen_fn      = gen_fn[::-1]
    dis_fn      = [2**(n+6) for n in range(dis_ls-1)]+[1]   # discriminative number of filters

    lr          = 0.0002                # learning rate of adam
    b1          = 0.5                   # momentum term of adam
    l2_fac      = 1e-8                  # L2 weight regularization factor

    batch_size  = 25

    epoch_iters = batch_size * 1000      #steps inside one epoch
    epoch_count = 100                    #how many epochs to do globally    

    k           = 1                     # number of D updates vs G updates

    npx         = zx_to_npx(zx, gen_ls) # num of pixels width/height of images in X

    ##
    # data input folder
    sub_name    = "honey"#'hex1'#
    home        = os.path.expanduser("~")
    texture_dir = home + "/DILOG/dcgan_code-master/texture_gan/%s/" % sub_name
    data_iter   = get_texture_iter(texture_dir, npx=npx, mirror=False, batch_size=batch_size)

    save_name   = sub_name+ "_filters%d_npx%d_%dgL_%ddL_%dGlobal_%dPeriodic_%sAffine_%dLocal" % (dis_fn[0],npx,gen_ls, dis_ls,nz_global,nz_periodic,periodic_affine ,nz_local)

    load_name   = None                  # if None, initializing network from scratch
    # load_name   = "efros_filters64_npx257_5gL_5dL_epoch1.sgan"


    @classmethod
    def print_info(cls):
        ##
        # output some information
        print "Learning and generating samples from zx ", cls.zx, ", which yields images of size npx ", zx_to_npx(cls.zx, cls.gen_ls) 
        print "Producing samples from zx_sample ", cls.zx_sample, ", which yields images of size npx ", zx_to_npx(cls.zx_sample, cls.gen_ls) 
        print "Saving samples and model data to file ", cls.save_name


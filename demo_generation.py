from sgan import SGAN
import sys
import numpy as np
from data_io import save_tensor

if len(sys.argv) <=1:
    print "please give model filename"
    raise Exception('no filename specified')

name = sys.argv[1]
print "using stored model",name

##sample a periodically tiling texture
def mosaic_tile(sgan,NZ1=12,NZ2=12, repeat=(2,3)):
    ovp = 2  # how many z values should we keep for overlap, for 5 layer architecture and (5,5) kernels 2 is enough
    tot_subsample= 2**sgan.gen_depth   
    print "NZ1 NZ2 for tilable texture: ", NZ1, NZ2

    sample_zmb = np.random.uniform(-1.,1., (1, sgan.config.nz, NZ1,NZ2) )
    sample_zmb[:, :, :, -ovp * 2:] = sample_zmb[:, :, :, :ovp * 2]
    sample_zmb[:, :, -ovp * 2:, :] = sample_zmb[:, :, :ovp * 2, :]
    samples = sgan.generate(sample_zmb)
    
    #measure the optimal offset of pixels we crop from the edge of the tile: this should have loss of 0 if the tile is perfectly periodical
    ##note: for Theano code we had a nice analytical formula for the optimal offset
    ##note: for the Lasagna code we calculate this empirically since the conv. arithmetic is not well documented and varies depending on NZ1 and NZ2
    ## calc. the pixel discrepancy btw the left and right column, and top and bottom row, when cropping crop1 and crop2 pixels
    def offsetLoss(crop1,crop2):
        return np.abs(samples[:, :, :, crop1] - samples[:, :, :, -crop2]).mean()+ np.abs(samples[:, :, crop1] - samples[:, :, -crop2]).mean()    
    best=1e6
    crop1=0
    crop2=0
    for i in range(ovp*tot_subsample/2,ovp*tot_subsample):
        for j in range(ovp*tot_subsample/2,ovp*tot_subsample):
            loss = offsetLoss(i,j)
            if loss < best:
                best=loss
                crop1=i
                crop2=j

    print "optimal offsets",crop1,crop2,"offset edge errors",best   
    samples = samples[:, :, crop1:-crop2, crop1:-crop2]
    s = (samples.shape[2],samples.shape[3])   
    print "tile sample size", samples.shape
    save_tensor(samples[0],"samples/TILE_%s_%s.jpg" % (name.replace('/','_'), s))
 
    if repeat is not None:
        sbig = np.zeros((3,repeat[0] * s[0], repeat[1] * s[1]))
        for i in range(repeat[0]):
            for j in range(repeat[1]):
                sbig[:,i * s[0]:(i + 1) * s[0], j * s[1]:(j + 1) * s[1]] = samples[0]
        save_tensor(sbig,"samples/TILE_%s_%s_%s.jpg" % (name.replace('/','_'), s, repeat))
    return

#sample a random texture
def sample_texture(sgan,NZ1=60,NZ2=60):    
    z_sample        = np.random.uniform(-1.,1., (1, c.nz, NZ1,NZ2) )
    data = sgan.generate(z_sample)
    save_tensor(data[0], 'samples/stored_%s.jpg' % (name.replace('/','_')))

sgan        = SGAN(name=name)
c=sgan.config
print "nz",c.nz
print "G values",c.gen_fn,c.gen_ks
print "D values",c.dis_fn, c.dis_ks

sample_texture(sgan)
mosaic_tile(sgan)


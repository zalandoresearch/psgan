from sgan import SGAN
import sys
import numpy as np
from data_io import save_tensor

if len(sys.argv) <=1:
    print "please give model filename"
    raise Exception('no filename specified')

name = sys.argv[1]
print "using stored model",name

sgan        = SGAN(name=name)
c=sgan.config
print "nz",c.nz
print "G values",c.gen_fn,c.gen_ks
print "D values",c.dis_fn, c.dis_ks
z_sample        = np.random.uniform(-1.,1., (1, c.nz, 50,50) )
data = sgan.generate(z_sample)
save_tensor(data[0], 'samples/stored_%s.jpg' % (name.replace('/','_')))


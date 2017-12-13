# Periodic Spatial Generative Adversarial Networks

This code implements Periodic Spatial Generative Adversarial Networks (PSGANs) on top of Lasagne/Theano.
The code was tested on top of Lasagne (version 0.2.dev1) and Theano (0.9.0dev2).

### Very Brief Model Description
PSGANs can generate sample textures of arbitrary size that look strikingly similar - but not exactly the same - compared to a single (or several) source image(s).
- PSGANs can be thought of as a convolutional roll-out of [Radford et al.'s](https://github.com/Newmu/dcgan_code) deep convolutional generative adversarial networks for texture synthesis
- the fully convolutional nature allows for real-time generation of high resolution images
- the method can fuse multiple source images and is highly scalable w.r.t. to output texture size and allows for generation of tiled textures
- can learn accurately periodic textures
- can learn whole texture manifolds, allowing sampling and morphing between the training textures

Please take a look at our paper for detailed description of PSGANs  [https://arxiv.org/abs/1705.06566](https://arxiv.org/abs/1705.06566) .

### Examples
**NEW**: better use of inference mode in the Ipython Notebook tutorial using pretrained statistics. In addition, a beatiful-looking pretrained model of DTD Scaly category is checked-in.

![](samples/largesamplescaly_filters69_npx161_5gL_5dL_20Global_10Periodic_TrueAffine_50Local_epoch43.jpg)

**NEW**: You can follow a **step-by-step tutorial** on inference and training PSGAN in the newly checked-in Ipython Notebook **PSGAN tutorial for sampling and training**.

You can generate samples from a stored model. E.g. you can use the checked-in model:
```
python demo_generation.py models/hex1_filters64_npx161_5gL_5dL_0Global_3Periodic_FalseAffine_30Local_epoch18.psgan
```
This model was trained on a single texture from the DTD Textures dataset, category Honeycomb https://www.robots.ox.ac.uk/~vgg/data/dtd/.
Example image:  

![](samples/stored_models_hex1_filters64_npx161_5gL_5dL_0Global_3Periodic_FalseAffine_30Local_epoch18.psgan.jpg)

We have also provided an example of learning a whole texture manifold from the whole DTD Honeycomb category

```
python demo_generation.py models/honey_filters64_npx161_5gL_5dL_60Global_3Periodic_FalseAffine_30Local_epoch100.psgan
```

![](samples/stored_models_honey_filters64_npx161_5gL_5dL_60Global_3Periodic_FalseAffine_30Local_epoch100.psgan.jpg)


### Training the Model
To train the model on new images, edit the config.py file and set the texture_dir variable to a folder containing the image(s) you want to learn from. You might also want to change other parameters of the network there (e.g. depth, number of filters per layer etc...). Then run
```
python psgan.py
```
to train the model. Snapshots of the model will be stored in the subfolder models/ and samples after each epoch will be stored in samples/

## License
Please make sure to cite/acknowledge our paper, if you use any of the contained code in your own projects or publication.


The MIT License (MIT)

Copyright © 2017 Zalando SE, https://tech.zalando.com

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

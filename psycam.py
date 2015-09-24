import argparse, os, sys

from random import randint

# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

import caffe

import os

# If your GPU supports CUDA and Caffe was built with CUDA support,
# uncomment the following to run Caffe operations on the GPU.
# caffe.set_mode_gpu()
# caffe.set_device(0) # select GPU device if multiple devices exist

def output_path():
    """ Create an output filename: look into folder dreams,
        return lowest INTEGER.jpg with leading zeros, e.g. 00020.jpg """
    # faster with sort

    index=0
    output_file = "dreams/%06d.jpg"%index

    while os.path.exists(output_file):
        index += 1
        output_file = "dreams/%06d.jpg"%index

    return output_file

def create_net(model_file):
    net_fn = os.path.join(os.path.split(model_file)[0], 'deploy.prototxt')
    param_fn = model_file

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True
    open('tmp.prototxt', 'w').write(str(model))

    net = caffe.Classifier('tmp.prototxt', param_fn,
                           mean = np.float32([104.0, 116.0, 122.0]), # ImageNet mean, training set dependent
                           channel_swap = (2,1,0)) # the reference model has channels in BGR order instead of RGB
    return net

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(net, img):
    return np.float32(np.rollaxis(img, 2)[::-1]) - net.transformer.mean['data']

def deprocess(net, img):
    return np.dstack((img + net.transformer.mean['data'])[::-1])

# regular, non-guided objective
def objective_L2(dst):
    dst.diff[:] = dst.data 


class Dreamer(object):
    def __init__(self, net, source_path, iterations, end, guide_path, octaves):
        self.img = np.float32(PIL.Image.open(source_path))
        self.net = net
        self.iterations = iterations
        self.objective = objective_L2
        self.octave_n = octaves

        self.end = end

        if guide_path:
            self.guide_features = self.create_guide(guide_path)
            self.objective = self.objective_guide

    # make this a product of a generator function with guide_features in its scope
    def objective_guide(self, dst):
        x = dst.data[0].copy()
        # guide_features is global here
        y = self.guide_features
        ch = x.shape[0]
        x = x.reshape(ch,-1)
        y = y.reshape(ch,-1)
        A = x.T.dot(y) # compute the matrix of dot-products with guide features
        dst.diff[0].reshape(ch,-1)[:] = y[:,A.argmax(1)] # select ones that match best

    # use self.end instead of end
    def create_guide(self, guide_path):
        guide = np.float32(PIL.Image.open(guide_path))
        h, w = guide.shape[:2]
        src, dst = net.blobs['data'], net.blobs[self.end]
        src.reshape(1,3,h,w)
        src.data[0] = preprocess(net, guide)
        self.net.forward(end=self.end)

        return dst.data[0].copy()

    def iterated_dream(self):
        self.net.blobs.keys()

        frame = self.img

        h, w = frame.shape[:2]
        s = 0.05 # scale coefficient

        for i in xrange(self.iterations):
            if self.end:
                frame = self.deepdream(frame, end=self.end, octave_n=self.octave_n)
            else:            
                frame = self.deepdream(frame, octave_n=self.octave_n)

            PIL.Image.fromarray(np.uint8(frame)).save(output_path())
            frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)

    def make_step(self, step_size=1.5, end='inception_4c/output', 
                  jitter=32, clip=True):
        """Basic gradient ascent step."""

        src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = self.net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
                
        self.net.forward(end=end)
        self.objective(dst)  # specify the optimization objective
        self.net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                
        if clip:
            bias = self.net.transformer.mean['data']
            src.data[:] = np.clip(src.data, -bias, 255-bias)  

    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
                  end='inception_4c/output', clip=True, **step_params):
        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.make_step(end=end, clip=clip, **step_params)
                
                # visualization
                vis = deprocess(self.net, src.data[0])
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                # is octave, i the depth?
                print octave, i, end, vis.shape
                clear_output(wait=True)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])


def parse_arguments(sysargs):
    """ Setup the command line options. """

    description = ''' deepdreaming.py is a handler script to simplify the usage
        of Googles DeepDream algorithm. The DeepDream algorithm takes an image
        as input and runs an overexpressed pattern recognition in form of
        a convolutional neural network over it. 
        See the original Googleresearch blog post
        http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html
        for more information or follow this
        http://www.knight-of-pi.org/installing-the-google-deepdream-software/
        tutorial for installing DeepDream on Ubuntu.
        Try guided dreams with the options -g FILE and -d 2 or shallow dreams
        with the options -d 2 -t 5.'''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-s', '--source', nargs='?', const='sky_1024.jpg', 
                                     default='sky_1024.jpg', help='input filename')
    parser.add_argument('-g', '--guide', nargs='?', default=None,
                                    help='Target for guided dreams')
    parser.add_argument('-i', '--iterations', nargs='?', type=int, const=1,
                                     default=1, help='Number of iterations')
    parser.add_argument('-d', '--depth', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 10),  const=5, default=5,
                                    help='Depth of the dream as an value between 1 and 10')
    parser.add_argument('-t', '--type', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 10),
                                    const=4, default=4, help='Layer type as an value between 1 and 6')
    parser.add_argument('-o', '--octaves', nargs='?', metavar='int', type=int,
                                         choices=xrange(1, 12),
                                         const=5, default=5, 
                                         help='The number of scales the algorithm is applied to')
    parser.add_argument('-r', '--random', action='store_true', 
                                         help='Overwrite depth, layer type and octave with random values ')
    #parser.add_argument('-m', '--model', nargs='?', metavar='int', type=int,
    #                                choices=xrange(1, 6), help='model 1..5',
    #                                const=1, default=1)

    return parser.parse_args(sysargs)

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    models_base = '../caffe/models'
    net = create_net(os.path.join(models_base, 'bvlc_googlenet/bvlc_googlenet.caffemodel'))

    # model selection turned off since i found no way of producing good
    # dreams with different models
    '''
    models = ('bvlc_googlenet/bvlc_googlenet.caffemodel',
                    'bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    'bvlc_reference_rcnn_ilsvrc13/bvlc_reference_rcnn_ilsvrc13.caffemodel',
                    'finetune_flickr_style/finetune_flickr_style.caffemodel',
                    'bvlc_alexnet/bvlc_alexnet.caffemodel')
    
    net = create_net(os.path.join(models_base, models[args.model-1]))
    '''

    numbering = ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']
    layer_types = ['1x1', '3x3', '5x5', 'output', '5x5_reduce', '3x3_reduce']

    layer = 'inception_' + numbering[args.depth-1] + '/' + layer_types[args.type-1]
    octaves = args.octaves

    # overwrite octaves and layer with random values
    if args.random == True:
        octaves = randint(1, 11)
        layer = ('inception_' + numbering[randint(0, len(numbering)-1)] + '/' +
                      layer_types[randint(0, len(layer_types)-1)])
    
    dreamer = Dreamer(net=net, source_path=args.source, 
                                   iterations=args.iterations, end=layer, 
                                   guide_path=args.guide, octaves=octaves)
    dreamer.iterated_dream()
    
    



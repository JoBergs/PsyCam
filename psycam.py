#!/usr/bin/python
#encoding:utf-8

import os, sys

from random import randint

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

# disable logging before net creation
os.environ["GLOG_minloglevel"] = "2"
import caffe

from utils import get_layer_descriptor, get_source_image, parse_arguments


def create_net(model_file):
    """ Create the neural network tmp.prototxt. """

    net_fn = os.path.join(os.path.split(model_file)[0], 'deploy.prototxt')

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True

    open('tmp.prototxt', 'w').write(str(model))


def load_net(model_file):
    """ Load the neural network tmp.prototxt.  """

    net = caffe.Classifier('tmp.prototxt', model_file,
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


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))


class PsyCam(object):
    def __init__(self, net):        
        self.net = net

    def iterated_dream(self, source_path, end, octaves):
        # import ipdb
        # ipdb.set_trace()

        frame = np.float32(PIL.Image.open(source_path))
        self.octave_n = octaves

        frame = self.deepdream(frame, end=end, octave_n=self.octave_n)

        dream_path = source_path.replace('.jpg', '_dream.jpg')

        PIL.Image.fromarray(np.uint8(frame)).save(dream_path)

    def make_step(self, step_size=1.5, end='inception_4c/output', jitter=32):
        """Basic gradient ascent step."""

        src = self.net.blobs['data'] # input image is stored in Net's 'data' blob
        dst = self.net.blobs[end]

        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        src.data[0] = np.roll(np.roll(src.data[0], ox, -1), oy, -2) # apply jitter shift
                
        self.net.forward(end=end)
        objective_L2(dst)  # specify the optimization objective
        self.net.backward(start=end)
        g = src.diff[0]
        # apply normalized ascent step to the input image
        src.data[:] += step_size/np.abs(g).mean() * g

        src.data[0] = np.roll(np.roll(src.data[0], -ox, -1), -oy, -2) # unshift image
                
        bias = self.net.transformer.mean['data']
        src.data[:] = np.clip(src.data, -bias, 255-bias)  

    def deepdream(self, base_img, iter_n=10, octave_n=4, octave_scale=1.4, 
                              end='inception_4c/output', **step_params):

        # prepare base images for all octaves
        octaves = [preprocess(self.net, base_img)]
        for i in xrange(octave_n-1):
            octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
        
        src = self.net.blobs['data']
        detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details

        #print(list(enumerate(octaves[::-1])))

        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            if octave > 0:
                # upscale details from the previous octave
                h1, w1 = detail.shape[-2:]
                detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

            src.reshape(1,3,h,w) # resize the network's input image size
            src.data[0] = octave_base+detail
            for i in xrange(iter_n):
                self.make_step(end=end, **step_params)
                vis = deprocess(self.net, src.data[0])  # visualization
                showarray(vis)               
                print(octave, i, end, vis.shape)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])


def start_dream(args):
    """ Gather all parameters (source image, layer descriptor and octave),
    create a net and start to dream. """

    source_path = get_source_image(args)
    layer = get_layer_descriptor(args)
    octave = (args.octaves if args.octaves else randint(1, 9))

    model_file = '../caffe/models/bvlc_googlenet/bvlc_googlenet.caffemodel'

    if args.network:
        create_net(model_file)
    net = load_net(model_file)

    psycam = PsyCam(net=net)
    psycam.iterated_dream(source_path=source_path, 
                                             end=layer, octaves=octave)    

if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        while True:
            start_dream(args)

            if not args.continually:
                break

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print('Quitting PsyCam')

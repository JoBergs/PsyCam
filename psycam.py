import argparse, os, sys, time

from random import randint

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

import caffe

def get_output_path(base_dir):
    """ Create an output filename: look into folder dreams,
        return lowest INTEGER.jpg with leading zeros, e.g. 00020.jpg """
    # faster with sort

    index=0
    output_file = os.path.join(base_dir, "%06d.jpg"%index)

    while os.path.exists(output_file):
        index += 1
        output_file = os.path.join(base_dir, "%06d.jpg"%index)

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


class PsyCam(object):
    def __init__(self, net):        
        self.net = net

    def iterated_dream(self, source_path, end, octaves):
        self.img = np.float32(PIL.Image.open(source_path))
        self.objective = objective_L2
        self.octave_n = octaves
        self.end = end
        print 'in iterated dream'
        # OFF to hide net displaying(doesn't work...)
        #self.net.blobs.keys()

        frame = self.img

        h, w = frame.shape[:2]
        s = 0.05 # scale coefficient

        if self.end:
            frame = self.deepdream(frame, end=self.end, octave_n=self.octave_n)
        else:            
            frame = self.deepdream(frame, octave_n=self.octave_n)

        PIL.Image.fromarray(np.uint8(frame)).save(get_output_path('dreams/'))
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
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])


def parse_arguments(sysargs):
    """ Setup the command line options. """

    description = '''psycam.py is a psycedelic surveilance camera using
        Googles DeepDream algorithm. The DeepDream algorithm takes an image
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
    parser.add_argument('-d', '--depth', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 5),  const=5, default=5,
                                    help='Depth of the dream as an value between 1 and 10')
    parser.add_argument('-t', '--type', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 6),
                                    const=4, default=4, help='Layer type as an value between 1 and 6')
    parser.add_argument('-o', '--octaves', nargs='?', metavar='int', type=int,
                                         choices=xrange(1, 10),
                                         const=5, default=5, 
                                         help='The number of scales the algorithm is applied to')
    parser.add_argument('-r', '--random', action='store_true', 
                                         help='Overwrite depth, layer type and octave with random values')

    parser.add_argument('-s', '--snapshot', action='store_true', 
                                         help='Make a single snapshot instead of running permanently')

    return parser.parse_args(sysargs)


def make_snapshot():    
    import picamera
    camera = picamera.PiCamera()
    camera.resolution = (500, 280)
    source_path = get_output_path('snapshots')
    camera.capture(source_path)
    print 'snapshot'
    camera.close()
    del camera
    del picamera
    return source_path

def clean_directory():
    os.remove("dreams/*.jpg")
    os.remove("snapshots/*.jpg")

if __name__ == "__main__":

    args = parse_arguments(sys.argv[1:])

    models_base = '../caffe/models'
    net = create_net(os.path.join(models_base, 'bvlc_googlenet/bvlc_googlenet.caffemodel'))

    numbering = ['3a', '3b', '4a', '4b', '4c']
    layer_types = ['1x1', '3x3', '5x5', 'output', '5x5_reduce', '3x3_reduce']

    l_index = args.depth - 1
    l_type = args.type - 1
    octave = args.octaves

    # delete modules that are no more required
    # update readme ect.
    # why does the net factory for other models crash?
    # could the RAM be cleaned, garbage collected or s.t. after every run?
    # create net only when required;

    # BACKUP old snapshots and dirs
    # just create new dir named date
    #   (if exists, store it with daytime incl minutes

    clean_directory()

    psycam = PsyCam(net=net)

    try:
        while True:
            source_path = make_snapshot()

            # overwrite octaves and layer with random values
            if args.random == True:
                octave = randint(1, 9)
                l_index = randint(0, len(numbering)-1)
                l_type = randint(0, len(layer_types)-1)
               
            layer = 'inception_' + numbering[l_index] + '/' + layer_types[l_type]

            psycam.iterated_dream(source_path=source_path, 
                                                    end=layer, octaves=octave)
            time.sleep(1)

            if args.snapshot == True:
                break

    except Exception, e:
        import traceback
        print traceback.format_exc()
        print 'Quitting PsyCam'


    

    
    



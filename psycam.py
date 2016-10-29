import argparse, datetime, os, sys, time, shutil, subprocess

from random import randint

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from google.protobuf import text_format

# disable logging before net creation
os.environ["GLOG_minloglevel"] = "2"
import caffe


def make_snapshot(size=[500, 280]):    
    import picamera

    # prolly resolution can be passed as size
    camera = picamera.PiCamera(resolution=size)

    source_path = add_timestamp('./dreams/photo.jpg')

    camera.capture(source_path)
    camera.close()
    del camera
    del picamera
    return source_path


def add_timestamp(path):
    now = datetime.datetime.now().ctime()
    timestamp = '_'.join(now.split()[:-1])
    stamped_path = path.replace('.jpg', '_' + timestamp + '.jpg')

    return stamped_path


def detect_rpi():
    try:
        with open('/proc/cpuinfo', 'r') as f:
            if 'BCM2709' in f.read():
                return True
    except:
        pass

    return False


def create_net(model_file):
    net_fn = os.path.join(os.path.split(model_file)[0], 'deploy.prototxt')
    param_fn = model_file

    # Patching model to be able to compute gradients.
    # Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
    model = caffe.io.caffe_pb2.NetParameter()
    text_format.Merge(open(net_fn).read(), model)
    model.force_backward = True


    # ONLY DO THIS WHEN THE FILE DOES NOT EXIST! TEST THAT
    open('tmp.prototxt', 'w').write(str(model))

    # probably mean needs to be CHANGED FOR OTHER NETS
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

        frame = self.img

        h, w = frame.shape[:2]
        s = 0.05 # scale coefficient

        if self.end:
            frame = self.deepdream(frame, end=self.end, octave_n=self.octave_n)
        else:            
            frame = self.deepdream(frame, octave_n=self.octave_n)

        dream_path = source_path.replace('.jpg', '_dream.jpg')

        PIL.Image.fromarray(np.uint8(frame)).save(dream_path)
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
                print(octave, i, end, vis.shape)
                
            # extract details produced on the current octave
            detail = src.data[0]-octave_base
        # returning the resulting image
        return deprocess(self.net, src.data[0])


def start_dream(args, source_path):
    models_base = '../caffe/models'
    net = create_net(os.path.join(models_base, 'bvlc_googlenet/bvlc_googlenet.caffemodel'))

    layer_depths = ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']
    layer_types = ['1x1', '3x3', '5x5', 'output', '5x5_reduce', '3x3_reduce']

    # move all of this in function preprocessing or something
    octave = randint(1, 9)
    l_depth = randint(0, len(layer_depths)-1)
    l_type = randint(0, len(layer_types)-1)

    if args.depth:
        l_depth = args.depth - 1
    if args.type:
        l_type = args.type - 1
    if args.octaves:
        octave = args.octaves

    # when running DeepDream on the RPi, restrict layer depth to 5 = '4d':
    # higher values crash the RPi
    if detect_rpi:
        l_depth = min(l_depth, 5)

    psycam = PsyCam(net=net)
           
    layer = 'inception_' + layer_depths[l_depth] + '/' + layer_types[l_type]

    print('Image: ',  source_path, 'Layer: ', layer, 'Octave: ', octave)

    psycam.iterated_dream(source_path=source_path, 
                                             end=layer, octaves=octave)



def parse_arguments(sysargs):
    """ Setup the command line options. """

    description = '''PsyCam is a psycedelic surveilance camera using the
        Google DeepDream algorithm. The DeepDream algorithm takes an image
        as input and runs an overexpressed pattern recognition in form of
        a convolutional neural network over it. 
        See the original Googleresearch blog post
        http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html
        for more information or follow this
        http://www.knight-of-pi.org/psycam-a-raspberry-pi-deepdream-surveilance-camera/
        tutorial for installing PsyCam on a Raspberry Pi.
        Try random surveilance with python psycam.py -r.
        For using this script on an ubuntu system (non-rpi) without camera, 
        give the parameter -i and the dream source file (*.jpg).'''

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-d', '--depth', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 10), 
                                    help='Depth of the dream as an value between 1 and 10')
    parser.add_argument('-t', '--type', nargs='?', metavar='int', type=int,
                                    choices=xrange(1, 6),
                                    help='Layer type as an value between 1 and 6')
    parser.add_argument('-o', '--octaves', nargs='?', metavar='int', type=int,
                                         choices=xrange(1, 12),
                                         help='The number of scales the algorithm is applied to')
    parser.add_argument('-c', '--continually', action='store_true', 
                                         help='Run psycam in an endless loop')
    parser.add_argument('-i', '--input', nargs='?', metavar='path', type=str,
                                    help='Use the path passed behind -i as source for the dream')
    parser.add_argument('-s', '--size', nargs=2, type=int, metavar='width height', default=[500, 280],
                                    help='Pass the image size for rpi camera snapshots as x y')
    

    return parser.parse_args(sysargs)

def handle_source_image(args):
    """ Input processing: if a source image is supplied, make a time-stamped
    duplicate;  if no image is supplied, make a snapshot with the given format."""

    if args.input:
        source_path = add_timestamp(args.input)
        shutil.copyfile(args.input, source_path)
    else:
        source_path = make_snapshot(args.size)

if __name__ == "__main__":
    try:
        args = parse_arguments(sys.argv[1:])
        while True:

            source_path = handle_source_image(args)
            start_dream(args, source_path)
            time.sleep(1)

            if not args.continually:
                break

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print('Quitting PsyCam')



    

    
    


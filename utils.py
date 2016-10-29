#encoding:utf-8

# Container for various helper functions for PsyCam based on 
# Google DeepDream.

import argparse, datetime, shutil

from random import randint


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
    parser.add_argument('-n', '--network', action='store_true', 
                                         help='Create neural network from model file')
    parser.add_argument('-i', '--input', nargs='?', metavar='path', type=str,
                                    help='Use the path passed behind -i as source for the dream')
    parser.add_argument('-s', '--size', nargs=2, type=int, metavar='width height', default=[500, 280],
                                    help='Pass the image size for rpi camera snapshots as x y')

    return parser.parse_args(sysargs)


def get_layer_descriptor(args):
    """ Process input arguments into a layer descriptor and return it. If the 
    machine is an RPi, limit the layer depth to '4D'. """

    layer_depths = ['3a', '3b', '4a', '4b', '4c', '4d', '4e', '5a', '5b']
    layer_types = ['1x1', '3x3', '5x5', 'output', '5x5_reduce', '3x3_reduce']

    # if given, take the input parameter; use random value elseway
    l_depth = (args.depth - 1 if args.depth else randint(0, len(layer_depths)-1))
    l_type = (args.type - 1 if args.type else randint(0, len(layer_types)-1))

    # restrict layer depth to 5 = '4d' for the RPi: higher values would crash 
    if detect_rpi:
        l_depth = min(l_depth, 5)    
           
    layer = 'inception_' + layer_depths[l_depth] + '/' + layer_types[l_type]

    print(''.join(['\nLayer: ', layer, '\n']))

    return layer


def get_source_image(args):
    """ Input processing: if a source image is supplied, make a time-stamped
    duplicate;  if no image is supplied, make a snapshot."""

    if args.input:
        source_path = add_timestamp(args.input)
        shutil.copyfile(args.input, source_path)
    else:
        source_path = make_snapshot(args.size)

    print(''.join(['\nBase image for the DeepDream: ', source_path, '\n']))

    return source_path


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
PsyCam
============

Google DeepDream
-----------------------------
The Google DeepDream algorithm is a modified neural network. Instead of
identifying objects in an input image, it changes the image into the direction
of its training data set, which produces impressive surrealistic, dream-like images.
You can find the original GitHub repository at
https://github.com/google/deepdream/blob/master/dream.ipynb

PsyCam
-------------------------
PsyCam is an extension of the DeepDream for the Raspberry Pi. With the RPi
camera module, PsyCam can make a photo and convert it into a DeepDream.

Installation
------------------
Either follow the manual installation instructions at

    http://www.knight-of-pi.org/deepdream-on-the-raspberry-pi-3-with-raspbian-jessie/

or perform the following steps in on a Raspberry Pi with Raspbian Jessie 
installed and the camera module being attached:

    $ mkdir deepdream && cd deepdream
    $ git clone https://github.com/JoBergs/PsyCam
    $ cd PsyCam
    $ sudo python install_tools.py packages
    $ sudo python install_tools.py caffe
    $ sudo python install_tools.py protobuf
    $ sudo python install_tools.py camera
    $ sudo reboot

The installation will take half a day or so.

Usage
-----------------------------------
The script psycam.py is controlled via command-line parameters. They are listed with

    $python psycam.py --help

Start PsyCam with the default _inception4c/output layer and octave 3:

    $python psycam.py

Start PsyCam with randomized layer and octave:

    $python psycam.py -r

Make just a single snapshot/dream:

    $python psycam.py -s

Start PsyCam set layer depth, type and network octave manually:

    $python psycam.py -d 2 -t 1 -o 6

Beware of the Critters!


Output images
--------------------------------

The dreams are stored in

    /home/pi/deepdream/PsyCam/dreams 

with the original photo and tagged with a timestamp.

* [Johannes Bergs](mailto:jo@knight-of-pi.org)

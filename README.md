PsyCam
============

PsyCam is a psychedelic surveilance camera for capturing the paranomal
and metaphysic activities around us; that is, a Raspberry Pi with
an camera module makes a snapshot and prozesses the image with Googles
DeepDream algorithm from Alexander Mordvintsev, Michael Tyka and
Christopher Olah (see https://github.com/google/deepdream).

Installation
-------------------------------

The Installation of DeepDream is quite delicate; see this tutorial
http://www.knight-of-pi.org/deepdreaming-on-a-raspberry-pi-2/
for installing DeepDream on a Raspberry Pi 2.

Autostart
--------------------------------
Open the file ~/.bashrc and add the lines

    cd /home/ubuntu/deepdream/psycam
    python psycam.py

Then, turn off the Rapsberry Pi, Disconnect it from everything but the camera and attach a battery.
The Raspberry Pi will now make Snapshots and convert the to DeepDream images with googles recipe.

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

Output filename
--------------------------------
The output filename will be the higest 5-digit integer number in ./dreams which
is not occupied already, e.g. "00023.jpg". The Snapshot will be stored with the same Format but in
./snaphots. Old dreams and snapshots are stored after starting PsyCam with a timestamp.

* [Johannes Bergs](mailto:jo@knight-of-pi.org)

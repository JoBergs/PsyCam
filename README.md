DeepDream
============

This GitHub repository is an extension of the original GitHub repository
https://github.com/google/deepdream from Alexander Mordvintsev, Michael Tyka and
Christopher Olah from Google. The original Readme is at the end of this file;
the same licence(file LICENCE) applies for this work.

Installation
-------------------------------

The Installation of DeepDream is quite delicate; see this tutorial
http://www.knight-of-pi.org/installing-the-google-deepdream-software/
for installing DeepDream on a Ubuntu 14.04 machine and this tutorial
http://www.knight-of-pi.org/deepdreaming-on-a-raspberry-pi-2/
for installing DeepDream on a Raspberry Pi 2.

Usage
-----------------------------------
The script deepdreaming.py is controlled via command-line parameters. They are listed with

    $python deepdreaming.py --help

The source file is chosen with the parameter -s:

    $python deepdreaming.py -s winter_1024.jpg

Select the number of iterations with the parameter -i:

    $python deepdreaming.py -s winter_1024.jpg -i 3

Be carefull, though, since choosing a large number here freezes the computer for quite some time.

Output filename
--------------------------------
The output filename will be the higest 5-digit integer number in ./dreams which
is not occupied already, e.g. "00023.jpg".

Original README:
--------------------------------

This repository contains IPython Notebook with sample code, complementing 
Google Research [blog post](http://googleresearch.blogspot.ch/2015/06/inceptionism-going-deeper-into-neural.html) about Neural Network art.
See [original gallery](https://photos.google.com/share/AF1QipPX0SCl7OzWilt9LnuQliattX4OUCj_8EP65_cTVnBmS1jnYgsGQAieQUc1VQWdgQ?key=aVBxWjhwSzg2RjJWLWRuVFBBZEN1d205bUdEMnhB) for more examples.

You can view "dream.ipynb" directly on github, or clone the repository, 
install dependencies listed in the notebook and play with code locally.

It'll be interesting to see what imagery people are able to generate using the described technique. If you post images to Google+, Facebook, or Twitter, be sure to tag them with [#deepdream](https://twitter.com/hashtag/deepdream) so other researchers can check them out too.

* [Alexander Mordvintsev](mailto:moralex@google.com)
* [Michael Tyka](https://www.twitter.com/mtyka)
* [Christopher Olah](mailto:colah@google.com)

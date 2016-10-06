import subprocess

from setuptools import setup

from setuptools.command.install import install

from utils import *

# Brute, i know.

# Installation should be
# sudo pip install psycam
# sudo reboot
# cd /home/pi/deepdream/psycam
# python psycam.py -r -s

# BACKUP config files before changing them!

# FUSE THIS with the psycam repo

def do_before():
    print("\n\nInstalling DeepDream. No no he's not dead, he's, he's restin'!\n\n")    

def do_after():
    install_apt_packages()
    install_caffe()
    install_protobuf()
    activate_camera()
    print('\n\nYou made it!\n\n')

class CustomInstall(install):
    def run(self):
        do_before()
        install.run(self)
        do_after()

setup(
    name='deepdream_install',        # Name of the PyPI-package.
    version='0.8',             # Version number, update for new releases
    description='Install Google DeepDream on a Raspberry Pi with Raspbian Jessie',
    author='Johannes Bergs',
    author_email='jo@knight-of-pi.org',
    install_requires=['pyzmq', 'jsonschema', 'pillow', 'numpy', 'scipy', 'ipython', 'jupyter', 'pyyaml'],
    cmdclass={'install': CustomInstall}
)

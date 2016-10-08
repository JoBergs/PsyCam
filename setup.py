import subprocess

from setuptools import setup

from setuptools.command.install import install

from install_tools import install_apt_packages, activate_camera

# NEWER:
#   mkdir deepdream && cd deepdream
#   git clone https://github.com/JoBergs/PsyCam
#   cd PsyCam
#   sudo pip install .
#   sudo python install_tools.py caffe
#   ect.


# NEW: does this work???
# sudo pip install psycam
# cd ~/deepdream
# sudo pip caffe

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
    # OFF FOR TESTING!
    install_apt_packages()

def do_after():  
    activate_camera()
    print('\n\nInstalled dependencies...\n\n')

class CustomInstall(install):
    def run(self):
        do_before()
        install.run(self)
        do_after()

# class InstallCaffe:
#     def run(self):
#         install_caffe()

# class InstallProtobuf:
#     def run(self):
#         install_protobuf()

# class InstallPsyCam:
#     def run(self):
#         install_pyscam()

setup(
    name='PsyCam',
    version='1.9',
    description='Install Google DeepDream on a Raspberry Pi with Raspbian Jessie',
    author='Johannes Bergs',
    #scripts=['install_tools'],
    author_email='jo@knight-of-pi.org',
    # OFF FOR TESTING
    install_requires=['pyzmq', 'jsonschema', 'pillow', 'numpy', 'scipy', 'ipython', 'jupyter', 'pyyaml'],
    url='https://github.com/JoBergs/PsyCam',
    cmdclass={'install': CustomInstall}
    #cmdclass={'install': CustomInstall, 'caffe': InstallCaffe, 'protobuf': InstallProtobuf, 'psycam': InstallPsyCam}
)

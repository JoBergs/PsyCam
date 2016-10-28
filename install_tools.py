import os, shutil, subprocess, sys

HOME = os.path.expanduser("~")


def execute_commands(path, commands, ubuntu=False):
    os.chdir(path)

    for command in commands:
        if ubuntu:
            command = 'sudo ' + command
        subprocess.call([command], shell=True)


########### camera ###########
def activate_camera(ubuntu=False):
    if ubuntu:
        print('Camera not activated for Ubuntu system!')
        return

    with open('/boot/config.txt', "r") as f:
        config = f.read()

    config += "\nstart_x=1\ngpu_mem=128"

    with open(HOME + '/tmp', "w") as f:
        f.write(config)

    subprocess.call(['sudo cp /boot/config.txt /boot/config.txt_BKP'], shell=True)
    subprocess.call(['sudo cp ' + HOME + '/tmp /boot/config.txt'], shell=True)
########### /camera ###########

########### apt ###########
apt_packages = ["gfortran", "cython",
    "libprotobuf-dev", "libleveldb-dev", "libsnappy-dev", "libopencv-dev", "libhdf5-serial-dev", "protobuf-compiler", "git",
    "--no-install-recommends libboost-all-dev",
    "python-dev", "libgflags-dev", "libgoogle-glog-dev", "liblmdb-dev", "libatlas-base-dev", "python-skimage"]

def install_packages():
    subprocess.call(['sudo apt-get update && sudo apt-get -y upgrade'], shell=True)

    # TODO: fuse strings and pass to execute_commands
    for package in apt_packages:
        subprocess.call(['sudo apt-get install -y ' + package], shell=True)

    subprocess.call(['sudo pip install -r requirements.txt'], shell=True)
########### /apt ###########

########### caffe ###########
caffe_commands = ['make all', 'make runtest', 'make pycaffe',
                              'make test', './scripts/download_model_binary.py models/bvlc_googlenet']

caffe_replace = [['# CPU_ONLY := 1', 'CPU_ONLY := 1'],
                         ['/usr/lib/python2.7/dist-packages/numpy/core/include',
                          '/usr/local/lib/python2.7/dist-packages/numpy/core/include'],
                         ['INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include',
                         'INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial/'],
                         ['LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib',
                         'LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/arm-linux-gnueabihf/hdf5/serial/']]

#base_path = '/home/pi'
dd_path = os.path.join(HOME, 'deepdream')
caffe_path = os.path.join(dd_path, 'caffe')
bash_path = os.path.join(HOME, '.bashrc')


def install_caffe(ubuntu=False):

    # abort if caffe is already installed
    if os.path.isdir(caffe_path):
        return

    if not os.path.isdir(dd_path):
        os.mkdir(dd_path)
    os.chdir(dd_path)
    
    subprocess.call(['git clone https://github.com/BVLC/caffe'], shell=True) 
    
    # Create Makefile config
    os.chdir(caffe_path)
    shutil.copyfile('Makefile.config.example', 'Makefile.config')
    with open('Makefile.config', "r") as f:
        makefile = f.read()

    # hdf5 is at a different place when installing for ubuntu
    if ubuntu:
        caffe_replace[-1][-1] = "LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial/"

    for item in caffe_replace:
        makefile = makefile.replace(item[0], item[1])

    with open('Makefile.config', "w") as f:
        f.write(makefile)

    execute_commands('.', caffe_commands, ubuntu)

    with open(bash_path, "a") as f:
        f.write("export PYTHONPATH=" + HOME + "/deepdream/caffe/python:$PYTHONPATH")
########### /caffe ###########
    
########### protobuf ###########
download_protobuf = ['wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz', 
                                 'tar xf protobuf-2.6.1.tar.gz']
make_protobuf = ['sudo ./configure', 'sudo make', 'sudo make install', '. ~/.bashrc']
python_protobuf = ['python setup.py build', 'python setup.py google_test', 'sudo python setup.py install']

protobuf_path = os.path.join(dd_path, 'protobuf-2.6.1')
protobuf_python_path = os.path.join(protobuf_path, 'python')

def install_protobuf(ubuntu=False):
    # abort if protobuf is already installed
    if os.path.isdir(protobuf_path):
        return

    os.chdir(dd_path)

    with open(bash_path, "a") as f:
        f.write("export LD_LIBRARY_PATH=/usr/local/lib")

    subprocess.call(['cd protobuf-2.6.1'], shell=True)

    execute_commands('.', download_protobuf, ubuntu)
    execute_commands(protobuf_path, make_protobuf, ubuntu)
    execute_commands(protobuf_python_path, python_protobuf, ubuntu)
########### /protobuf ###########

if __name__ == "__main__":

    # flag for installing on a host computer with ubuntu or 
    # on the RPi with Raspbian
    ubuntu = False

    if len(sys.argv) >= 2:
        if len(sys.argv) == 3 and sys.argv[2] in ['-u', '--ubuntu']:
            ubuntu = True
        if sys.argv[1] == "packages":
            install_packages()  
        elif sys.argv[1] == "caffe":
            install_caffe(ubuntu)        
        elif sys.argv[1] == "protobuf":
            install_protobuf(ubuntu)        
        elif sys.argv[1] == "camera":
            activate_camera(ubuntu)

    else:
        print("""What should be installed?\nPass 'packages', 'caffe', 'protobuf'
                    or 'camera' as command line argument.
                    Also, add a '-u' for installing DeepDream on Ubuntu.""")
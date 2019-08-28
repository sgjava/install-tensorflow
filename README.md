![Title](images/title.png)

If you are interested in compiling the latest version of TensorFlow for x86_64 computers then this project will show you how. You should be experienced with Linux, TensorFlow and Python (or Java or C++) to make the most of this project. Latest Ubuntu 18.04 and probably other similar distributions will work. This is for a basic CPU based build. If you want to use GPU and/or CPU optimizations then you can tackle that by searching online. I needed a build for VMs and my old desktop CPU that doesn't have all the latest features. 

I created an Ubuntu 18.04.3 LTS desktop with VirtualBox 6, so I could install Eclipse and other tools to experiment with my fresh TensorFlow build. You could go headless if you choose and adjust memory as needed. I used 4 vCPUs and 8G of vRAM. Build time was around 6+ hours. I checked every few hours because of various connection failures during the build process. Just restart using the same build line from terminal history.

### Install Bazel
* Install dependencies
    * `sudo apt install git pkg-config zip g++ zlib1g-dev unzip python3 python3-dev python3-pip`
* Make Python 3 and pip 3 default
    * `sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1`
    * `sudo update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1`
* Download Bazel (I had to use this version instead of the latest. Build will notify you if Bazel version is incorrect)
    * `wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh`
* Run install script    
    * `chmod a+x bazel-0.26.1-installer-linux-x86_64.sh`
    * `./bazel-0.26.1-installer-linux-x86_64.sh --user`
    * `sudo reboot`
* Run Bazel from terminal.    
    * `bazel`

### Install Tensorflow
* Install dependencies
    * `pip install -U --user pip==9.0.1 six pyyaml h5py numpy==1.16.4 wheel setuptools mock future>=0.17.1`
    * `pip install -U --user keras_applications==1.0.6 --no-deps`
    * `pip install -U --user keras_preprocessing==1.0.5 --no-deps`
* Download source
    * `git clone https://github.com/tensorflow/tensorflow.git`
* Build Tensorflow pip package (I used defaults during configure)
    * `cd tensorflow`
    * `./configure`
    * `bazel build --config=v2 //tensorflow/tools/pip_package:build_pip_package`
* Build the package
    * `./bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`
* Install the package
    * `pip install -U --user /tmp/tensorflow_pkg/tensorflow*.whl`
* Downgrade numpy (see [Tons of warnings just by importing tf (2.0.0-beta1) #31364](https://github.com/tensorflow/tensorflow/issues/31364))
    * `pip uninstall numpy`
    * `pip install --user numpy==1.16.4`

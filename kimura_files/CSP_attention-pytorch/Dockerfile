FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV CPATH=/usr/local/include:$CPATH
ENV CUDA_PATH=/usr/local/cuda
ENV PATH=$CUDA_PATH/bin:$PATH
ENV CPATH=$CUDA_PATH/include:$CPATH
ENV LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:$LD_LIBRARY_PATH
ENV PYTORCH_VERSION=1.2.0

RUN rm -rf /var/lib/apt/lists/*\
			/etc/apt/source.list.d/cuda.list\
			/etc/apt/source.list.d/nvidia-ml.list
			
RUN apt-get update\
  && apt-get -y install build-essential\
 						apt-utils\
						ca-certificates\
						cmake\
						wget\
						git\
						vim

# Python
######################################
RUN apt-get update\
  && apt-get install -y software-properties-common\
  && add-apt-repository ppa:deadsnakes/ppa\
  && apt-get update\
  && apt-get install -y python3.7 python3.7-dev python3-distutils-extra\
  && wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py\
  && python3.7 ~/get-pip.py\
  && ln -s /usr/bin/python3.7 /usr/local/bin/python3\
  && ln -s /usr/bin/python3.7 /usr/local/bin/python\
  && pip install setuptools \
  && pip install numpy scipy pandas cloudpickle scikit-learn matplotlib Cython seaborn


# OpenCV
######################################
RUN apt-get install -y --no-install-recommends libatlas-base-dev\
		libgflags-dev \
		libgoogle-glog-dev \
		libhdf5-serial-dev \
		libleveldb-dev \
		liblmdb-dev \
		libprotobuf-dev \
		libsnappy-dev \
		protobuf-compiler
RUN git clone --branch 4.0.1 https://github.com/opencv/opencv ~/opencv && \
mkdir -p ~/opencv/build && cd ~/opencv/build && \
cmake -D CMAKE_BUILD_TYPE=RELEASE \
			-D CMAKE_INSTALL_PREFIX=/usr/local \
			-D WITH_IPP=OFF \
			-D WITH_CUDA=OFF \
			-D WITH_OPENCL=OFF \
			-D BUILD_TESTS=OFF \
			-D BUILD_PERF_TESTS=OFF \
			.. &&\
	make -j"$(nproc)" install && \
	ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2


# pytorch
######################################
RUN pip install future \
	&& pip install numpy \
	&& pip install protobuf\
	&& pip install enum34\
	&& pip install pyyaml \
	&& pip install typing\
	&& pip install pillow\
	&& pip install matplotlib\
	&& pip install scikit-learn\
	&& pip install tqdm\
	&& pip install scipy\
	&& pip install pandas

RUN pip install torch==${PYTORCH_VERSION} torchvision




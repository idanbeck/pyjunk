# Base Ubuntu 18.04 image
#FROM ubuntu:18.04
#FROM nvidia/cuda:latest
#FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.1-base
#CMD nvidia-smi

#RUN echo "NVidia and CUDA Drivers"
#
#RUN apt-get update && apt-get install -y build-essential
#RUN apt-get --purge remove -y nvidia*
#
#WORKDIR /src
#
#RUN pwd
#RUN ls /src
#ADD ./src/drivers/. /src/drivers/.                                                 > Get the install files you used to install CUDA and the NVIDIA drivers on your host
#
#RUN ls
#RUN /src/drivers/NVIDIA-Linux-x86_64-460.39.run -s -N --no-kernel-module   > Install the driver.
#RUN rm -rf /tmp/selfgz7                                                   > For some reason the driver installer left temp files when used during a docker build (i don't have any explanation why) and the CUDA installer will fail if there still there so we delete them.
#RUN /src/drivers/cuda_11.2.1_460.32.03_linux.run -noprompt            > CUDA driver installer.
##RUN /tmp/nvidia/cuda-samples-linux-6.0.37-18176142.run -noprompt -cudaprefix=/usr/local/cuda-6.0   > CUDA samples comment if you don't want them.
#RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64         > Add CUDA library into your PATH
#RUN touch /etc/ld.so.conf.d/cuda.conf                                     > Update the ld.so.conf.d directory
#RUN rm -rf /temp/*  > Delete installer files.

RUN echo "Jupyter Notebook Container"

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python-dev \
    vim

RUN pip3 install --upgrade pip

# Working directory
#RUN mkdir /src
WORKDIR /src

# CUDA 11.0
RUN pip install torch==1.8.0+cu111 \
    torchvision==0.9.0+cu111 \ 
    torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html

ADD ./src/requirements.txt /src/.
#COPY /src/. /src/.

# Install requirements
RUN pip3 install -r /src/requirements.txt

EXPOSE 80

# start app
CMD ["jupyter", "notebook", "--allow-root", "--ip='0.0.0.0'", "--port=80"]



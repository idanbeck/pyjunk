Simple template for setting up a new pytorch notebook in a docker container using docker-compose


To build
```
docker build -t experiments .
```

To run

```
docker run --gpus all -v /mnt/c/dev/simple.sensor.perception/notebooks/experiments/src:/src -v /mnt/c/dev/simple.sensor.perception/repos:/src/repos -p 8080:8080 experiments

# On VM
docker run --gpus all -v /home/idan_beck/dev/simple.sensor.perception/notebooks/experiments/src:/src -v /home/idan_beck/dev/simple.sensor.perception/repos:/src/repos -p 80:80 experiments
```

If you want to run without GPU use 
```
docker run -v /mnt/c/dev/simple.sensor.perception/notebooks/experiments/src:/src -v /mnt/c/dev/simple.sensor.perception/repos:/src/repos -p 8888:8888 experiments
```

Docker compose is not currently working

Also, for the above to function you must ensure that the following works for you:

```
docker run --gpus all nvcr.io/nvidia/k8s/cuda-sample:nbody nbody -gpu -benchmark
```

If you're seeing issues with the above on WSL2, you are likely not on the correct version of Windows.  Follow the following link: https://docs.nvidia.com/cuda/wsl-user-guide/index.html.  Note that `nvidia-smi` won't work for you so you need to run the above docker benchmark


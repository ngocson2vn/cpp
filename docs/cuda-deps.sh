# cuda
wget https://developer.download.nvidia.com/compute/cuda/13.1.0/local_installers/cuda_13.1.0_590.44.01_linux.run


pip install nvidia-cudnn-cu13

# User mode
pip install -U nvidia-cudnn-cu13

# Install nvshmem
## arm
wget https://developer.download.nvidia.com/compute/nvshmem/3.5.19/local_installers/nvshmem-local-repo-debian12-3.5.19_3.5.19-1_arm64.deb

## x64
wget https://developer.download.nvidia.com/compute/nvshmem/3.5.19/local_installers/nvshmem-local-repo-debian12-3.5.19_3.5.19-1_amd64.deb

apt install ./nvshmem-local-repo-debian12-3.5.19_3.5.19-1_arm64.deb
apt install /var/nvshmem-local-repo-debian12-3.5.19/libnvshmem3-cuda-13_3.5.19-1_arm64.deb

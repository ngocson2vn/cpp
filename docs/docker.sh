#==================================
# Installation on Debian
#==================================
sudo apt update
sudo apt install -y apt-transport-https ca-certificates curl gnupg lsb-release
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
apt install -y nvidia-container-toolkit

# Move docker dir
systemctl stop docker.socket
systemctl stop docker
systemctl status docker
mv /var/lib/docker /data00/
ln -sf /data00/docker /var/lib/docker
mv /var/lib/containerd /data00/
ln -sf /data00/containerd /var/lib/containerd

# Docker daemon configs
cat <<EOF > /etc/docker/daemon.json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "data-root": "/data00/docker",
    "ipv6": true,
    "fixed-cidr-v6": "fd00::/80"
}
EOF

systemctl start docker
systemctl status docker

#==================================
# Enable gpus and privileged mode
#==================================
IMAGE=image_name
CONTAINER=container_name
# Interactive mode for debugging
docker run --rm -it --privileged --gpus all --network host --cgroupns host -v /sys/fs/cgroup:/sys/fs/cgroup:rw --name ${CONTAINER} $IMAGE /sbin/init
# Deamon mode
docker run -d --privileged --gpus all --network host --cgroupns host -v /sys/fs/cgroup:/sys/fs/cgroup:rw --name ${CONTAINER} $IMAGE /sbin/init


# enter docker
docker exec -it <container_name> bash

# Support ipv6
vim /etc/docker/daemon.json
# "ipv6": true

systemctl restart docker

# Copy
docker cp <host_src_path> <container_name_or_id>:<container_dest_path>
docker cp <container_name_or_id>:<container_src_path> <host_dest_path>

# build
docker build -t sony_torch291_cuda:v1.0.0.1 -f Dockerfile.torch .

docker commit sony_pilot_torch_270 sony_pilot_torch_270:latest

docker save -o ${HOME}/workspace/docker_images/sony_pilot_torch_270.tar sony_pilot_torch_270:latest

docker load -i sony_pilot_torch_270.tar

# interactive mode
docker run -it --name <container_name> <image_name> /usr/bin/bash

# daemon mode
docker run -d --name <container_name> <image_name> /usr/bin/bash -c 'while true; do sleep 3600; done'

# mount
docker run -d --mount type=bind,source=/usr/lib/x86_64-linux-gnu,target=/opt/tiger/x86_64-linux-gnu --name <container_name> <image_name> /usr/bin/bash -c 'while true; do sleep 3600; done'

# gpu
docker run -d --gpus all --name <container_name> <image_name> /usr/bin/bash -c 'while true; do sleep 3600; done'

# enter docker
docker exec -it <container_name> bash

# Support ipv6
vim /etc/docker/daemon.json
# "ipv6": true

systemctl restart docker

# Copy
docker cp <host_src_path> <container_name_or_id>:<container_dest_path>

# build
docker build -t data.aml.snapshot_torch291_cuda:v1.0.0.1 -f Dockerfile.torch .

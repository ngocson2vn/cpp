# https://enterprise-support.nvidia.com/s/article/understanding-pcie-configuration-for-maximum-performance#:~:text=In%20order%20to%20verify%20PCIe%20speed,command%20lspc%20may%20be%20used.&text=Similar%20to%20the%20width%20parameter,capabilities%20and%20status%20are%20reported.&text=Note%3A%20The%20main%20difference%20between,encoding%20overhead%20of%20the%20packet.
apt install -y pciutils

lspci | grep NVIDIA
17:00.0 3D controller: NVIDIA Corporation Device 2236 (rev a1)
31:00.0 3D controller: NVIDIA Corporation Device 2236 (rev a1)
b1:00.0 3D controller: NVIDIA Corporation Device 2236 (rev a1)
ca:00.0 3D controller: NVIDIA Corporation Device 2236 (rev a1)

lspci -s 31:00.0 -vvv | grep Speed
# Get MPS server PID
```Bash
echo get_server_list | nvidia-cuda-mps-control
```

# List MPS Client PIDs
```Bash
echo "get_client_list <MPS_SERVER_PID>" | nvidia-cuda-mps-control
echo "get_client_list 151" | nvidia-cuda-mps-control
```

# In k8s env
```Bash
echo get_device_client_list | nvidia-cuda-mps-control
```
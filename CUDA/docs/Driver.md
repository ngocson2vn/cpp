# Driver Stub
```Bash
cd cuda-13.0/lib64

ls -l libcuda.so*
libcuda.so -> stubs/libcuda.so
libcuda.so.1 -> libcuda.so
```

# Device Files
In my Linux box: `ls -l /dev/nvidia*` shows the following device files:
```Bash
ls -l /dev/nvidia*
crw-rw-rw- 1 root root 195, 254 Dec  1 20:12 /dev/nvidia-modeset
crw-rw-rw- 1 root root 505,   0 Dec  1 20:10 /dev/nvidia-uvm
crw-rw-rw- 1 root root 505,   1 Dec  1 20:10 /dev/nvidia-uvm-tools
crw-rw-rw- 1 root root 195,   0 Dec  1 20:10 /dev/nvidia0
crw-rw-rw- 1 root root 195,   1 Dec  1 20:10 /dev/nvidia1
crw-rw-rw- 1 root root 195,   2 Dec  1 20:10 /dev/nvidia2
crw-rw-rw- 1 root root 195,   3 Dec  1 20:10 /dev/nvidia3
crw-rw-rw- 1 root root 195,   4 Dec  1 20:10 /dev/nvidia4
crw-rw-rw- 1 root root 195,   5 Dec  1 20:10 /dev/nvidia5
crw-rw-rw- 1 root root 195,   6 Dec  1 20:10 /dev/nvidia6
crw-rw-rw- 1 root root 195,   7 Dec  1 20:10 /dev/nvidia7
crw-rw-rw- 1 root root 195, 255 Dec  1 20:10 /dev/nvidiactl

/dev/nvidia-caps:
total 0
cr-------- 1 root root 240, 1 Dec  1 20:10 nvidia-cap1
cr--r--r-- 1 root root 240, 2 Dec  1 20:10 nvidia-cap2
```

Does NVIDIA Driver solib libcuda.so communicates with GPU driver kernel module via these device files? <br/>
Can reading assembly code of libcuda.so find the evidence to prove that? </br>

Yes, your understanding is correct. The NVIDIA user-space driver library (`libcuda.so`) communicates with the NVIDIA GPU kernel driver modules (`nvidia.ko`, `nvidia-uvm.ko`, etc.) primarily by opening and interacting with these device files.

Here is the breakdown of how this works and how you can find evidence of it in the assembly code.

### 1. The Communication Mechanism

In Linux, the standard way for user-space applications to talk to kernel drivers is through **`ioctl` (Input/Output Control)** system calls performed on file descriptors associated with device files.

* **`libcuda.so`** acts as the user-space client.
* **`/dev/nvidia*`** are the "portals" into the kernel.
* **`nvidia.ko`** acts as the kernel-space server.

When you initialize CUDA (e.g., `cudaFree(0)` or `cuInit`), `libcuda.so` performs the following steps:

1. **Open Control Device:** It opens `/dev/nvidiactl` (the "control" device) to query the driver version and list available GPUs.
2. **Open GPU Device:** Once it identifies which GPU to use (e.g., GPU 0), it opens the specific device file `/dev/nvidia0`.
3. **Send Commands:** It uses `ioctl` to send command buffers, kernel launch parameters, and memory allocation requests to the kernel module.

### 2. Can reading the assembly of `libcuda.so` prove this?

**Yes.** Even though `libcuda.so` is proprietary and stripped (meaning debug symbols are removed), it must still make system calls to the Linux kernel to function.

If you analyze the assembly (using tools like `objdump`, `readelf`, or a decompiler like Ghidra/IDA Pro), you will find specific evidence.

#### What to look for in the Assembly:

1. **Imported Symbols:**
`libcuda.so` links against the standard C library (`libc`). You will see it importing and calling the `open` (or `open64`) and `ioctl` functions.
* *Assembly Check:* Look for `CALL <ioctl@plt>` or `JMP <ioctl@plt>`.


2. **String Constants:**
The file paths must be stored somewhere in the binary's data section to be passed to the `open` function.
* *Evidence:* You can usually find the string `/dev/nvidiactl` or format strings like `/dev/nvidia%d` inside the `.rodata` (read-only data) section of the library.


3. **The System Call Pattern:**
In the assembly logic, you will see a pattern roughly equivalent to this C pseudocode:
```c
int fd = open("/dev/nvidiactl", O_RDWR);
ioctl(fd, SOME_NVIDIA_COMMAND_ID, &arguments);
```



### 3. A Easier Way to Prove It (Dynamic Analysis)

While reading static assembly is definitive, it is difficult because the file is large and complex. A much faster way to "see" this communication in real-time is using **`strace`** (System Call Trace).

You can verify the communication without knowing assembly by running a simple CUDA program (like `deviceQuery`) under `strace`.

**Command:**

```bash
strace -e trace=open,openat,ioctl ./my_cuda_app
```

**What you will see in the output:**
You will see explicit calls where `libcuda.so` opens these specific files:

```text
openat(AT_FDCWD, "/dev/nvidiactl", O_RDWR) = 3
openat(AT_FDCWD, "/dev/nvidia0", O_RDWR)   = 4
ioctl(3, _IOC(...), 0x7...)                = 0
ioctl(4, _IOC(...), 0x7...)                = 0
```

### Summary of your Device Files

* **`/dev/nvidiactl`:** The global control device. `libcuda.so` talks to this first to manage global state and discover devices.
* **`/dev/nvidia0` - `/dev/nvidia7`:** The actual character devices for your 8 GPUs. `libcuda.so` opens these to submit work to specific cards.
* **`/dev/nvidia-uvm`:** Used for **Unified Virtual Memory**. `libcuda.so` opens this to manage memory that is shared between the CPU and GPU (allowing page faulting on the GPU).
* **`/dev/nvidia-modeset`:** Generally used by the display driver (X11/Wayland) rather than pure compute, but part of the same driver stack.


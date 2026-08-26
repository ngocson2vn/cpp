# patchelf commands
 ```Bash
pip install patchelf

# First, we’ll add an rpath to our binary executable for our preferred glibc:
patchelf --add-rpath /opt/glibc/lib python3.7
patchelf --add-rpath /opt/glibc/lib libtensorflow_cc.so.1
patchelf --add-rpath /opt/glibc/lib libtensorflow_framework.so.1

# Similarly, we can update the rpath with the –set-rpath option. This might break the program, so use it with caution:
patchelf --set-rpath "/path/glibc-older:/path/libsdl:/path/libgl" my_prog

# To remove an existing rpath:
patchelf --remove-rpath /path/glibc-older my_prog

# We can also update the dynamic linker with —set-interpreter:
patchelf --set-interpreter /opt/glibc/lib/ld-linux-x86-64.so.2 /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7
 ```

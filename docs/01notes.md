# Source node vs Sink node
In a Directed acyclic graph, a source node is a node (also known as a vertex) with no incoming connections from other nodes, while a sink node is a node without outgoing connections.

# Compiler
https://www.quora.com/What-are-the-best-resources-to-learn-about-C++-compiler

# GDB
## Print real type of an object
set print object on
whatis obj

## Print vtable of an object
info vtbl obj

## Generate core file
```Bash
gdb attach $PID
set pagination off
set width 65536
generate-core-file
detach
```
File `core.$PID` should be created in the current working directory.

# std::map
For operator [ ], the type of the index is not the only difference from ordinary arrays. In addition,
you can’t have a wrong index. If you use a key as the index for which no element yet exists, a new
element gets inserted into the map automatically. The value of the new element is initialized by the
default constructor of its type. Thus, to use this feature, you can’t use a value type that has no default
constructor. Note that the fundamental data types provide a default constructor that initializes their
values to zero.

# How can an user-space application call a device driver function?
User-space app --> /dev/device file --> device driver functions

# Comma Operator
```C++
int a = (1, 2, 3);
```
This is the [comma operator](http://en.wikipedia.org/wiki/Comma_operator): evaluation of `a, b` first causes `a` to be evaluated, then `b`, and the result is that of `b`.

`int a = (1, 2, 3)`; first evaluates 1, then 2, finally 3, and uses that last 3 to initialise a. It is useless here, but it can be useful when the left operand of , has side effects (usually: when it's a function call).

# Squiggles
Red squiggly line:  
<img src="./images/red_squiggly_line.png" alt="Red Squiggly Line" width="50%" />

# _GLIBCXX_USE_CXX11_ABI
https://gcc.gnu.org/onlinedocs/libstdc++/manual/using_dual_abi.html
>The _GLIBCXX_USE_CXX11_ABI macro (see Macros) controls whether the declarations in the library headers use the old or new ABI. So the decision of which ABI to use can be made separately for each source file being compiled. Using the default configuration options for GCC the default value of the macro is 1 which causes the new ABI to be active, so to use the old ABI you must explicitly define the macro to 0 before including any library headers. (Be aware that some GNU/Linux distributions configure GCC 5 differently so that the default value of the macro is 0 and users must define it to 1 to enable the new ABI.)

```C++
undefined symbol _ZN10tensorflow12OpDefBuilder4AttrENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

// demangled:
// tensorflow::OpDefBuilder::Attr(std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >)
```
liblagrange_mini_ops.so was built from obj_1.o, obj_2.o, obj_3.o, ..., obj_n.o.  
If any obj_k.so was built with "-D_GLIBCXX_USE_CXX11_ABI=1" (by default), then it will use ABI1.  
Which means `std::__cxx11::`

# Extract Static Library
```Bash
ar x libfolly.a
```

# Feature Column
In machine learning, a feature column represents an individual measurable property or characteristic of the data being used to train a model. Each feature column corresponds to a specific input variable that the model will use to make predictions. 

Here are some key aspects of feature columns:

1. **Representation of Input Data**: Feature columns are the means by which raw data is transformed and represented in a format suitable for the machine learning model. They can be numerical (e.g., age, salary) or categorical (e.g., gender, country).

2. **Preprocessing and Transformation**: Feature columns often require preprocessing and transformation to be useful in a model. For instance, categorical data may be one-hot encoded, numerical data may be normalized, and text data may be tokenized.

3. **Types of Feature Columns**:
   - **Numerical Columns**: Represent continuous values like age, height, or temperature.
   - **Categorical Columns**: Represent discrete values like country, gender, or product type.
   - **Bucketized Columns**: Continuous values divided into discrete buckets.
   - **Embedding Columns**: Represent high-dimensional data (like words or categories) in lower-dimensional space.
   - **Crossed Columns**: Combine two or more feature columns to capture interaction effects between features.

4. **Usage in Frameworks**: Many machine learning frameworks and libraries, such as TensorFlow and Scikit-Learn, provide tools for defining and managing feature columns. For example, in TensorFlow, `tf.feature_column` provides a way to specify and transform feature columns for use in models.

### Example in TensorFlow

Here’s a simple example of defining feature columns in TensorFlow:

```python
import tensorflow as tf

# Define numerical feature columns
age = tf.feature_column.numeric_column("age")
income = tf.feature_column.numeric_column("income")

# Define categorical feature columns with vocabulary
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["male", "female"])

# Define an embedding column for a high-dimensional categorical feature
gender_embedding = tf.feature_column.embedding_column(gender, dimension=8)

# Combine all feature columns
feature_columns = [age, income, gender_embedding]

# Build a feature layer for a Keras model
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

# Example input data
features = {
    "age": [25],
    "income": [50000],
    "gender": ["male"]
}

# Transform features using the feature layer
transformed_features = feature_layer(features)

print(transformed_features)
```

In this example, `age` and `income` are numerical columns, while `gender` is a categorical column that is represented as an embedding. The `DenseFeatures` layer then combines these feature columns and transforms the raw input data into a format suitable for feeding into a neural network.

<<<<<<< HEAD
# Multiple glibc on a Single Linux
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

# Get rpath and set rpath
```Bash
readelf -d libtensorflow_cc.so.1 | head -20

Dynamic section at offset 0x2b4d6a08 contains 38 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libtensorflow_framework.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libpthread.so.0]
 0x0000000000000001 (NEEDED)             Shared library: [libdl.so.2]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [librt.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libstdc++.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000e (SONAME)             Library soname: [libtensorflow_cc.so.1]
 0x000000000000001d (RUNPATH)            Library runpath: [$ORIGIN/../_solib_local/_U_S_Stensorflow_Clibtensorflow_Ucc.so.1.15.5___Utensorflow:$ORIGIN/:$ORIGIN/../../nvidia/cusparse/lib:$ORIGIN/../nvidia/cusparse/lib:$ORIGIN/../../nvidia/cublas/lib:$ORIGIN/../nvidia/cublas/lib:$ORIGIN/../../tensorrt:$ORIGIN/../tensorrt:$ORIGIN/../../nvidia/cuda_cupti/lib:$ORIGIN/../nvidia/cuda_cupti/lib:$ORIGIN/../../nvidia/cusolver/lib:$ORIGIN/../nvidia/cusolver/lib:$ORIGIN/../../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_runtime/lib]
 0x000000000000000c (INIT)               0x2287000
 0x000000000000000d (FINI)               0xb89f0f4
 0x0000000000000019 (INIT_ARRAY)         0x2b18d000
 0x000000000000001b (INIT_ARRAYSZ)       26400 (bytes)
 0x000000000000001a (FINI_ARRAY)         0x2b193720
 0x000000000000001c (FINI_ARRAYSZ)       8 (bytes)

# Also
chrpath -l libpython3.7m.so
patchelf --print-rpath /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7

# Add rpath
patchelf --force-rpath --add-rpath /opt/glibc/lib /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7
```

# Fix rpath for python
```Bash
patchelf --force-rpath --add-rpath /opt/glibc/lib /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7
patchelf --add-rpath /usr/lib/x86_64-linux-gnu /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7

patchelf --force-rpath --set-rpath /data00/son.nguyen/.pyenv/versions/3.7.3/lib /data00/son.nguyen/.pyenv/versions/3.7.3/lib/libpython3.7m.so.1.0
patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu /data00/son.nguyen/.pyenv/versions/3.7.3/lib/libpython3.7m.so.1.0
patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu /data00/son.nguyen/.pyenv/versions/3.7.3/lib/libpython3.so


patchelf --set-rpath "/data00/son.nguyen/.pyenv/versions/3.7.3/lib:/opt/glibc/lib:/usr/lib/x86_64-linux-gnu" /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7

/usr/bin/patchelf --set-rpath /usr/lib/x86_64-linux-gnu /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7

cd /data00/son.nguyen/.pyenv/versions/3.7.3/lib/python3.7/lib-dynload
for solib in $(ls)
do 
    echo $solib
    patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu $solib
done

cd /data00/son.nguyen/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_core/python
patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu _pywrap_tensor_float_32_execution.so
patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu _pywrap_tensorflow_internal.so
patchelf --force-rpath --add-rpath /usr/lib/x86_64-linux-gnu _tf_stack.so

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu python

patchelf --force-rpath --set-rpath /usr/lib/x86_64-linux-gnu /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7
readelf -d /data00/son.nguyen/.pyenv/versions/3.7.3/bin/python3.7
readelf -d /data00/son.nguyen/.pyenv/versions/3.7.3/lib/libpython3.so
readelf -d /data00/son.nguyen/.pyenv/versions/3.7.3/lib/libpython3.7m.so.1.0

cd /usr/lib/x86_64-linux-gnu
sudo ln -s /lib/x86_64-linux-gnu/libgcc_s.so.1 .
```

# iTerm2
https://iterm2colorschemes.com/
```Bash
# Import all color schemes
tools/import-scheme.sh schemes/*
```

# Bazel
```Bash
# You can find logs in the output base for the workspace, which for Linux is typically $HOME/.cache/bazel/_bazel_$USER/<MD5 sum of workspace path>/

# You can find the output base for a workspace by running bazel info output_base in that workspace. Note though that there's a command.log file which contains the output of the last command, and bazel info is itself a command, so that will overwrite command.log. You can do echo -n $(pwd) | md5sum in a bazel workspace to get the md5, or find the README files in the output base directories which say what workspace each is for.
bazel info output_base

# cxxopts
export BAZEL_CXXOPTS="-D_GLIBCXX_USE_CXX11_ABI=0"

# .bazelrc
build --cxxopt=-D_GLIBCXX_USE_CXX11_ABI=0

# show commands with --subcommands
bazel build --subcommands --config=cuda --explain=explain.txt //tensorflow:libtensorflow_cc.so --verbose_failures --jobs 128
# Commands will be logged in build/0a31f298a9820565a8a30548380f28ee/command.log
```

# Blade
```Bash
# Force global dependency
global_settler(prefer_deps=["thirdparty/tensorflow:1155nv_cuda114_sony"])

Blade(warning): Global setting change branch 1.15.3-gpu-cu10.1-gcc8 --> 1155nv_cuda114_sony for thirdparty/tensorflow, depended by lagrange/operators
Blade(warning): Global setting change branch 1.15.3-gpu --> 1155nv_cuda114_sony for thirdparty/tensorflow, depended by lagrange/operators
Blade(warning): Global setting change branch 2.9.2-cuda11.4-cudnn8.2.4-gcc8 --> 1155nv_cuda114_sony for thirdparty/tensorflow, depended by lagrange/operators/shared_variable
```

# vtable for __cxxabiv1::__class_type_info
```Python
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/data00/son.nguyen/.pyenv/versions/3.7.3/lib/python3.7/site-packages/tensorflow_core/python/framework/load_library.py", line 61, in load_op_library
    lib_handle = py_tf.TF_LoadLibrary(library_filename)
tensorflow.python.framework.errors_impl.NotFoundError: /data00/son.nguyen/.pyenv/versions/3.7.3/lib/python3.7/site-packages/sony/tensorsharp/lib/libhybrid_fm_ops_ab0.so: undefined symbol: _ZTVN10__cxxabiv117__class_type_infoE
```

Solution: using `g++` instead of `gcc`  
```Bash
g++ -shared *.o -o libhybrid_fm_ops_ab0.so 
```

# Template Non-Deduced Context
Given the code in [test3.cpp](../Examples/type_deduction/test3.cpp):  
**Deduction Process**:  
1. Template Parameters:
- `T`: The primary template parameter, which can be deduced from the function argument.
- `typename enable_if<uses_write_v<T>>::type`: The second template parameter, which is dependent on `T`.

2. Function Call:  
- When you call `serialize(stdout, w)` with `w` being of type `Widget`, the compiler starts by deducing `T` as `Widget`.

3. Non-Deduced Context:  
- The second template parameter `typename enable_if<uses_write_v<T>>::type` is in a non-deduced context because it is not directly connected to the function parameter list.
- The C++ standard specifies that template parameters appearing in such positions are in non-deduced contexts.

**Why the Compiler Does Not Deduce from the Function Call**  
Non-Deduced Context Rule  
The C++ standard N4778 specifies that certain types of template arguments are not deduced. Specifically:  
- Nested Types: When a type is nested within another type or part of a template argument, it is often placed in a non-deduced context.
- Dependent Types: If the type depends on another template parameter, it might not be deduced.

In the original code, `typename enable_if<uses_write_v<T>>::type` is a type dependent on `T`. It is nested within `enable_if`, making it a non-deduced context. The compiler doesn't attempt to deduce it from the function call arguments because of this classification.

# Template Default Arguments
Given the code in [test1.cpp](../Examples/type_deduction/test1.cpp):  
The default template arguments are inherited from the primary template in C++. This behavior is consistent with the C++ standard.

Here’s a brief overview of how it works:

1. **Primary Template Default Arguments**: If you instantiate a template without specifying all the template arguments, the compiler uses the default arguments provided in the primary template definition.

2. **Specializations and Default Arguments**: When you provide a specialization for a template, if that specialization does not specify all the arguments, the compiler will use the default arguments specified in the primary template.

In your specific case, `enable_if` is a template with two parameters: `bool B` and `typename X` with `X` defaulting to `int`. When you use `enable_if<true>`, the compiler:
- Looks for a specialization of `enable_if<true>` (i.e., `enable_if<true, X>`).
- Since `X` is not specified, the default argument `int` is used from the primary template definition.

**C++ Standard Reference**:
This behavior is described in the C++ standard under the section dealing with template arguments. Specifically:
- **C++17 Standard [temp.arg.nontype]**: Default template arguments are used when a specific instantiation doesn’t provide them.
- **C++20 Standard [temp.arg]**: Similarly, default arguments from the primary template are used when arguments are not explicitly provided.

So, to summarize, yes, default template arguments are inherited from the primary template, and this is a standard rule in C++.

# GLIBC_PRIVATE
```Bash
/usr/lib/x86_64-linux-gnu/libm.so.6: symbol __strtof128_nan, version GLIBC_PRIVATE not defined in file libc.so.6 with link time reference
```
Reason:
```Bash
readelf -sW /usr/lib/x86_64-linux-gnu/libm.so.6 | grep __strtof128_nan
     6: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __strtof128_nan@GLIBC_PRIVATE (13)
  3240: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND __strtof128_nan@@GLIBC_PRIVATE
```
The symbol doesn't exist in `/lib/x86_64-linux-gnu/libc.so.6`


# gdb displays next instruction
```GDB
display/i $pc
```

# symbol lookup error
```C++
./main: symbol lookup error: ./path/to/libX.so: undefined symbol: SYMBOL_Y
```
Reason: 
- Function `f1` in `libX.so` calls function `f2` defined in `main` program.
- But `main` was not built with `-rdynamic` or `--export-dynamic`, so `f2` was not placed in the dynamic symbol table.

https://man7.org/linux/man-pages/man3/dlopen.3.html#EXAMPLES
>Any global symbols in the executable that were placed into its dynamic symbol table by ld(1) can also be used to resolve references in a dynamically loaded shared object.  Symbols may be placed in the dynamic symbol table either because the executable was linked with the flag "-rdynamic" (or, synonymously, "--export-dynamic"), which causes all of the executable's global symbols to be placed in the dynamic symbol table, or because ld(1) noted a dependency on a symbol in another object during static linking.

# Check GPU Memory Usage
```Bash
while true; do nvidia-smi -i 0 --query-gpu=index,gpu_name,utilization.gpu,temperature.gpu,memory.total,memory.used,memory.free --format=csv; echo; sleep 1; done
```

# TensorFlow logging
```Bash
# if severity > TF_CPP_MIN_LOG_LEVEL, output message
export TF_CPP_MIN_LOG_LEVEL=0

# if level < TF_CPP_MIN_VLOG_LEVEL, output message
export TF_CPP_MIN_VLOG_LEVEL=3
```

# TensorFlow Session Run Call Stack:
```C++
Thread 35 (LWP 39171):
#0  syscall () at ../sysdeps/unix/sysv/linux/x86_64/syscall.S:38
#1  0x00007f0851d7b582 in nsync::nsync_mu_semaphore_p_with_deadline(nsync::nsync_semaphore_s_*, timespec) () from ./lib/libtensorflow_cc.so.2
#2  0x00007f0851d7ab99 in nsync::nsync_sem_wait_with_cancel_(nsync::waiter*, timespec, nsync::nsync_note_s_*) () from ./lib/libtensorflow_cc.so.2
#3  0x00007f0851d781ab in nsync::nsync_cv_wait_with_deadline_generic(nsync::nsync_cv_s_*, void*, void (*)(void*), void (*)(void*), timespec, nsync::nsync_note_s_*) () from ./lib/libtensorflow_cc.so.2
#4  0x00007f0851d78683 in nsync::nsync_cv_wait_with_deadline(nsync::nsync_cv_s_*, nsync::nsync_mu_s_*, timespec, nsync::nsync_note_s_*) () from ./lib/libtensorflow_cc.so.2
#5  0x00007f0851212a3c in tensorflow::DirectSession::WaitForNotification(tensorflow::Notification*, long) () from ./lib/libtensorflow_cc.so.2
#6  0x00007f0851212efd in tensorflow::DirectSession::WaitForNotification(tensorflow::Notification*, tensorflow::DirectSession::RunState*, tensorflow::CancellationManager*, long) () from ./lib/libtensorflow_cc.so.2
#7  0x00007f085122213e in tensorflow::DirectSession::RunInternal(long, tensorflow::RunOptions const&, tensorflow::CallFrameInterface*, tensorflow::DirectSession::ExecutorsAndKeys*, tensorflow::RunMetadata*, tensorflow::thread::ThreadPoolOptions const&) () from ./lib/libtensorflow_cc.so.2
#8  0x00007f08512242e7 in tensorflow::DirectSession::Run(tensorflow::RunOptions const&, std::vector<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor>, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor> > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<tensorflow::Tensor, std::allocator<tensorflow::Tensor> >*, tensorflow::RunMetadata*, tensorflow::thread::ThreadPoolOptions const&) () from ./lib/libtensorflow_cc.so.2
#9  0x00007f0851211833 in tensorflow::DirectSession::Run(tensorflow::RunOptions const&, std::vector<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor>, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor> > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<tensorflow::Tensor, std::allocator<tensorflow::Tensor> >*, tensorflow::RunMetadata*) () from ./lib/libtensorflow_cc.so.2
#10 0x00007f085122158f in tensorflow::DirectSession::Run(std::vector<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor>, std::allocator<std::pair<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, tensorflow::Tensor> > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> >, std::allocator<std::__cxx11::basic_string<char, std::char_traits<char>, std::allocator<char> > > > const&, std::vector<tensorflow::Tensor, std::allocator<tensorflow::Tensor> >*) () from ./lib/libtensorflow_cc.so.2
#11 0x00000000033d6dcd in main::$_0::operator() (this=0x59300f0, i=3) at run_pet_model/main.cpp:194
#12 0x00000000033d6d01 in std::__invoke_impl<int, main::$_0, int> (__f=..., __args=@0x59300e8: 3) at /opt/path/to/gccs/x86_64-x86_64-gcc-830/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/bits/invoke.h:60
#13 0x00000000033d6c52 in std::__invoke<main::$_0, int> (__fn=..., __args=@0x59300e8: 3) at /opt/path/to/gccs/x86_64-x86_64-gcc-830/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/bits/invoke.h:95
#14 0x00000000033d6c15 in std::thread::_Invoker<std::tuple<main::$_0, int> >::_M_invoke<0ul, 1ul> (this=0x59300e8) at /opt/path/to/gccs/x86_64-x86_64-gcc-830/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/thread:244
#15 0x00000000033d6bc5 in std::thread::_Invoker<std::tuple<main::$_0, int> >::operator() (this=0x59300e8) at /opt/path/to/gccs/x86_64-x86_64-gcc-830/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/thread:253
#16 0x00000000033d6a4e in std::thread::_State_impl<std::thread::_Invoker<std::tuple<main::$_0, int> > >::_M_run (this=0x59300e0) at /opt/path/to/gccs/x86_64-x86_64-gcc-830/lib/gcc/x86_64-linux-gnu/8.3.0/../../../../include/c++/8.3.0/thread:196
#17 0x00007f077a1aabff in std::execute_native_thread_routine (__p=0x59300e0) at ../../../.././libstdc++-v3/src/c++11/thread.cc:80
#18 0x00007f077a683ca9 in start_thread (arg=0x7f066e7fc000) at pthread_create.c:486
#19 0x00007f07791ea76f in clone () at ../sysdeps/unix/sysv/linux/x86_64/clone.S:95
```

# TensorFlow GraphDef Manipulation
## Add a new node to Graph
```Python
with tf.Session() as sess:
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  graph_pbtxt_path = graph_file_path + "txt"
  graph_writer = open(graph_pbtxt_path, "w")
  stupid_node_def = tf.NodeDef()
  stupid_node_def.name = "X"
  stupid_node_def.op = "Stupid"
  stupid_node_def.input.append("matrixA")
  stupid_node_def.attr["T"].CopyFrom(tf.AttrValue(type=dtypes.as_dtype(tf.float32).as_datatype_enum))
  graph_def.node.append(stupid_node_def)
  graph_writer.write(str(graph_def))
  graph_writer.close()
  print(graph_pbtxt_path)
```

## Modify an existing node
```Python
  graph = tf.get_default_graph()
  graph_def = graph.as_graph_def()
  for node in graph_def.node:
    if node.name == "matrixB":
      new_value = tf.AttrValue()
      new_value.CopyFrom(node.attr["value"])
      new_value.tensor.tensor_shape.CopyFrom(tf.TensorShape([5, 5]).as_proto())
      node.attr["value"].CopyFrom(new_value)
```

# clang format
```Bash
sudo apt install clang-format
```

# union
In C++, a union is a user-defined datatype in which we can define members of different types of data types just like structures. But one thing that makes it different from structures is that the member variables in a union share the same memory location, unlike a structure that allocates memory separately for each member variable. The size of the union is equal to the size of the largest data type.

Memory space can be used by one member variable at one point in time, which means if we assign value to one member variable, it will automatically deallocate the other member variable stored in the memory which will lead to loss of data.

# binary/bit format
```Python
n = 4
print("{0:08b}".format(n))

```

# OpenSSL
Could not build the ssl module!
Python requires an OpenSSL 1.0.2 or 1.1 compatible libssl with X509_VERIFY_PARAM_set1_host().

```Bash
wget https://www.openssl.org/source/openssl-1.0.2.tar.gz
tar -xzf openssl-1.0.2.tar.gz
cd openssl-1.0.2
./config --prefix=/usr
make && sudo make install

yes | cp -vrf /usr/lib64/libcrypto.so.1.0.0 /usr/lib/x86_64-linux-gnu/libcrypto.so.1.0.0
yes | cp -vrf /usr/lib64/libssl.so.1.0.0 /usr/lib/x86_64-linux-gnu/libssl.so.1.0.0
cd /usr/lib/x86_64-linux-gnu
ln -sf libcrypto.so.1.0.0 libcrypto.so
ln -sf libssl.so.1.0.0 libssl.so

# Verify
pyenv global 3.9.0
python
Python 3.9.0 (default, Oct  8 2024, 03:46:07) 
[GCC 8.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import ssl
>>> ssl.OPENSSL_VERSION
'OpenSSL 1.0.2o  27 Mar 2018'
```

# DWARF debug info
```Bash
sudo apt install -y dwarfdump
dwarfdump -ls output/bin/main | grep source_file_name.cc
```

# Get cudnn version
```Bash
cd /opt/tiger/pilot_gpu_service_bin

cat <<EOF > main.c
#include <stdio.h>

size_t cudnnGetVersion();

int main(int argc, char** argv) {
  printf("CUDNN_VERSION: %ld\n", cudnnGetVersion());
}
EOF

gcc -Wl,-rpath=./lib -Wl,--dynamic-linker=./lib/ld-linux-x86-64.so.2 main.c ./lib/libcudnn.so.8 ./lib/libc.so.6 ./lib/ld-linux-x86-64.so.2 -o main

./main
```


Or
```Bash
cat <<EOF > main.c
#include <stdio.h>
#include <dlfcn.h>

typedef size_t (*func)();

int main(int argc, char** argv) {
  void* handle = dlopen("./lib/libcudnn.so.8", RTLD_NOW | RTLD_GLOBAL);
  if (!handle) {
    fprintf(stderr, "ERROR: %s\n", dlerror());
    return 1;
  }
  
  void* func_ptr = dlsym(handle, "cudnnGetVersion");
  printf("CUDNN_VERSION: %ld\n", reinterpret_cast<func>(func_ptr)());
}
EOF

gcc -Wl,-rpath=./lib -Wl,--dynamic-linker=./lib/ld-linux-x86-64.so.2 main.c ./lib/libdl.so.2 ./lib/libc.so.6 ./lib/ld-linux-x86-64.so.2 -o main
```

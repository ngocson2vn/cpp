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

# Jupyter Notebook
```Bash
pip install jupyter
jupyter notebook --port=8080
```

# Macro
```C++
#define f(a,b) a##b
#define g(a) #a
#define h(a) g(a)
```
An argument is macro-replaced before it is substituted into the replacement list, except where it appears as the operand of # (stringize) or ## (concatenate).

`h(f(1,2))`:  
In your macro h, the parameter a is not an argument of one of those two operators, so the argument is macro-replaced and then substitued into the replacement list. That is, the argument f(1,2) is macro replaced to become 1##2, and then to 12, and then it is substituted into g(12), which is (again) macro-replaced to become "12".

`g(f(1,2))`: 
When you invoke g directly, the parameter a is an argument of the # operator, so its argument is not macro-replaced before subsitution: f(1,2) is substituted directly into the replacement list, yielding "f(1,2)".

Another example:
```C++
#define CUDA_ARCH_STR(x) "__CUDA_ARCH__: " #x
#define STRINGIZE_CUDA_ARCH(x) CUDA_ARCH_STR(x)
#pragma message(STRINGIZE_CUDA_ARCH(__CUDA_ARCH__) ", __CUDA_ARCH_FEAT_SM90_ALL is defined")
```
Step 1: The macro `__CUDA_ARCH__` will be expanded to `900`, then be substituted into the body of the macro `STRINGIZE_CUDA_ARCH(x)`.  
`STRINGIZE_CUDA_ARCH(__CUDA_ARCH__) -> CUDA_ARCH_STR(900)`  

Step 2: The macro `CUDA_ARCH_STR(900)` will be expanded to `"__CUDA_ARCH__: " "90"`  

Finally, we got:  
```C++
#pragma message("__CUDA_ARCH__: " "90" ", __CUDA_ARCH_FEAT_SM90_ALL is defined")
```
<br/>

# enum
In C++, whether you can use an enum member directly without the enum's name scope depends on the type of enum you are using: **unscoped (traditional) enums** or **scoped enums**.

### 1. **Unscoped Enum** (C++98 style)

An **unscoped enum** is declared with the `enum` keyword, without `class` or `struct`. In this case, the enum members are injected into the surrounding scope, so you can use them directly without the enum name as a qualifier.

#### Example:
```cpp
enum Color {
    Red,    // Can be accessed as Red
    Green,  // Can be accessed as Green
    Blue    // Can be accessed as Blue
};

int main() {
    Color myColor = Red; // No need to qualify with Color::
    return 0;
}
```

Here, `Red`, `Green`, and `Blue` are accessible directly without the `Color::` prefix because they are placed directly in the surrounding scope. This can lead to name conflicts if other enums or variables share the same names.

### 2. **Scoped Enum** (C++11 and newer)

A **scoped enum** is declared with `enum class` or `enum struct`. In this case, the enum members are **not** injected into the surrounding scope, so you must qualify them with the enum name.

#### Example:
```cpp
enum class Color {
    Red,    // Must be accessed as Color::Red
    Green,  // Must be accessed as Color::Green
    Blue    // Must be accessed as Color::Blue
};

int main() {
    Color myColor = Color::Red; // Must use Color:: prefix
    return 0;
}
```

Here, `Red`, `Green`, and `Blue` can only be accessed as `Color::Red`, `Color::Green`, and `Color::Blue`. Scoped enums provide better type safety and prevent naming conflicts, as the enum members are not accessible outside the enum's scope.

### Summary

- **Unscoped Enums**: Members are injected into the surrounding scope and can be accessed without the enum name, but this may lead to naming conflicts.
- **Scoped Enums**: Members are not injected into the surrounding scope and must be accessed with the enum name as a prefix, ensuring better type safety and avoiding conflicts.
<br/><br/>


# CV-qualifiers
**CV-qualifiers** in C++ refer to the type qualifiers `const` and `volatile`, which modify a type to specify constraints on how objects of that type can be used. The name "cv-qualifiers" comes from the initials of `const` and `volatile`. These qualifiers are a critical part of C++'s type system and are used to enforce rules about mutability and access.

### Types of CV-Qualifiers

1. **`const` (Constant Qualifier)**:
   - Indicates that the object cannot be modified after it is initialized.
   - Attempts to modify a `const` object result in a compile-time error.
   - Example:
     ```cpp
     const int x = 10; // x cannot be modified
     // x = 20; // Error: assignment of read-only variable 'x'
     ```

2. **`volatile` (Volatile Qualifier)**:
   - Indicates that the value of the object may be changed at any time, outside the control of the current code (e.g., by hardware or another thread).
   - Prevents the compiler from optimizing out reads or writes to the variable.
   - Example:
     ```cpp
     volatile int flag = 0;
     while (flag == 0) {
         // Do something
     }
     // The compiler will not optimize this loop
     ```

3. **`const volatile`**:
   - Combines `const` and `volatile`, meaning the object cannot be modified directly by the code, but it might change unexpectedly (e.g., a read-only hardware register).
   - Example:
     ```cpp
     const volatile int status = 0xFF;
     int currentStatus = status; // Allowed
     // status = 0; // Error: cannot modify a const variable
     ```

---

### Application of CV-Qualifiers

CV-qualifiers can apply to various types of entities, such as:
1. **Variables**:
   - Define variables that are constant, volatile, or both.

2. **Pointers**:
   - CV-qualifiers can be applied to the pointer itself or the data it points to:
     ```cpp
     const int *ptr1;      // Pointer to constant integer
     int *const ptr2;      // Constant pointer to an integer
     const int *const ptr3; // Constant pointer to a constant integer
     volatile int *ptr4;   // Pointer to a volatile integer
     ```

3. **Member Functions**:
   - When applied to member functions, `const` means the function cannot modify the object it is called on.
   - `volatile` means the function can be called on `volatile` objects.
     ```cpp
     class MyClass {
     public:
         void foo() const;       // Can't modify `*this`
         void bar() volatile;    // Can be called on volatile objects
         void baz() const volatile; // Both constraints apply
     };
     ```

4. **Function Parameters and Return Types**:
   - CV-qualifiers can be used to specify that function arguments or return values are constant or volatile.

---

### CV-Qualifier Rules
1. **Top-Level CV-Qualifiers**:
   - Apply directly to the variable or object itself.
   - Ignored when copying the variable by value.
     ```cpp
     const int a = 10;
     int b = a; // Allowed: `b` is not const
     ```

2. **Low-Level CV-Qualifiers**:
   - Apply to the type being pointed to (not the pointer itself).
   - These qualifiers are preserved during assignments or conversions involving pointers.
     ```cpp
     const int x = 5;
     const int *p = &x;   // Pointer to a constant integer
     int *q = p;          // Error: cannot convert `const int*` to `int*`
     ```

### Summary
CV-qualifiers (`const` and `volatile`) are essential tools for controlling the behavior and safety of variables and objects in C++. They allow you to declare immutability, handle special hardware scenarios, and enable proper optimization while enforcing type safety.

# GCC
```
/usr/bin/gcc @bazel-out/k8-opt/bin/stablehlo_compiler-2.params
```
Arguments are in the file `bazel-out/k8-opt/bin/stablehlo_compiler-2.params`

# Make a constexpr function run at runtime
By wrapping it by a non-constexpr function, we can force the constexpr function run at runtime. For example,
```C++
  template<typename A, typename O, typename B>
  auto get_layout_a(const ComposedLayout<A, O, B>& layout) {
    return layout.layout_a();
  }
```
where, `layout_a()` is a constexpr function.

# ADL - Argument-Dependent Lookup
https://en.cppreference.com/w/cpp/language/adl


# Type of an object
In C++, the type of an object is **not stored at runtime by the compiler**. Instead, the type information is determined entirely at compile time for most use cases. Here's how it works:

### 1. **Compile-time Storage (Symbol Table)**:
   - During compilation, the compiler keeps a **symbol table** that maps each identifier (e.g., variable or function name) to its type, scope, and other metadata.
   - This information is used for type checking, function overloading, template instantiation, and other compile-time operations.
   - Once the program is compiled into machine code, type information is typically **not included in the binary** unless required for specific features (e.g., polymorphism).  
   For example:
     ```cpp
     int x = 10;
     double y = 3.14;
     ```
     In the symbol table:
     ```
     Identifier   Type      Scope    Attributes
     x            int       global   -
     y            double    global   -
     ```

### 2. **Runtime Type Information (RTTI)**:
   - For polymorphic types (i.e., classes with at least one virtual function), the compiler generates a **vtable** (virtual table) and attaches it to objects of the class at runtime. The vtable contains a pointer to the type information structure.
   - This is used for features like `dynamic_cast` and `typeid` in polymorphic scenarios.
   - The type information is stored in a metadata structure, usually as part of the vtable mechanism, but this only applies to polymorphic types.

### 3. **Non-polymorphic Types**:
   - For non-polymorphic types (e.g., structs, plain old data types, and classes without virtual functions), type information is completely resolved at compile time. The runtime system has no concept of their type, as only raw memory and machine instructions remain.

### Summary:
- The **type of an object** is primarily used at **compile time** for correctness and optimization.
- At runtime, type information exists only for **polymorphic types** through RTTI and vtables.
- For other types, the compiler does not store or provide type information at runtime. It is your responsibility as a programmer to ensure type safety in such cases.
<br/><br/>

# static_cast
### Overview of `static_cast` in C++
`static_cast` is a C++ casting operator that provides a way to perform **type-safe conversions** at compile time. It is used for conversions that the compiler can verify are valid during compilation.

Unlike `reinterpret_cast`, which simply changes the way memory is interpreted, `static_cast` performs stricter checks, ensuring that the conversion is semantically meaningful and safe in the context of the type system.

---

### Key Characteristics of `static_cast`
1. **Compile-Time Cast**:
   - All checks for validity happen at compile time, and no runtime overhead is involved.
   
2. **Type-Safe (Within Limits)**:
   - Only allows conversions between compatible types.
   - For example, `int` to `float`, `void*` to another pointer type, or between related classes in an inheritance hierarchy.

3. **Restrictions**:
   - Cannot cast between completely unrelated types.
   - No runtime type-checking like `dynamic_cast` for polymorphic types.

4. **Usage**:
   - Safer than `reinterpret_cast` for many use cases but less flexible.

---

### Syntax
```cpp
static_cast<new_type>(expression)
```
- `new_type`: The type you want to cast to.
- `expression`: The value or object to be cast.

---

### How It Works Step by Step

1. **Parsing and Validation**:
   - The compiler analyzes the cast to determine if it’s valid.
   - For instance:
     - Converting between numeric types (e.g., `int` to `float`).
     - Upcasting or downcasting in an inheritance hierarchy.
     - Casting `void*` to a specific pointer type.

2. **Compile-Time Checks**:
   - The compiler verifies that:
     - The conversion is defined by the C++ standard.
     - The types are compatible (e.g., related classes, numeric types, or void pointers).

3. **Code Generation**:
   - If the cast is valid, the compiler generates appropriate machine instructions to perform the conversion (if needed).
   - In some cases (e.g., pointer conversions), no actual instructions are generated, and the compiler just interprets the value differently.

---

### Concrete Examples

#### Example 1: Numeric Conversions
```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 10;
    double y = static_cast<double>(x); // Convert int to double

    cout << "x (int): " << x << endl;
    cout << "y (double): " << y << endl;

    return 0;
}
```
**How It Works**:
- The compiler generates instructions to convert the integer value `x` to a double.
- This involves widening the value (adding extra precision).

---

#### Example 2: Upcasting in an Inheritance Hierarchy
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void show() { cout << "Base class" << endl; }
};

class Derived : public Base {
public:
    void show() override { cout << "Derived class" << endl; }
};

int main() {
    Derived d;
    Base* b = static_cast<Base*>(&d); // Upcasting: Derived* -> Base*
    b->show();

    return 0;
}
```
**How It Works**:
- The `static_cast` allows safe conversion from `Derived*` to `Base*` (upcasting).
- Since `Derived` is derived from `Base`, the compiler verifies the relationship and generates no additional runtime checks.

---

#### Example 3: Downcasting in an Inheritance Hierarchy
```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void show() { cout << "Base class" << endl; }
};

class Derived : public Base {
public:
    void show() override { cout << "Derived class" << endl; }
};

int main() {
    Base b;
    Derived* d = static_cast<Derived*>(&b); // Unsafe downcasting!
    d->show(); // Undefined behavior

    return 0;
}
```
**How It Works**:
- The `static_cast` forces a conversion from `Base*` to `Derived*` (downcasting).
- The compiler trusts the programmer but does not insert runtime checks.
- If the object is not actually a `Derived`, accessing members of `Derived` leads to **undefined behavior**.

---

#### Example 4: Casting `void*` to a Specific Type
```cpp
#include <iostream>
using namespace std;

int main() {
    int x = 42;
    void* ptr = &x; // Generic void pointer
    int* intPtr = static_cast<int*>(ptr); // Cast back to int*

    cout << "Value: " << *intPtr << endl;

    return 0;
}
```
**How It Works**:
- `static_cast` converts the generic `void*` pointer back to its original type `int*`.
- The compiler assumes the cast is correct and generates no additional runtime checks.

---

#### Example 5: Prohibited Casts
```cpp
#include <iostream>
using namespace std;

class A {};
class B {};

int main() {
    A a;
    // B* b = static_cast<B*>(&a); // Error: No relationship between A and B

    return 0;
}
```
**Why It Fails**:
- The compiler rejects this cast because `A` and `B` are unrelated types, and there’s no meaningful way to perform the conversion.

---

### Behind the Scenes

1. **Pointer Adjustments** (for Inheritance):
   - If casting between base and derived class pointers, the compiler adjusts the pointer to account for the offset between the base and derived class in memory.
   - Example:
     ```cpp
     struct Base { int x; };
     struct Derived : Base { int y; };

     Derived d;
     Base* b = static_cast<Base*>(&d);
     ```
     Here, `b` points to the `Base` portion of `d`, and the compiler adjusts the pointer accordingly.

2. **No Runtime Checks**:
   - Unlike `dynamic_cast`, `static_cast` does not verify at runtime whether the cast is safe.

3. **Numeric Conversions**:
   - For numeric types, the compiler generates machine instructions to convert the value (e.g., widening or narrowing conversions).

---

### Use Cases of `static_cast`

1. **Safe Conversions**:
   - Between related types in an inheritance hierarchy (e.g., upcasting).
   - Between numeric types (e.g., `float` to `int`).

2. **Converting `void*` to Specific Pointer Types**:
   - Used in low-level programming (e.g., working with C-style APIs).

3. **Avoiding Implicit Conversions**:
   - Makes explicit what would otherwise happen implicitly.

---

### Summary
`static_cast` is a safer and more controlled casting operator compared to `reinterpret_cast`, but it does not provide runtime safety like `dynamic_cast`. It’s primarily a compile-time mechanism for:
- Numeric conversions.
- Pointer adjustments in inheritance.
- Casting between `void*` and specific pointer types.

Its use is recommended when the cast is logically valid, and you want the compiler to enforce type correctness.
<br/>

# The ->* operator
The `->*` operator in C++ is used to dereference a pointer to a member function (or member variable) when it is called on a pointer to an object of the class. <br/>
The official name of the `->*` operator in C++ is the **pointer-to-member operator**.

This operator is specifically required because member functions have an implicit `this` pointer, so they are fundamentally different from non-member functions. The `->*` operator helps resolve the combination of a pointer to an object and a pointer to a member function.  

Ref: https://en.cppreference.com/w/cpp/language/operator_member_access#Built-in_pointer-to-member_access_operators

Example: [Pointer to a Member Function](../Examples/operators/pointer_to_member.cc)

## Why `->*` Is Needed
The `->*` operator is necessary because:

A member function pointer cannot be invoked directly like a normal function pointer due to the implicit this pointer.
The operator specifies that the call is being made on the object pointed to by `objPtr`.


# rvalue reference
The C++ standard specifies that a function returning a `T&&` produces an unnamed temporary expression. Unnamed temporaries are rvalues by nature. Here's a breakdown:

An rvalue reference (`T&&`) itself is just a type, not a value category.

It indicates that the function allows binding to an rvalue.  
When the function result is used in an expression, the value category of the result is what matters.  
Value category of a function call:

A function returning by value produces an rvalue (a temporary).  
A function returning by reference (T&) produces an lvalue.  
A function returning by rvalue reference (T&&) also produces an rvalue because the value category of the result is prvalue (pure rvalue).

=======
# Octal escape sequences
In C++, sequences like `\000` are **octal escape sequences**. They represent characters by their value in octal (base 8) format. Here’s what you need to know:

- `\000` is an escape sequence that represents the **null character** (`'\0'`), which has a value of **0** in both decimal and octal.
- In general, an octal escape sequence takes the form of `\nnn`, where `nnn` can be up to three octal digits (ranging from `\000` to `\377`).
  
For example:
- `\000` → represents the null character (`'\0'`).
- `\101` → represents the character `'A'`, because 101 in octal is 65 in decimal, which is the ASCII code for `'A'`.

These octal escape sequences are used to represent ASCII or Unicode characters directly by their numeric values in the program.

# std::enable_if_t
```C++
// If bool_value = true, then typename = void
// If bool_value = false, then substitution failure will occurr
typename = std::enable_if_t<bool_value>
```

# Pack expansion
A pattern followed by an ellipsis, in which the name of at least one pack appears at least once, is expanded into zero or more instantiations of the pattern, where the name of the pack is replaced by each of the elements from the pack, in order. Instantiations of alignment specifiers are space-separated, other instantiations are comma-separated.
```C++
template<class... Us>
void f(Us... pargs) {}
 
template<class... Ts>
void g(Ts... args)
{
    f(&args...); // “&args...” is a pack expansion
                 // “&args” is its pattern
}
 
g(1, 0.2, "a"); // Ts... args expand to int E1, double E2, const char* E3
                // &args... expands to &E1, &E2, &E3
                // Us... pargs expand to int* E1, double* E2, const char** E3
```

Sample code: [pack_expansion](../Examples/pack_expansion/)

**Note:** the C++ language does not allow you to expand a pack into a comma-separated list of expressions like that inside parentheses unless it's part of a valid construct such as an initializer list, a fold expression, or a braced list. For example,
```C++
  // This initializer_list argument pack expansion is essentially equal to
  // using a fold expression with a comma operator. Clang however, refuses
  // to compile a fold expression with a depth of more than 256 by default.
  // There seem to be no such limitations for initializer_list.
  (void)std::initializer_list<int>{0, (printType<Args>(), 0)...};
```
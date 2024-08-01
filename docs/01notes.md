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

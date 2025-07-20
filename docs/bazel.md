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

# Query targets
./bazel query "@xla//xla:*"
./bazel query "@xla//xla/service:*"
./bazel query "@xla//xla/service/gpu:*"
./bazel query "@xla//xla/stream_executor:*"

./bazel query "@xla//xla/tsl/profiler/utils:*"
./bazel query "@xla//xla/tsl/profiler/backends/cpu:*"
./bazel query "@xla//xla/tsl/protobuf:*"

# Dynamic linker
# The -rdynamic flag is used to instruct the linker to export all dynamic symbols to the dynamic linker, making them available at runtime. 
--linkopt="-rdynamic"

# Disable remote cache
--noremote_accept_cached
--noremote_upload_local_results

# Multiple configs
--config=cuda --config=torch_cuda --config=torch_debug

# Query deps
bazel query "deps(//mlir/disc:disc_compiler_main)" 2>&1 | grep libtensorflow_framework_import_lib
```

# TensorFlow
```Bash
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4/}
export CUDA_TOOLKIT_PATH="${CUDA_HOME}"
export TF_CUDA_HOME=${CUDA_HOME} # for cuda_supplement_configure.bzl
export TF_CUDA_PATHS="${CUDA_HOME},${HOME}/.cache/cudnn/"
export TF_CUDA_COMPUTE_CAPABILITIES="7.5,8.0,8.6,8.9,9.0"
export CUDACXX=${CUDACXX:-"${CUDA_HOME}/bin/nvcc"}
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/extras/CUPTI/lib64:$LD_LIBRARY_PATH

SRC_FILE_LIST=+tensorflow/core/common_runtime/direct_session.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/executor.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/framework/device.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/gpu/gpu_device.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/device/device_event_mgr.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/cuda/cuda_gpu_executor.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/gpu/gpu_stream.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/gpu/gpu_event.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/gpu/gpu_util.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/event.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/cuda/cuda_platform.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/executor_cache.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/gpu/gpu_process_state.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/stream_executor/cuda/cuda_driver.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/framework/op_kernel.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/framework/shape_inference.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/framework/tensor.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/framework/function.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/immutable_executor_state.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/propagator_state.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/common_runtime/simple_propagator_state.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/kernels/reverse_op.cc,
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/kernels/linalg/matrix_band_part_op.cc
SRC_FILE_LIST=${SRC_FILE_LIST},+tensorflow/core/kernels/scan_ops.cc
eval "${BAZEL_BIN} build ${BAZEL_JOBS_LIMIT} --config=opt --linkopt=-g --per_file_copt=${SRC_FILE_LIST}@-O0,-g,-fno-inline --strip=never --verbose_failures //tensorflow:libtensorflow_cc.so"
```

# Common repos and targets
```Python
"@com_google_protobuf//:protobuf",
"@com_github_gflags_gflags//:gflags",

# libtensorflow_framework.so.2
"@org_tensorflow//tensorflow:libtensorflow_framework_import_lib",
```

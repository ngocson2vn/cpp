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
```

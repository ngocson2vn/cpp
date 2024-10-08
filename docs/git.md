# submodule
```Bash
git submodule add -f https://github.com/tensorflow/tensorflow.git third_party/tensorflow
git config -f .gitmodules submodule.third_party/tensorflow.shallow true
git submodule update --init --recursive
```
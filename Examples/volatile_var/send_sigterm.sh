#!/bin/bash

pid=$(ps -u $(whoami) -www | grep main | grep -v grep | awk '{print $1}')
cwd=$(realpath /proc/${pid}/cwd)
if [[ "$(pwd)" == "${cwd}" ]]; then
  kill -SIGTERM ${pid}
fi

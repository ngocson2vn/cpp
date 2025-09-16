#!/bin/bash

export TORCH_LOGS="+dynamo"
export TORCHDYNAMO_VERBOSE="1"

export SKIP_TRACE=1
export DYNAMO_CACHE_START_INDEX=1

export PYTHONPATH=/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python

python3.11 model_runner.py -m test_inductor

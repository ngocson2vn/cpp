# Dependencies
```Bash
# python
Python 3.11.2

# torch
pip install torch 

unset http_proxy https_proxy
pip3.11 install torch 
pip3.11 install pydot matplotlib ipdb

sudo apt-get install -y graphviz cmake ccache

# openai triton
cd triton
pip install -e .
```

# b triton
cd triton
pip3.11 install -r python/requirements.txt
pip3.11 install -e python/

# Install cutlass
pip3.11 install cuda-python==12.4.0
pip3.11 install nvidia-cutlass
# cutlass templates will be installed to `site-packages/cutlass_library/`

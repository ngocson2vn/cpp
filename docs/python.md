# Pdb and Autocomplete
Create ~/.pdbrc
```Python
# this adds tab completion
import rlcompleter
__import__('pdb').Pdb.complete = rlcompleter.Completer(locals()).complete
```

# pip install without cache
```Bash
pip3 install --no-cache-dir -I bytedlogid==0.2.1
```

# Update requirements of a wheel file
```Bash
wheel unpack ./file_name.whl

# Update file_name/file_name.dist-info/METADATA

wheel pack ./file_name
```

# Debugging
Using debugpy: [Example Debug Inductor](../.vscode/launch.json)
```json
        {
            "name": "Debug Inductor",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceRoot}/MLCompiler/pytorch/test_inductor.py",
            "cwd": "${workspaceRoot}/MLCompiler/pytorch",
            "env": {
                "PYTHONPATH": "/data00/home/son.nguyen/workspace/triton_dev/bytedance/triton/python",
                "LD_LIBRARY_PATH": "/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/extras/CUPTI/lib64"
            }
        }
```

# Python 3.11
```
export PIP_BREAK_SYSTEM_PACKAGES=1
pip3 install/uninstall PACKAGE_NAME
```

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
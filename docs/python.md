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
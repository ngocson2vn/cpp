# Pdb and Autocomplete
Create ~/.pdbrc
```Python
# this adds tab completion
import rlcompleter
__import__('pdb').Pdb.complete = rlcompleter.Completer(locals()).complete
```

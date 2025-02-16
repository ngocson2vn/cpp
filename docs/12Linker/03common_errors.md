# Error: stack smashing detected
```Bash
./run.sh
*** stack smashing detected ***: <unknown> terminated
```
**Solution:**<br/>
Double-check **LD_LIBRARY_PATH** and remove suspicious paths.<br/>
For example, removing the following path resolves the issue:
```Bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu
```

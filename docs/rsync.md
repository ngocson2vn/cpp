# Parallel rsync
```Bash
# On client side
split -n tarball.tar
ls | xargs -n1 -P 10 -I% /usr/bin/rsync -avP % server:~/workspace/tools/

# On server side
cat part* > tarball.tar

# rsync create directory on demand
# -r, --recursive             recurse into directories
# -R, --relative              use relative path names
rsync -avrRP src dst
```

# rsync the contents of a directory
To copy all contents of a folder, including hidden ones, specify the directory path without using *. 

The trailing slash / on the source indicates the contents of the directory should be copied.

```bash
rsync -avP /path/to/src/ /path/to/dst/
```

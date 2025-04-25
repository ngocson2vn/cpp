# Parallel rsync
```Bash
# On client side
split -n tarball.tar
ls | xargs -n1 -P 10 -I% /usr/bin/rsync -avP % server:~/workspace/tools/

# On server side
cat part* > tarball.tar

# rsync create directory on demand
# -r option will create the destination directory it it does not exist
rsync -arvP src dst
```

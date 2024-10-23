# Parallel rsync
```Bash
# On client side
split -n tarball.tar
ls | xargs -n1 -P 10 -I% /usr/bin/rsync -avP % server:~/workspace/tools/

# On server side
cat part* > tarball.tar
```

# tmux server
By default, tmux server will create a socket in `/tmp/tmux-0/`
```bash
find /tmp/tmux-0/
/tmp/tmux-0/
/tmp/tmux-0/default
```

## Start another tmux server
In order to start another tmux server, we must specify a different socket directory:
```bash
export TMUX_TMPDIR=/tmp/tmux2
mkdir -p ${TMUX_TMPDIR}

find /tmp/tmux2
/tmp/tmux2
/tmp/tmux2/tmux-0
/tmp/tmux2/tmux-0/default

# New session
tmux
```

# Enable mouse
```bash
echo 'set -g mouse on' >> ~/.tmux.conf
```

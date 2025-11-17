# /etc/ssh/sshd_config
```config
AllowTCPForwarding yes
PubkeyAuthentication yes
AllowUsers user1 user2

# TCP Keep Alive
TCPKeepAlive yes
```

# Test sshd_config
```Bash
sshd -T
```

# Debugging
```Bash
vim /etc/ssh/sshd_config
LogLevel DEBUG
```

# Remote-SSH
Command + Shift + P: Open User Settings
```
"remote.SSH.useLocalServer": false,
```

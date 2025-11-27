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

# Bash TMOUT
```
TMOUT is a built-in environment variable in the Bash shell (and some other shells like Ksh) that specifies an idle timeout period for interactive shell sessions. It's measured in seconds and is designed to enhance security by automatically terminating inactive sessions, which helps prevent unauthorized access if a user leaves a terminal unattended.
```
**How It Works**:<br/>
If TMOUT is set to a positive integer (e.g., TMOUT=900 for 15 minutes), and no input is received within that timeframe, the shell will exit automatically.
- A value of 0 disables the timeout (unlimited idle time).
- If unset, there's no timeout by default.
- It's read-only in some configurations, meaning you can't change it mid-session, but you can unset it or override it in shell configuration files.
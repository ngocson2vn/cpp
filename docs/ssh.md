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


# Test ssh proxy
Tunnel: `MacBook -> proxy -> target`
```Bash
# Test if proxy is working
ssh -N -D 1080 son.nguyen@proxy.laniakea.com

# Get verbose
ssh -vvv \
  -o ConnectTimeout=10 \
  -W '[IPv6]:22' \
  proxy.laniakea.com

# -D 1080: Creates a local SOCKS proxy on port 1080.
# -N: Prevents requesting a shell session (fixes your specific error).
# -v: Enables verbose mode so you can see if the tunnel sets up successfully.
```

# VSCode reports that port forwarding is disabled
Solution:

1. Ensure that server side sshd_config includes `AllowTCPForwarding yes`

2. Remove the host from client site `~/.ssh/known_hosts`

# SSh Tunnel
```bash
ssh -L 9999:127.0.0.1:9999 gpudev_va2
```
This command sets up an **SSH local port forward** (often called an SSH tunnel). It securely routes traffic from a port on your local computer to a specific port on a remote server.

Here is the breakdown of exactly what each part does:

* **`ssh`**: The standard command used to securely connect to a remote machine.
* **`-L`**: The flag that tells SSH to enable **L**ocal port forwarding.
* **`9999:127.0.0.1:9999`**: The routing rule, which follows the strict format `[local_port]:[destination_host]:[destination_port]`.
* **`9999` (first)**: The port on your *local* computer that SSH will open and listen to.
* **`127.0.0.1`**: The destination address *from the perspective of the remote server*. In this case, `127.0.0.1` means "localhost" on the remote server itself.
* **`9999` (second)**: The target port on the remote server where your traffic will be delivered.


* **`gpudev_va2`**: The name of the remote server you are logging into. Because this is an alias rather than a standard IP address (like `192.168.1.5`) or domain, it means `gpudev_va2` is defined in your local SSH configuration file (usually located at `~/.ssh/config`), which tells SSH the actual IP address, username, and security keys to use for this connection.


The best way to understand this is to think of the remote server (`gpudev_va2`) as a **proxy** or a **middleman**.

When you set up an SSH tunnel, the SSH program on your local laptop doesn't just send raw data directly to the final destination. Instead, it securely wraps your data and hands it to the SSH daemon (the background service handling SSH) running on the remote server.

Once the data arrives, the remote server unwraps it and asks, *"Where should I send this now?"*

Because the remote server is the one asking the question, the destination address is resolved entirely from **its network location, not yours**.

### The Step-by-Step Journey

If you run `ssh -L 9999:127.0.0.1:9999 gpudev_va2` and open `http://localhost:9999` in your web browser, here is the exact journey of your traffic:

1. **Your Laptop:** Your web browser sends a request to port `9999` on your local machine.
2. **The Tunnel Entrance:** Your local SSH client intercepts this request, encrypts it, and sends it over the internet to `gpudev_va2`.
3. **The Tunnel Exit:** The SSH daemon on `gpudev_va2` receives the encrypted data, decrypts it, and looks at the routing rule you provided: `127.0.0.1:9999`.
4. **The Final Delivery:** Because the remote server is executing this rule, it opens a network connection to *its own* `127.0.0.1` (localhost) on port `9999` and drops the data there.

> **The Golden Rule of SSH Forwarding:** The middle section of the `-L` flag (`[destination_host]`) is always resolved by the remote server, as if you were physically sitting at that server and typing the address into a web browser.

### The Contrast: Routing to a Third Machine

To make this "perspective" concept crystal clear, consider what happens if you change that middle address to something else:

`ssh -L 9999:10.0.5.50:80 gpudev_va2`

If you ran this command, the journey changes at step 4. When the traffic arrives at `gpudev_va2`, the server looks at the rule and says, *"I need to send this to `10.0.5.50` on port `80`."*

`10.0.5.50` might be an internal database or a private web server sitting on the same company network as the GPU server—a machine your laptop cannot reach over the public internet. `gpudev_va2` acts as a "jump host," fetching the internal data on your behalf and sending it back through the secure tunnel to your laptop.

### Why use 127.0.0.1 on the remote server at all?

You might wonder why a Jupyter Notebook or TensorBoard running on the remote GPU server is bound to `127.0.0.1` instead of its public IP address in the first place.

It is a security best practice. By configuring the tool to listen *only* to `127.0.0.1`, the tool refuses to accept connections from the outside internet. The only way to access it is to be physically on the machine, or to use an SSH tunnel to securely drop yourself "inside" the machine's local environment.
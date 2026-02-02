# SO_REUSEPORT
Here is a detailed breakdown of the `SO_REUSEPORT` mechanism.

### High-Level Summary

`SO_REUSEPORT` is a socket option available in modern operating systems (Linux kernel 3.9+, BSDs, macOS) that allows **multiple sockets to bind to exactly the same IP address and port combination**.

Traditionally, if one process bound to `0.0.0.0:80`, any other process trying to bind to that same address would fail with an "Address already in use" error. `SO_REUSEPORT` lifts this restriction, enabling multiple worker processes (or threads) to open their own distinct listening sockets on the same port. The kernel then automatically distributes incoming connections across these sockets.

---

### The Mechanism: How It Works

The mechanism fundamentally changes how the kernel handles incoming packets for a specific port. Here is the step-by-step flow:

#### 1. Independent Sockets

Unlike the traditional "pre-fork" model (where a parent process opens *one* socket and forks children who inherit that *same* file descriptor), `SO_REUSEPORT` allows each worker to create its own distinct socket file descriptor.

#### 2. The Security Check

To prevent "port hijacking" (where a malicious user binds to a port already used by a critical service), the kernel enforces a strict security rule: **All sockets binding to the same port must belong to the same Effective User ID (EUID).**

#### 3. Kernel-Level Load Balancing

This is the core of the mechanism. When an incoming TCP connection request (a SYN packet) arrives, the kernel must decide which of the multiple listening sockets should receive it.

* **Hashing:** The kernel calculates a hash based on the connection's **4-tuple** (Source IP, Source Port, Destination IP, Destination Port).
* **Distribution:** Based on this hash, the kernel deterministically assigns the connection to **one** of the listening sockets.
* **Separate Queues:** Each listener process has its own dedicated accept queue. The connection is placed into that specific process's queue.

---

### Key Advantages

#### 1. Elimination of the "Thundering Herd"

In the traditional model (shared socket), when a new connection arrived, all sleeping worker processes were woken up to compete for the lock to accept the connection. This caused significant CPU churn (lock contention).

With `SO_REUSEPORT`, only the specific process selected by the hash is woken up.

#### 2. Improved CPU Locality

Because the kernel distributes connections based on a hash, a specific connection flows consistently to the same process. This works well with **CPU affinity** (pinning processes to specific CPU cores), reducing CPU cache misses and improving performance for high-throughput networking.

#### 3. Zero-Downtime Updates

You can start a new generation of worker processes (binding to the port) alongside the old generation. You can then gracefully shut down the old processes. The kernel simply stops hashing new connections to the closed sockets and directs them to the new ones.

---

### Comparison: Traditional vs. SO_REUSEPORT

| Feature | Traditional (Shared FD / Fork) | SO_REUSEPORT |
| --- | --- | --- |
| **Architecture** | One socket, one queue, shared by many processes. | Multiple sockets, multiple queues. |
| **Load Balancing** | Processes compete (race) to `accept()` the next connection. | Kernel distributes connections via hashing. |
| **Bottleneck** | The single Accept Queue lock (spinlock). | None (parallel processing). |
| **Wake-up Behavior** | Thundering Herd (many wake up, one succeeds). | Precise Wake-up (only the target process wakes). |

---

### Important Caveats

While powerful, `SO_REUSEPORT` has specific behaviors you must be aware of:

* **Hash Changes:** If the number of listening sockets changes (e.g., a process crashes or a new one starts), the hashing algorithm's distribution changes. Packets for *in-flight* handshakes (connections currently being established) might be routed to the wrong process, potentially causing connection resets during scaling events.
* **UDP Support:** This mechanism works excellently for UDP as well (e.g., used heavily by DNS servers), distributing datagrams across workers.

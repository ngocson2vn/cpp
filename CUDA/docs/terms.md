# snooping
In compute architecture, snooping is a hardware mechanism where individual cache controllers continuously monitor (or "snoop") a shared bus or interconnect to watch for memory transactions made by other processors.

**How Snooping Works**
- Active Monitoring: Every cache controller "listens" to the address and control lines on the shared communication bus.
- Detecting Changes: If a core writes new data to a memory address, a broadcast message goes out on the bus. The snooping logic in the other cache controllers detects this broadcast.
- Taking Action: If a controller sees an operation on a memory address it also holds in its own local cache, it reacts immediately. It typically invalidates its local copy (marking it stale so it won't be used) or updates it with the new data.

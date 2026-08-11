# Profile
## Step 1: Add NVTX ranges
Modify /path/to/app.py:
```Python
with torch.no_grad(), torch.autograd.profiler.emit_nvtx():
    torch.cuda.nvtx.range_push(f"inference")
    outputs = model(inputs)
    torch.cuda.nvtx.range_pop()
```
<br/>

If the app is written in C++, then use the corresponding NVTX C APIs:
```C++
#include <nvtx3/nvToolsExt.h>

nvtxRangePushA("ipc_sync");
std::unique_ptr<IOMessage, std::function<void(IOMessage*)>> response = ipc->ipc_sync(builder.message());
nvtxRangePop();
```

## Step 2: Adjust perf_event_paranoid
Change the paranoid level to 1 to enable CPU kernel sample collection
```Bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

## Step 3: Launch app with `nsys launch`
Please note that both `nsys launch` and `nsys xxx` must use the same env var `TMPDIR`.
```Bash
# Terminal 1
nsys launch --session-new=master --trace=cuda,nvtx,osrt --python-sampling=true python3 /path/to/app.py
```

```Python
    begin_cmd = [
        "/usr/bin/numactl",
        "--interleave=all",
    ],
    if os.getenv("NSYS_LAUNCH", "0") in ["1", "true"]:
        begin_cmd = [
            "/usr/local/cuda-13.1/bin/nsys",
            "launch",
            "--session-new=master",
            "--trace=cuda,nvtx,osrt",
            "--python-sampling=true"
        ]
    cmd = begin_cmd + [
        "python3",
        f"{start_script}",
    ]
```

## Step 4: Collect profiling data
Please note that both `nsys launch` and `nsys xxx` must use the same env var `TMPDIR`.
```Bash
# Terminal 2
#===============================
# Double check TMPDIR
#===============================
echo $TMPDIR

/usr/local/cuda-13.1/bin/nsys sessions list
/usr/local/cuda-13.1/bin/nsys status --session=master

/usr/local/cuda-13.1/bin/nsys start --session=master --output=./manhattan_worker_profile --force-overwrite=true --sample=cpu --backtrace=dwarf
/usr/local/cuda-13.1/bin/nsys status --session=master

# Wait for a while
/usr/local/cuda-13.1/bin/nsys stop --session=master
```
<br/>

# View C++ call stack
Step 1: Left click and drag to select the desired block <br/>
Step 2: Right click and select 'Filter and Zoom in' <br/>
Step 3: Navigate to the bottom pane and select "Bottom-Up View"
<img src="./images/view_call_stack.png" width="80%" />


# Check kernel duration
<img src="./images/kernel_duration.png" width="80%" />
<br/>

# Show kernels in Events View window
<img src="./images/show_kernels.png" width="80%" />


# Common Issues
## Unable to collect CPU kernel IP/backtrace samples. perf event paranoid level is 2.
**Solution:** Change the paranoid level to 1 to enable CPU kernel sample collection:
```Bash
sudo sh -c 'echo 1 >/proc/sys/kernel/perf_event_paranoid'
```

# Query nsys report using sqlite3
### Step 1: Export the Profile to SQLite
```bash
nsys export --type sqlite worker_profile.nsys-rep
```

### Step 2: Using Built-in `sqlite3` (No external dependencies)

This method uses the standard library to execute the query and iterates through the results to print them row by row.

```python
import sqlite3

# 1. Connect to the exported SQLite database
db_path = "worker_profile.sqlite"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 2. Define the SQL query
query = """
SELECT 
    s.value AS Kernel_Name, 
    k.gridX || 'x' || k.gridY || 'x' || k.gridZ AS Grid_Size 
FROM CUPTI_ACTIVITY_KIND_KERNEL AS k 
JOIN StringIds AS s ON k.demangledName = s.id;
"""

# 3. Execute the query and fetch the results
cursor.execute(query)
results = cursor.fetchall()

# 4. Print the results in a formatted table
print(f"{'Kernel_Name':<60} | {'Grid_Size'}")
print("-" * 80)
for row in results:
    kernel_name = row[0]
    # Truncate extremely long kernel names for display purposes
    if len(kernel_name) > 58:
        kernel_name = kernel_name[:55] + "..."
    grid_size = row[1]
    print(f"{kernel_name:<60} | {grid_size}")

# 5. Close the connection
conn.close()
```
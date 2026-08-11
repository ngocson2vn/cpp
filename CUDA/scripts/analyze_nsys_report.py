import sys
import math
import random
import sqlite3

# 1. Connect to the database
conn = sqlite3.connect("worker_profile.sqlite")
cursor = conn.cursor()

# 2. Step One: Find the boundaries of the first 'sony_inference' session
# Because we are not joining tables here, this is extremely fast.
cursor.execute("""
    SELECT globalTid, start, end 
    FROM NVTX_EVENTS 
    WHERE text = 'sony_inference' 
    ORDER BY start;
""")
all_sessions = cursor.fetchall()

if not all_sessions:
  print("No 'sony_inference' sessions found.")
  sys.exit(1)

rand_idx = random.randint(0, len(all_sessions) - 1)
print(f"Inference session idx {rand_idx}")
session = all_sessions[rand_idx]
global_tid, session_start, session_end = session

# 3. Step Two: Find the kernels restricted strictly to these boundaries
# By passing the exact timestamps as parameters, SQLite filters the data instantly.
kernel_query = """
SELECT 
    s.value AS Kernel_Name,
    k.gridX || 'x' || k.gridY || 'x' || k.gridZ AS GridSize,
    (k.gridX * k.gridY * k.gridZ) AS TotalBlocks,
    (k.end - k.start) AS Duration_ns
FROM CUPTI_ACTIVITY_KIND_RUNTIME AS r
JOIN CUPTI_ACTIVITY_KIND_KERNEL AS k 
    ON k.correlationId = r.correlationId
JOIN StringIds AS s 
    ON k.demangledName = s.id
WHERE r.globalTid = ? 
  AND r.start >= ? 
  AND r.end <= ?
ORDER BY TotalBlocks;
"""
# ORDER BY k.start;

# Execute the second query using the variables we found in Step 1
cursor.execute(kernel_query, (global_tid, session_start, session_end))
rows = cursor.fetchall()

tsv_file = "nsys_cuda_kernels.tsv"
tsv = open(tsv_file, "w")

# 4. Print the formatted output
print(f"--- Kernels for 'sony_inference' session starting at {session_start} ns ---")
tsv.write("Idx\tKernelName\tGridSize\tTotalBlocks\tDuration(ns)\n")
# print(f"{'KernelName':<256} | {'TotalBlocks':<15} | {'Duration(ns)'}")
# print("-" * 90)

MAX_LEN = 128
for idx, row in enumerate(rows):
    kernel_name = row[0]
    if len(kernel_name) > MAX_LEN:
       kernel_name = kernel_name[:MAX_LEN] + " ..."
    grid_size = row[1]
    total_blocks = math.prod([int(e) for e in grid_size.split("x")])
    duration_ns = row[2]
    tsv.write(f"{idx}\t{kernel_name}\t{grid_size}\t{total_blocks}\t{duration_ns}\n")
    # print(f"{kernel_name:<256} | {grid_size:<15} | {total_blocks:<15} | {duration_ns}")

conn.close()

tsv.close()
print(f"Output: {tsv_file}")

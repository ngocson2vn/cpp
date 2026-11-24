import numpy as np
import multiprocessing as mp
from multiprocessing import shared_memory

import time

def worker(jobs: mp.Queue, rets: mp.Queue):
  shm_name, shape, dtype = jobs.get()
  shm = shared_memory.SharedMemory(name=shm_name)
  # print(f"==> Opened shared memory {shm_name}")
  arr = np.ndarray(shape=shape, dtype=dtype, buffer=shm.buf)
  rets.put(arr.sum())
  del arr
  shm.close()
  

def main():
  NPROCS = 128
  jobs = mp.Queue()
  rets = mp.Queue()

  procs: list[mp.Process] = []
  for _ in range(NPROCS):
    procs.append(mp.Process(target=worker, args=(jobs, rets)))
  for p in procs:
    p.start()

  shm_list: list[shared_memory.SharedMemory] = []
  for _ in range(NPROCS):
    data = np.random.rand(1024, 2048)
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_list.append(shm)
    arr = np.ndarray(shape=data.shape, dtype=data.dtype, buffer=shm.buf)
    arr[:] = data
    jobs.put((shm.name, arr.shape, arr.dtype))
    # print(f"==> Created shared memory {shm.name}")

  for _ in range(NPROCS):
    print(rets.get())

  for p in procs:
    p.join()

  # time.sleep(60)
  for shm in shm_list:
    shm.close()
    shm.unlink()


if __name__ == "__main__":
  mp.set_start_method("spawn")
  main()

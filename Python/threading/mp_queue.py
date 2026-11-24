import os
import multiprocessing as mp

# The jobs may be shared unevenly;
# one worker might process 5 jobs and another 0,
# but every worker still gets its stop signal.
def worker(jobs: mp.Queue, rets: mp.Queue):
  name = f"worker_{os.getpid()}"
  while True:
    n = jobs.get()
    if n is None:
      break
    rets.put(f"{name} processed job {n}")

def main():
  NPROCS = 8
  JOB_COUNT = 2048
  jobs = mp.Queue()
  rets = mp.Queue()

  # Start worker processes
  procs = []
  for _ in range(NPROCS):
    procs.append(mp.Process(target=worker, args=(jobs, rets, )))
  for p in procs:
    p.start()

  # Enqueue jobs
  for i in range(JOB_COUNT):
    jobs.put(i)

  # Stop signal for each worker process
  for i in range(NPROCS):
    jobs.put(None)

  # MUST read rets before calling p.join() in order to empty the rets pipe 
  # so that the feeder thread in each worker process can flush its own buffer to the rets pipe
  for _ in range(JOB_COUNT):
    print(f"ret = {rets.get()}")

  # Get results
  for p in procs:
    p.join()

if __name__ == "__main__":
  main()

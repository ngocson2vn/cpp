import multiprocessing as mp

def worker(counter):
  with counter.get_lock():
    counter.value += 1

def main():
  NPROCS = 1024
  counter = mp.Value('q', 0)
  procs = [mp.Process(target=worker, args=(counter, )) for _ in range(NPROCS)]
  for p in procs:
    p.start()
  for p in procs:
    p.join()

  print(f"counter: {counter.value}")

if __name__ == "__main__":
  main()

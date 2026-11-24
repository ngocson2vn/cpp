from concurrent.futures import ProcessPoolExecutor

def worker(n):
  return 2**n

def main():
  pool = ProcessPoolExecutor(3)
  results = [pool.submit(worker, i).result() for i in range(10)]
  print(f"results: {results}")

if __name__ == "__main__":
  main()

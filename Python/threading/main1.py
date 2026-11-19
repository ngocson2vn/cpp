from concurrent.futures import ThreadPoolExecutor

def worker(x):
  return x**10

def main():
  with ThreadPoolExecutor(10) as tp:
    outs = list(tp.map(worker, range(10)))
  print(outs)


if __name__ == "__main__":
  main()

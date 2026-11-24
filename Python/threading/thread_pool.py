from concurrent.futures import ThreadPoolExecutor

tp = ThreadPoolExecutor(10)

def worker(x):
  return x**10

def main1():
  future = tp.submit(worker, 2)
  print(future.result())

def main2():
  outs = list(tp.map(worker, range(10)))
  print(outs)


if __name__ == "__main__":
  main1()
  main2()


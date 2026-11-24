import threading
import queue
from queue import Queue
import time

class Event(object):
  def __init__(self):
    self.status = False

  def fire(self):
    self.status = True

  def fired(self):
    return self.status

def producer(n: int, jobs: Queue):
  for i in range(n):
    jobs.put(i)
  jobs.put(None)

def consumer(jobs: Queue, event: Event):
  while True:
    try:
      job_id = jobs.get(timeout=0.1)
      if not job_id:
        event.fire()
        print("==> jobs is empty <==")
        return
      print(f"Processed job {job_id}")
    except queue.Empty:
      if event.fired():
        return
      time.sleep(0.1)

def main():
  jobs = Queue()
  threads: list[threading.Thread] = []

  # One producer thread
  threads.append(threading.Thread(target=producer, args=(1000, jobs, )))

  event = Event()

  # 10 consumer threads
  threads.extend([
    threading.Thread(target=consumer, args=(jobs, event, ))
    for _ in range(10)
  ])

  for t in threads:
    t.start()

  for t in threads:
    t.join()

if __name__ == "__main__":
  main()

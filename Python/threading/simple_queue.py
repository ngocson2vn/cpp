import threading
from collections import deque

class SimpleQueue(object):
  def __init__(self):
    self.items = deque()
    self.lock = threading.Lock()
    self.not_empty = threading.Condition(self.lock)

  def put(self, item):
    with self.lock:
      self.items.append(item)
      self.not_empty.notify()

  def get(self):
    with self.lock:
      while not self.items:
        self.not_empty.wait()
      return self.items.popleft()

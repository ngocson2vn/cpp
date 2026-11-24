import multiprocessing as mp
from multiprocessing.connection import Connection
import time

class Server:
  def __init__(self, conn: Connection):
    self.conn = conn

  def __call__(self):
    while True:
      msg: str = self.conn.recv()
      if msg is None:
        break
      self.conn.send(msg.upper())



def main():
  client_end, server_end = mp.Pipe()
  server = Server(server_end)
  server_process = mp.Process(target=server, daemon=True)
  server_process.start()

  for msg in ["hello", "how", "are", "you"]:
    client_end.send(msg)
    ret = client_end.recv()
    print(ret)

  time.sleep(3600)

if __name__ == "__main__":
  main()

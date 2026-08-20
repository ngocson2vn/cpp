import time
import threading

def process(track):
    print(f"Process Track object {id(track)}")

class Track:
    def __del__(self):
        print(f"Track object {id(self)} to be destroyed")

track_obj = Track()
t = threading.Thread(target=process, args=(track_obj,))
del track_obj
print("'del track_obj' is executed")
time.sleep(3)
t.start()
t.join()

class Track:
    def __del__(self):
        print(f"Track object {id(self)} to be destroyed")

buf16 = Track()
buf18 = buf16
del buf16          # nothing is printed
print(f"Track object {id(buf18)} still alive:", buf18 is not None)

del buf18          # now the object is destroyed
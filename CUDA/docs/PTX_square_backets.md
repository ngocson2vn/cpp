# Square backets []
Square brackets `[%r1]` mean **dereferencing the memory address stored in `%r1`**, and they are always required when fetching a value from or storing a value to a memory address.


It has two use cases:
### The "Load" Case
```MLIR
ld.shared.v2.b32 	{%r378, %r379}, [%r443+512];
```
In this case, the brackets are on the Right Side (the Source).
- The Logic: "Take the address in `%r443+512`. Go to that Memory Location ([]). COPY what is inside that location and put it into my hands (`{%r378, %r379}`)."
- Result: You are reading from the location.

### The "Store" Case
```MLIR
@%p42 st.global.v4.b32 [ %rd28 + 0 ], { %r374, %r375, %r376, %r377 };
```
In this case, the brackets are on the Left Side (the Destination).
- The Logic: "Take the data in my hands (`{ %r374, %r375, %r376, %r377 }`). Take the address in `%rd28`. Go to that Memory Location ([]). PASTE the data into that location."
- Result: You are writing to the location.


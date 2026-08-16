import torch

ROWS = 8
COLS = 64
VEC = 8
K = 64//VEC

def S(i):
  return VEC * (i % K)

A = torch.randint(0, 10, (ROWS, COLS)).tolist()
print("Before swizzling:")
for row in A:
  # Update the first 8 elements in every row i
  for j in range(VEC):
    row[j] = j

  # Create printRow for row i
  printRow = []
  for k in range(COLS//VEC):
    chunk = tuple(row[(k*VEC):(k*VEC + VEC)])
    printRow.append(chunk)
  print(printRow)
print()

# Perform swizzling
swizzled_A = []
for i in range(ROWS):
  row = [0] * COLS
  print(f"S({i}) = {S(i)}")
  for j in range(COLS):
    idx = j ^ S(i)
    row[idx] = A[i][j]
  swizzled_A.append(row)

print("\nAfter swizzling:")
for row in swizzled_A:
  # Create printRow for row i
  printRow = []
  for k in range(COLS//VEC):
    chunk = tuple(row[(k*VEC):(k*VEC + VEC)])
    printRow.append(chunk)
  print(printRow)
print()

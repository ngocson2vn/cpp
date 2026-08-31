EPSILON = 1e-3
MAX_PRINTS = 10

def diff(a, b):
  matched = True
  mismatch_count = 0
  print_count = 0
  for i in range(a.shape[0]):
      for j in range(a.shape[1]):
          diff = abs((a[i, j] - b[i, j]))
          if diff > EPSILON:
              matched = False
              mismatch_count += 1
              if print_count < MAX_PRINTS:
                print(f"{a[i, j]} != {b[i, j]}")
              elif print_count < 100:
                print(".", end="")
              print_count += 1
  if print_count >= MAX_PRINTS:
    print()
  return matched, mismatch_count
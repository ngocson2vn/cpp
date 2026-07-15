#include <cstdio>

int main() {
  int worker_state = 0b010101;
  int slot_id = __builtin_ctzll(worker_state);
  printf("slot_id = %d\n", slot_id);

  worker_state = 0b010100;
  slot_id = __builtin_ctzll(worker_state);
  printf("slot_id = %d\n", slot_id);
}
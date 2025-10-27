# Pointer to Implementation
### Core Idea
- The public class declaration in the header contains only:
  - Public interface methods.
  - A forward declaration of the `Impl` class.
  - A smart pointer (or raw pointer) to `Impl`.
- The `Impl` class lives in the `.cpp` and contains:
  - All private members.
  - Implementation of logic, including headers that would otherwise leak into the public header.

### Concrete Example: A Logger Class
[pattern_pointer_to_impl](../Examples/pattern_pointer_to_impl/)
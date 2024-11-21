#include <iostream>
#include <memory>
#include <string>

class MemoryArena {
 public:
  MemoryArena(size_t arena_size) {
    buffer_ = static_cast<char*>(malloc(arena_size));
  }

  virtual ~MemoryArena() {
    free(buffer_);
  }

  char* Alloc(size_t size) {
    char* ptr = buffer_ + offset_;
    offset_ += size;
    return ptr;
  }

  char* buffer() {
    return buffer_;
  }

 private:
  char* buffer_ = NULL;
  size_t offset_ = 0;
};

class ArenaString {
 public:
  ArenaString(const char* content, MemoryArena* arena) {
    void* memory = arena->Alloc(sizeof(std::string));
    std::cout << "arena memory memoryess: " << memory << std::endl;
    std::cout << "BEFORE populating string data: '" << *static_cast<std::string*>(memory) << "'" << std::endl;
    ptr_ = new (memory) std::string(content);
    std::cout << "string memory memoryess: " << ptr_ << std::endl;
    std::cout << "AFTER populating string data: '" << *ptr_ << "'" << std::endl;
  }

 private:
  std::string* ptr_;
};

int main(int argc, char** argv) {
  // Create arena
  auto arena = MemoryArena(1024); // 1KiB
  {
    ArenaString s("test arena and placement new operator", &arena);
  }

  // We still can retrieve the string address
  std::cout << "Confirm: '" << *reinterpret_cast<std::string*>(arena.buffer()) << "'" << std::endl;
}

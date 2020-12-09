#include <iostream>
#include <string>
#include <typeinfo>

class Slice {
    public:
        Slice(const Slice& src): data_(src.data()), size_(src.size()) { }

        Slice(const std::string& s) : data_(s.data()), size_(s.size()) { }

        const char* data() const { return data_; }

        size_t size() const { return size_; }
    
    private:
        const char* data_;
        size_t size_;
};

class Builder {
    public:
        void Add(const Slice& key, const Slice& value) {
            std::cout << key.data() << std::endl;
            std::cout << value.data() << std::endl;
        }
};

void printSlice(Slice slice) {
    std::cout << "data: " << slice.data() << std::endl;
    std::cout << "size: " << slice.size() << std::endl;
}

int main() {
    std::string key = "key1";
    std::string value = "value1";
    Slice k1 = key;
    const Slice& k2 = key;
    // Builder* builder = new Builder();
    // builder->Add(key, value);

    printSlice(k1);
}
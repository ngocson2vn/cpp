#include <iostream>

// --- The CRTP Base Class ---
template <typename Derived>
class Task {
public:
    // This is the public interface that clients will call.
    void execute() {
        std::cout << "Setting up task environment...\n";
        
        // This line is the enforcer. 
        // If 'Derived' does not have a 'process_impl()' method, 
        // the compiler will fail right here when this template is instantiated.
        static_cast<Derived*>(this)->process_impl();
        
        std::cout << "Cleaning up task environment...\n";
    }
};

// --- A "Good" Derived Class ---
// It correctly implements the required method.
class DataProcessingTask : public Task<DataProcessingTask> {
public:
    // The Base class requires this exact method signature
    void process_impl() {
        std::cout << "Processing data...\n";
    }
};

// --- A "Bad" Derived Class ---
// It forgets to implement the required method.
class BrokenTask : public Task<BrokenTask> {
public:
    void some_other_method() {
        std::cout << "Doing something else...\n";
    }
};

int main() {
    // 1. This works perfectly.
    DataProcessingTask good_task;
    good_task.execute(); 
    
    // 2. This will trigger a COMPILE-TIME ERROR.
    BrokenTask bad_task;
    // bad_task.execute(); // UNCOMMENTING THIS BREAKS THE BUILD
    
    return 0;
}

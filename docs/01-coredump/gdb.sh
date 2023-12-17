set scheduler-locking on
set scheduler-locking step
set step-mode on
set print elements 0
set pagination off

# Check disassembly-flavor 
show disassembly-flavor

# Measure execution time
python import time
python s=time.time()
continue
python print(time.time() - s)

# Check hang threads
set pagination off
set logging on
thread apply all bt

# Break point at a function
b pilot/gpu/common/tensorflow_session.cpp:lagrange::pilot::TensorflowSession::run_session

# Regular expression breakpoint command
info functions lagrange::pilot::TensorflowSession::run_session(int
rb pilot/gpu/common/tensorflow_session.cpp:lagrange::pilot::TensorflowSession::run_session(int

grep -B 50 "lagrange::pilot::TensorflowSession::run_session" gdb.txt


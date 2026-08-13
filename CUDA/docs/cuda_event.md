## CUDA Event
A CUDA event is a lightweight synchronization primitive. For example:
```python
import torch
event = torch.cuda.Event()
stream_a = torch.cuda.Stream()
stream_b = torch.cuda.Stream()

# Producer on stream_a
triton_tem_fused_2.run(buf_in, buf_out, stream=stream_a.cuda_stream)
event.record(stream_a)

# Consumer that needs buf_out, running on stream_b
stream_b.wait_event(event)
triton_tem_fused_3.run(buf_out, stream=stream_b.cuda_stream)
```

## Key points
### 1. A CUDA stream is a FIFO command queue

### 2. Record a CUDA event
```python
event.record(stream_a)
```
This is a thin wrapper around the CUDA runtime call:

```c
cudaEventRecord(event, stream_a)
```

**What it actually does:**

- It inserts a *marker* into `stream_a` at the current position in that stream's command queue.

### 3. The timing a CUDA event become "signaled"
The marker becomes "signaled" (complete) **only after every operation that was previously submitted to `stream_a` has finished executing on the GPU.**

**Important consequences:**

- Because the Triton kernel was launched on `stream_a` *before* the `record`, the event effectively marks the completion of that kernel (and any other work that was already in the stream). In other words, a CUDA event becomes signaled if and only if all work that was enqueued in its stream before the record call has finished executing on the GPU. When you call `event.record(stream_a)` and the stream has no remaining unfinished work, the event is recorded in the **already-signaled** state. From that moment on it is considered complete — any subsequent `stream.wait_event(event)` will see it as signaled and will not wait.

- If you later launch more kernels on `stream_a` *after* the record, those later kernels are **not** covered by this particular event.

- An event can be recorded multiple times. Each new `record` overwrites the previous completion point.
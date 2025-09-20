# Question List
1. How are symbolic variables created?

2. How does Scheduler fuse two SchedulerNode nodes?

3. How does Scheduler codegen a FusedSchedulerNode to a Triton kernel?

4. How to fuse more nodes?
The following pattern should be fused horizontally: <br/>
```
[M, K] x [K, N1] # matmul_1
[M, K] x [K, N2] # matmul_2
[M, K] x [K, N3] # matmul_3
```

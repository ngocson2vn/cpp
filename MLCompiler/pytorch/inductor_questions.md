# Question List
1. How are symbolic variables created?

2. How does Scheduler fuse two SchedulerNode nodes?

3. How does Scheduler codegen a FusedSchedulerNode to a Triton kernel?

4. How to fuse more nodes?
Goal: reduce >= 10 ms <br/>

Step 1: Try to identify more fusible patterns: <br/>
The following pattern should be fused horizontally: <br/>
```
[M, K] x [K, N1] # matmul_1
[M, K] x [K, N2] # matmul_2
[M, K] x [K, N3] # matmul_3
```
Solution: 
- Create a Triton template for this pattern

<br/>
Step 2: Implement a custom fusion algo

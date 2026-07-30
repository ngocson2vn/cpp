# Common CUDA Errors
## invalid configuration argument
For example,
```
Reason 2 root error(s) found.
  (0) Internal: invalid configuration argument
	 [[{{node combine_trans_11/block.0_1/ptffn_mlp_1/adaptive_ffn_1/pwff_adapt_tower1_1_0_6/gelu_12/mul_3}}]]
	 [[Select_10/_4421]]
  (1) Internal: invalid configuration argument
	 [[{{node combine_trans_11/block.0_1/ptffn_mlp_1/adaptive_ffn_1/pwff_adapt_tower1_1_0_6/gelu_12/mul_3}}]]
```
In this case, the node `combine_trans_11/block.0_1/ptffn_mlp_1/adaptive_ffn_1/pwff_adapt_tower1_1_0_6/gelu_12/mul_3` executes a TF operator which tries to launch a CUDA kernel using the CUDA Driver API `cuLaunchKernel`. However, this API returns `invalid configuration argument` error because some parameter is invalid.

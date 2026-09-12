# TensorFlow Optimization Pipeline
## Step 1: Dump weight tensors from PS
Save weight tensors into a pickle file.

## Step 2: Replace VariableV2
Replace VariableV2 ops with Placeholder nodes, which accept weight tensors

## Step 3: Apply an optimization pass

## Step 4: Run optimized model

## Step 5: Diff output tensors with previous output tensors

## Step 6: Revert the pass if diff is too large

## Step 7: Run next optimization pass
Repeat step 3

## Step 8: Create a frozen graph
- Replace Placeholder nodes accepting weight tensors with Const nodes
- Run an end2end diff check
- Serialize the Graph to a frozen protobuf file
- Generate a cache key from optimization pass list
- Save the pipeline to a pickle file using the cache key as file name
- Upload the pickle file to an object storage bucket

## Step 9: Next round
- Repeat step 1
- Generate a cache key from optimization pass list
- Download the cached pipeline from the object storage bucket
- Repeat step 8


# Optimization Passes
## Concat Forward to reduce H2D latency
- Client side: Instead of feeding a model with multiple 2D input tensors, we concatenate them along axis=1 and then feed the concatenated tensor into the model. 
- Snapshot: Replace all 2D Placeholder nodes with just one 2D Placeholder node and one SplitV node, which splits the 2D Placeholder node according to a 1D tensor along axis=1

## Grappler
1. Constant Folding
2. Common Subexpression Elimination

## BF16 Compute
- Create a separate Graph object for BF16
- Create Placeholder nodes with BF16
- Cast Const nodes to BF16
- Copy other nodes using tf.Operation.from_node_def

## OpsFusion
For applying a TF compiler

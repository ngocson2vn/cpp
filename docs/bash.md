# Array
```Bash
MODEL_NAME=this_is_a_sample_model_r6522063_0

# Using () to turn a space-separated string into an array
parts=($(echo ${MODEL_NAME} | tr '_' ' '))

# Access the array backward using a negative index
ref_id=${parts[-2]}

echo ${ref_id}
```
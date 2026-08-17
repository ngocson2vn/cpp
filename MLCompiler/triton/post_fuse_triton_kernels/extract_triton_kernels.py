import ast


def extract_triton_kernels_via_ast(file_path):
    print(f"Parsing AST for: {file_path}...")

    with open(file_path, "r") as f:
        source_code = f.read()

    # Parse the entire file into an Abstract Syntax Tree
    tree = ast.parse(source_code)

    extracted_kernels = {}

    # Walk through every node in the AST
    for node in ast.walk(tree):
        # We are looking for function calls: async_compile.triton(...)
        if isinstance(node, ast.Call):
            # Check if the function being called is an attribute (e.g., obj.method)
            if isinstance(node.func, ast.Attribute):
                # Check if the method is 'triton' and the object is 'async_compile'
                if node.func.attr == "triton" and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "async_compile":

                        # Ensure the call has at least 2 positional arguments
                        if len(node.args) >= 2:
                            arg0 = node.args[0]  # The kernel name
                            arg1 = node.args[1]  # The multiline source code

                            # In Python 3.8+, string literals are represented as ast.Constant
                            if isinstance(arg0, ast.Constant) and isinstance(
                                arg1, ast.Constant
                            ):
                                kernel_name = arg0.value
                                kernel_source = arg1.value

                                extracted_kernels[kernel_name] = kernel_source

    return extracted_kernels



# --- Example Usage ---
# extracted_code = extract_function_string_via_ast(kernel_source_string, "triton_poi_fused_12")
# print(extracted_code)
def extract_function_string_via_ast(kernel_source_string, target_func_name):
    # Parse the raw string into a syntax tree
    tree = ast.parse(kernel_source_string)
    
    for node in ast.walk(tree):
        # Look for a function definition matching the target name
        if isinstance(node, ast.FunctionDef) and node.name == target_func_name:
            # Slice the exact source code segment from the original string
            # Note: ast.get_source_segment requires Python 3.8+
            func_source = ast.get_source_segment(kernel_source_string, node)
            return func_source
            
    return None


if __name__ == "__main__":
    target_file = "model.py"

    kernels = extract_triton_kernels_via_ast(target_file)

    if not kernels:
        print("No async_compile.triton() calls found.")
    else:
        print(f"Successfully extracted {len(kernels)} kernels.\n")

        # Example: Print the first 10 lines of triton_poi_fused_12
        target_kernel = "triton_poi_fused_10"
        if target_kernel in kernels:
            print(f"--- {target_kernel} ---")
            source_lines = kernels[target_kernel].strip()
            # print("\n".join(source_lines))
            target_kernel_src = extract_function_string_via_ast(source_lines, target_kernel)
            print(target_kernel_src)

# PyTorch FX GraphModule
**Definition:** A GraphModule is a Python-level abstraction in PyTorch's FX framework, representing a computation graph derived from symbolic tracing or manual construction. It is a subclass of torch.nn.Module and encapsulates a computational graph (torch.fx.Graph) along with associated parameters and buffers.

**Purpose:** Used for graph-based transformations, optimizations, and analysis in Python, providing a high-level interface for manipulating PyTorch models.

**Key Characteristics:**<br/>
- Python-Based: The GraphModule is implemented in Python and is part of the torch.fx module, making it accessible for Python-based graph manipulation.

- Structure: Contains a torch.fx.Graph object, which is a directed acyclic graph (DAG) of nodes representing operations (e.g., call_function, call_module, placeholder, output) and their dependencies.

- Dynamic Dimensions: Supports dynamic shapes through symbolic shapes (e.g., SymInt from torch.fx.experimental.symbolic_shapes) or None for flexible dimensions.

- Usage: Primarily used for tasks like model optimization, quantization, or conversion to other formats (e.g., TorchScript, ONNX). It allows developers to inspect and modify the graph programmatically.

# TorchScript
TorchScript is a way to create serializable and optimizable models in PyTorch by converting Python-based PyTorch code into a static, intermediate representation (IR) that can be executed independently of Python. It enables deployment in production environments, such as C++ runtimes, and supports optimizations for performance. TorchScript bridges the gap between PyTorch's eager execution (dynamic computation) and a more static, graph-based execution model.

## Underlying Computational Graph**
The computational graph in TorchScript is a static representation of the model's operations, stored as an Intermediate Representation (IR). This graph is the core of TorchScript's execution and optimization capabilities.<br/><br/>

**Structure of the Computational Graph**
- Nodes: Represent operations (e.g., matrix multiplication, activation functions, or control flow like loops).

- Edges: Represent tensors flowing between operations.

- Attributes: Store metadata like constant values or parameters.

- Graph Representation: The graph is expressed in a human-readable format (via model.graph) and can be optimized or compiled.


# Lowering Pipeline
PyTorch model -> AOT Module with dynamic shapes -> TorchScript -> [Torch-MLIR + MHLO] -> MHLO

# TORCH_LIBRARY macro

# Dynamo Symbolic Symbol Logging
```Python
# /data00/home/son.nguyen/.pyenv/versions/3.9.0/lib/python3.9/site-packages/torch/fx/experimental/symbolic_shapes.py
# export TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL=u5
# The following changes help print out user code stack trace.
    def _log_create_unbacked_symbol(
        self,
        prefix: str,
        symbol: sympy.Symbol,
        vr: ValueRanges,
        source: Optional[Source] = None,
        sym_node: Optional[SymNode] = None,
    ) -> None:
        is_debug = config.extended_debug_create_symbol is not None and str(
            symbol
        ) in config.extended_debug_create_symbol.split(",")
        sloc: Union[str, SLoc]
        sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
        # if source is None:
        #     sloc, maybe_extra_debug = self._get_stack_summary(is_debug)
        # else:
        #     sloc, maybe_extra_debug = source.name(), ""
        maybe_extra_debug = ""
        log.info(
            "%s %s [%s, %s] %s%s",
            prefix,
            symbol,
            vr.lower,
            vr.upper,
            sloc,
            maybe_extra_debug,
            stack_info=is_debug,
        )
        trace_structured(
            "create_unbacked_symbol",
            metadata_fn=lambda: {
                "symbol": str(symbol),
                "node_id": id(sym_node),
                "vr": f"[{vr.lower}, {vr.upper}]",
                "user_stack": structured.get_user_stack(3),
                "stack": "", # structured.get_framework_stack(),
            },
        )
```

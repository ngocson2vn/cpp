#!/usr/bin/env python3
"""
Analyze Torch Inductor generated model.py for independent kernels
and suggest multi-stream scheduling using the `ast` module.

Implements the 5 steps:

1. Extract the operation sequence from the main `call(args)` body.
2. Recover read / write sets for each kernel (heuristic based on
   allocation-before-first-use patterns common in Inductor output).
3. Build the dependence DAG (producer → consumer on buffers).
4. Multi-stream list scheduling (greedy load-balancing + critical-path
   affinity) and report independent groups.
5. Emit a suggested multi-stream skeleton (events + stream assignment)
   that can be used as a starting point for a rewritten `call`.

Usage:
    python analyze_independent_kernels.py [/path/to/model.py]
"""

from __future__ import annotations

import ast
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Op:
    """One statement of interest inside call()."""
    kind: str                  # "alloc" | "kernel" | "del" | "other"
    lineno: int
    name: Optional[str] = None # kernel name or buffer name
    tensors: List[str] = field(default_factory=list)  # all tensor Names appearing
    raw: Optional[ast.AST] = None


@dataclass
class KernelInfo:
    idx: int                   # index in the op sequence
    name: str
    lineno: int
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    preds: Set[int] = field(default_factory=set)   # predecessor kernel indices
    succs: Set[int] = field(default_factory=set)
    stream: int = -1           # assigned stream
    level: int = -1            # topological level


# ---------------------------------------------------------------------------
# Step 1 – Extract operation sequence via AST
# ---------------------------------------------------------------------------

def is_tensor_name(id: str) -> bool:
    return id.startswith(("arg", "buf", "s")) or id in {"args"}


def extract_tensor_names(node: ast.AST) -> List[str]:
    """Collect Name ids that look like tensors from a Call or expression."""
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and is_tensor_name(n.id):
            names.append(n.id)
    return names


def classify_stmt(stmt: ast.AST) -> Optional[Op]:
    """Turn an AST statement into an Op if it is interesting."""
    lineno = getattr(stmt, "lineno", -1)

    # Delete
    if isinstance(stmt, ast.Delete):
        targets = []
        for t in stmt.targets:
            if isinstance(t, ast.Name):
                targets.append(t.id)
        return Op("del", lineno, tensors=targets, raw=stmt)

    # Assign
    if isinstance(stmt, ast.Assign):
        # Simple name = ...
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            val = stmt.value

            # empty_strided_cuda / empty_strided_* / reinterpret_tensor
            if isinstance(val, ast.Call):
                func = val.func
                fname = None
                if isinstance(func, ast.Name):
                    fname = func.id
                elif isinstance(func, ast.Attribute):
                    fname = func.attr

                if fname and ("empty_strided" in fname or fname == "reinterpret_tensor"):
                    return Op("alloc", lineno, name=target,
                              tensors=extract_tensor_names(val) + [target],
                              raw=stmt)

            # reuse pattern: bufY = bufX
            if isinstance(val, ast.Name) and is_tensor_name(val.id):
                return Op("alloc", lineno, name=target,  # treat as alias/alloc
                          tensors=[val.id, target], raw=stmt)

            return Op("other", lineno, name=target, tensors=extract_tensor_names(stmt), raw=stmt)

        # Tuple unpacking of args (the very first statement)
        return Op("other", lineno, tensors=extract_tensor_names(stmt), raw=stmt)

    # Expr – possible kernel.run(...)
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "run":
            # triton_xxx.run(...) or MultiKernelCall.run(...)
            kernel_name = None
            if isinstance(func.value, ast.Name):
                kernel_name = func.value.id
            tensors = extract_tensor_names(call)
            return Op("kernel", lineno, name=kernel_name or "unknown",
                      tensors=tensors, raw=stmt)

        # extern_kernels.xxx(...)
        if isinstance(func, ast.Attribute):
            if isinstance(func.value, ast.Name) and func.value.id == "extern_kernels":
                tensors = extract_tensor_names(call)
                return Op("kernel", lineno, name=f"extern.{func.attr}",
                          tensors=tensors, raw=stmt)

    return None


def flatten_body(stmts: List[ast.AST]) -> List[ast.AST]:
    """Recursively flatten With / For / If bodies (we only care about linear order)."""
    out = []
    for s in stmts:
        if isinstance(s, (ast.With, ast.For, ast.If)):
            out.extend(flatten_body(s.body))
            if isinstance(s, ast.If) and s.orelse:
                out.extend(flatten_body(s.orelse))
        else:
            out.append(s)
    return out


def extract_ops(tree: ast.AST) -> List[Op]:
    """Step 1: walk the main call() and produce a linear list of Ops."""
    call = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            call = node
            break
    if call is None:
        raise RuntimeError("No top-level def call(args) found")

    # The real work lives inside with torch.cuda._DeviceGuard(...)
    flat = []
    for stmt in call.body:
        if isinstance(stmt, ast.With):
            flat.extend(flatten_body(stmt.body))
        else:
            flat.append(stmt)

    ops: List[Op] = []
    for stmt in flat:
        op = classify_stmt(stmt)
        if op is not None:
            ops.append(op)
    return ops


# ---------------------------------------------------------------------------
# Step 2 – Recover read / write sets
# ---------------------------------------------------------------------------

def recover_rw(ops: List[Op]) -> List[KernelInfo]:
    """
    Heuristic:
    - Maintain the set of buffers that have been allocated but never written.
    - When a kernel first appears with such a buffer → it is a write.
    - Subsequent appearances of a buffer → read (unless it is also a fresh alloc
      in the same kernel, which is rare).
    This matches the typical Inductor pattern: allocate → immediately write.
    """
    allocated_unwritten: Set[str] = set()
    last_writer: Dict[str, int] = {}          # buffer → kernel index
    kernels: List[KernelInfo] = []

    for op in ops:
        if op.kind == "alloc" and op.name:
            # New allocation or alias
            allocated_unwritten.add(op.name)
            # If it is a pure alias (bufY = bufX) we still treat the new name
            # as a fresh buffer that needs a writer.

        elif op.kind == "del":
            for t in op.tensors:
                allocated_unwritten.discard(t)

        elif op.kind == "kernel":
            kidx = len(kernels)
            info = KernelInfo(idx=kidx, name=op.name or "?", lineno=op.lineno)

            # tensors that appear in the call
            seen = set(op.tensors)

            # writes = those that are still unwritten (first use after alloc)
            writes = seen & allocated_unwritten
            reads = seen - writes

            # Also treat any buffer that has a previous writer as a read
            # (even if somehow still in allocated_unwritten – safety)
            for t in list(writes):
                if t in last_writer:
                    reads.add(t)
                    writes.discard(t)

            info.reads = reads
            info.writes = writes

            # update state
            for t in writes:
                allocated_unwritten.discard(t)
                last_writer[t] = kidx
            for t in reads:
                # still keep them as written
                pass

            kernels.append(info)

    return kernels


# ---------------------------------------------------------------------------
# Step 3 – Build dependence DAG
# ---------------------------------------------------------------------------

def build_dag(kernels: List[KernelInfo]) -> None:
    """Add pred / succ edges based on buffer producer-consumer."""
    last_writer: Dict[str, int] = {}

    for k in kernels:
        # edges from previous writers of the reads
        for b in k.reads:
            if b in last_writer:
                pred = last_writer[b]
                k.preds.add(pred)
                kernels[pred].succs.add(k.idx)

        # update last writers
        for b in k.writes:
            last_writer[b] = k.idx


# ---------------------------------------------------------------------------
# Step 4 – Multi-stream scheduling
# ---------------------------------------------------------------------------

def topological_levels(kernels: List[KernelInfo]) -> None:
    """Compute longest-path level (critical path distance from sources)."""
    indeg = {k.idx: len(k.preds) for k in kernels}
    q = deque([k.idx for k in kernels if not k.preds])
    for i in q:
        kernels[i].level = 0

    while q:
        u = q.popleft()
        for v in kernels[u].succs:
            kernels[v].level = max(kernels[v].level, kernels[u].level + 1)
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)


def schedule(kernels: List[KernelInfo], num_streams: int = 4) -> None:
    """
    Greedy list scheduling:
    - Prefer the stream of the heaviest predecessor (affinity)
    - Otherwise pick the least-loaded stream.
    """
    topological_levels(kernels)

    # ready = sources
    ready = deque(sorted([k.idx for k in kernels if not k.preds],
                         key=lambda i: -kernels[i].level))  # critical path first
    finished = set()
    stream_load = [0] * num_streams          # number of kernels assigned
    stream_last_level = [-1] * num_streams

    while ready:
        u = ready.popleft()
        k = kernels[u]

        # affinity: prefer stream of a predecessor that has the highest level
        best_stream = -1
        best_pred_level = -1
        for p in k.preds:
            if kernels[p].stream >= 0 and kernels[p].level > best_pred_level:
                best_pred_level = kernels[p].level
                best_stream = kernels[p].stream

        if best_stream < 0:
            # least loaded
            best_stream = min(range(num_streams), key=lambda s: stream_load[s])

        k.stream = best_stream
        stream_load[best_stream] += 1
        stream_last_level[best_stream] = max(stream_last_level[best_stream], k.level)
        finished.add(u)

        # unlock successors
        for v in sorted(k.succs, key=lambda i: -kernels[i].level):
            if all(p in finished for p in kernels[v].preds):
                if v not in ready:
                    ready.append(v)


def independent_groups_at_level(kernels: List[KernelInfo]) -> Dict[int, List[List[int]]]:
    """
    For each topological level, return the groups of kernels that can run
    concurrently (they have no mutual dependence, which is automatic at the
    same level).
    """
    by_level: Dict[int, List[int]] = defaultdict(list)
    for k in kernels:
        by_level[k.level].append(k.idx)

    # At the same level they are independent by construction of longest-path levels
    return {lvl: [idxs] for lvl, idxs in by_level.items()}  # one group per level


# ---------------------------------------------------------------------------
# Step 5 – Emit analysis report + skeleton
# ---------------------------------------------------------------------------

def report(kernels: List[KernelInfo], num_streams: int = 4) -> str:
    lines = []
    lines.append("=" * 72)
    lines.append("Torch Inductor Multi-Stream Analysis Report")
    lines.append("=" * 72)
    lines.append(f"Total kernels          : {len(kernels)}")
    if not kernels:
        return "\n".join(lines)

    max_level = max(k.level for k in kernels)
    lines.append(f"Critical-path length   : {max_level + 1} levels")
    lines.append(f"Suggested #streams     : {num_streams}")

    # Stream occupancy
    stream_counts = [0] * num_streams
    for k in kernels:
        if 0 <= k.stream < num_streams:
            stream_counts[k.stream] += 1
    lines.append(f"Kernels per stream     : {stream_counts}")

    # Independent kernels at the widest level
    by_level = defaultdict(list)
    for k in kernels:
        by_level[k.level].append(k)
    widest = max(by_level.items(), key=lambda x: len(x[1]))
    lines.append(f"Widest parallel level  : level {widest[0]} with {len(widest[1])} kernels")
    lines.append("")
    lines.append("Sample of independent kernels that can run concurrently")
    lines.append("(first 12 of the widest level):")
    for k in widest[1][:12]:
        lines.append(f"  [{k.idx:4d}] {k.name:<40s}  writes={sorted(k.writes)[:3]} ...")

    lines.append("")
    lines.append("-" * 72)
    lines.append("Suggested stream assignment (kernel_idx → stream)")
    lines.append("-" * 72)
    # compact representation
    assignment = [k.stream for k in kernels]
    lines.append(str(assignment[:80]) + (" ..." if len(assignment) > 80 else ""))

    lines.append("")
    lines.append("=" * 72)
    lines.append("Skeleton for a multi-stream rewrite of call() (illustrative)")
    lines.append("=" * 72)
    lines.append("""
# --- inside the DeviceGuard ---
NUM_STREAMS = 4
streams = [torch.cuda.Stream() for _ in range(NUM_STREAMS)]
# event that makes a buffer ready: buffer_name → (stream_idx, event)
buffer_event = {}

def wait_for(buf, target_stream_idx):
    if buf in buffer_event:
        src_s, ev = buffer_event[buf]
        if src_s != target_stream_idx:
            streams[target_stream_idx].wait_event(ev)

# Then for each kernel k (in the original topological order):
#   s = assignment[k.idx]
#   for r in k.reads:
#       wait_for(r, s)
#   with torch.cuda.stream(streams[s]):
#       triton_xxx.run(..., stream=streams[s].cuda_stream)   # or get_raw_stream
#   ev = streams[s].record_event()
#   for w in k.writes:
#       buffer_event[w] = (s, ev)
#
# Finally synchronize all streams before the return.
for s in streams:
    s.synchronize()
""")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(path: str):
    print(f"Parsing {path} ...")
    src = Path(path).read_text()
    tree = ast.parse(src)

    print("Step 1: extracting operations via AST ...")
    ops = extract_ops(tree)
    print(f"  → {len(ops)} interesting statements "
          f"({sum(1 for o in ops if o.kind=='kernel')} kernels, "
          f"{sum(1 for o in ops if o.kind=='alloc')} allocs)")

    print("Step 2: recovering read/write sets ...")
    kernels = recover_rw(ops)
    print(f"  → {len(kernels)} kernels with R/W information")

    # quick sanity
    total_writes = sum(len(k.writes) for k in kernels)
    total_reads = sum(len(k.reads) for k in kernels)
    print(f"  → total write mentions: {total_writes}, read mentions: {total_reads}")

    print("Step 3: building dependence DAG ...")
    build_dag(kernels)
    n_edges = sum(len(k.succs) for k in kernels)
    print(f"  → {n_edges} dependence edges")

    print("Step 4: multi-stream scheduling (4 streams) ...")
    schedule(kernels, num_streams=4)

    print("Step 5: generating report ...")
    print()
    print(report(kernels, num_streams=4))

    # also dump a machine-readable summary
    summary_path = Path("./kernel_schedule_summary.txt")
    with summary_path.open("w") as f:
        f.write(f"# kernel_idx  stream  level  name  n_reads  n_writes\n")
        for k in kernels:
            f.write(f"{k.idx:5d}  {k.stream:2d}  {k.level:3d}  {k.name:<45s}  "
                    f"{len(k.reads):3d}  {len(k.writes):3d}\n")
    print(f"\nDetailed schedule written to {summary_path}")


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "./model.py"
    main(model)

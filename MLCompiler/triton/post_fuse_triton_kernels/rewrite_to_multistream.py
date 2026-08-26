#!/usr/bin/env python3
"""
Rewrite the Inductor-generated model.py so that independent kernels
are launched on multiple CUDA streams (using the analysis from
analyze_independent_kernels.py).

This produces model_multistream.py with:
- Multiple streams created once
- Event-based cross-stream synchronization
- Each kernel launched on its assigned stream
- Original memory planning / dels / reallocs preserved

The rewrite is source-level (line-based) so formatting and comments stay intact.
"""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Re-use the analysis classes / functions (copied + simplified for self-contained)
# ---------------------------------------------------------------------------

@dataclass
class Op:
    kind: str
    lineno: int
    name: Optional[str] = None
    tensors: List[str] = field(default_factory=list)
    end_lineno: int = -1


@dataclass
class KernelInfo:
    idx: int
    name: str
    lineno: int
    end_lineno: int
    reads: Set[str] = field(default_factory=set)
    writes: Set[str] = field(default_factory=set)
    preds: Set[int] = field(default_factory=set)
    succs: Set[int] = field(default_factory=set)
    stream: int = 0
    level: int = 0


def is_tensor_name(id: str) -> bool:
    return id.startswith(("arg", "buf")) or id.startswith("s") and id[1:].isdigit()


def extract_tensor_names(node: ast.AST) -> List[str]:
    names = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and is_tensor_name(n.id):
            names.append(n.id)
    return list(dict.fromkeys(names))  # unique, preserve order


def flatten_body(stmts):
    out = []
    for s in stmts:
        if isinstance(s, (ast.With, ast.For, ast.If)):
            out.extend(flatten_body(s.body))
            if isinstance(s, ast.If) and s.orelse:
                out.extend(flatten_body(s.orelse))
        else:
            out.append(s)
    return out


def extract_ops_and_kernels(tree: ast.AST) -> Tuple[List[Op], List[KernelInfo]]:
    call = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call")

    flat = []
    for stmt in call.body:
        if isinstance(stmt, ast.With):
            flat.extend(flatten_body(stmt.body))
        else:
            flat.append(stmt)

    ops: List[Op] = []
    for stmt in flat:
        lineno = getattr(stmt, "lineno", -1)
        end_lineno = getattr(stmt, "end_lineno", lineno)

        if isinstance(stmt, ast.Delete):
            targets = [t.id for t in stmt.targets if isinstance(t, ast.Name)]
            ops.append(Op("del", lineno, tensors=targets, end_lineno=end_lineno))
            continue

        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0].id
            val = stmt.value
            if isinstance(val, ast.Call):
                fname = None
                if isinstance(val.func, ast.Name):
                    fname = val.func.id
                elif isinstance(val.func, ast.Attribute):
                    fname = val.func.attr
                if fname and ("empty_strided" in (fname or "") or fname == "reinterpret_tensor"):
                    ops.append(Op("alloc", lineno, name=target,
                                  tensors=extract_tensor_names(val) + [target],
                                  end_lineno=end_lineno))
                    continue
            if isinstance(val, ast.Name) and is_tensor_name(val.id):
                ops.append(Op("alloc", lineno, name=target, tensors=[val.id, target], end_lineno=end_lineno))
                continue
            ops.append(Op("other", lineno, name=target, tensors=extract_tensor_names(stmt), end_lineno=end_lineno))
            continue

        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            func = call.func
            if isinstance(func, ast.Attribute) and func.attr == "run":
                kname = func.value.id if isinstance(func.value, ast.Name) else "unknown"
                ops.append(Op("kernel", lineno, name=kname,
                              tensors=extract_tensor_names(call), end_lineno=end_lineno))
                continue
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "extern_kernels":
                ops.append(Op("kernel", lineno, name=f"extern.{func.attr}",
                              tensors=extract_tensor_names(call), end_lineno=end_lineno))
                continue

    # Now recover R/W and build KernelInfo list (same heuristic as analyzer)
    allocated_unwritten: Set[str] = set()
    last_writer: Dict[str, int] = {}
    kernels: List[KernelInfo] = []

    for op in ops:
        if op.kind == "alloc" and op.name:
            allocated_unwritten.add(op.name)
        elif op.kind == "del":
            for t in op.tensors:
                allocated_unwritten.discard(t)
        elif op.kind == "kernel":
            kidx = len(kernels)
            info = KernelInfo(idx=kidx, name=op.name or "?", lineno=op.lineno, end_lineno=op.end_lineno)
            seen = set(op.tensors)
            writes = seen & allocated_unwritten
            reads = seen - writes
            for t in list(writes):
                if t in last_writer:
                    reads.add(t)
                    writes.discard(t)
            info.reads = reads
            info.writes = writes
            for t in writes:
                allocated_unwritten.discard(t)
                last_writer[t] = kidx
            kernels.append(info)

    # DAG
    last_writer.clear()
    for k in kernels:
        for b in k.reads:
            if b in last_writer:
                pred = last_writer[b]
                k.preds.add(pred)
                kernels[pred].succs.add(k.idx)
        for b in k.writes:
            last_writer[b] = k.idx

    # levels + schedule (improved balancing)
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

    NUM_STREAMS = 4
    ready = deque(sorted([k.idx for k in kernels if not k.preds], key=lambda i: -kernels[i].level))
    finished = set()
    stream_load = [0] * NUM_STREAMS

    while ready:
        u = ready.popleft()
        k = kernels[u]

        # Prefer a stream that already has a predecessor (affinity)
        # but among those, or if none, pick the least loaded.
        candidate_streams = set()
        for p in k.preds:
            if kernels[p].stream >= 0:
                candidate_streams.add(kernels[p].stream)
        if candidate_streams:
            best = min(candidate_streams, key=lambda s: stream_load[s])
        else:
            best = min(range(NUM_STREAMS), key=lambda s: stream_load[s])

        k.stream = best
        stream_load[best] += 1
        finished.add(u)

        for v in sorted(k.succs, key=lambda i: -kernels[i].level):
            if all(p in finished for p in kernels[v].preds):
                if v not in ready and v not in finished:
                    ready.append(v)

    return ops, kernels


# ---------------------------------------------------------------------------
# Source-level rewrite
# ---------------------------------------------------------------------------

def rewrite(src_path: Path, out_path: Path, num_streams: int = 4):
    print(f"Reading {src_path} ...")
    original = src_path.read_text().splitlines(keepends=True)

    print("Parsing + analyzing ...")
    tree = ast.parse("".join(original))
    ops, kernels = extract_ops_and_kernels(tree)
    print(f"  {len(kernels)} kernels scheduled onto {num_streams} streams")
    stream_counts = [0] * num_streams
    for k in kernels:
        stream_counts[k.stream] += 1
    print(f"  occupancy: {stream_counts}")

    # Map from starting lineno of a kernel.run statement → KernelInfo
    kernel_by_lineno: Dict[int, KernelInfo] = {k.lineno: k for k in kernels}

    # We will rebuild the file line by line, injecting code at strategic points.
    # Strategy:
    # 1. Right after the first `with torch.cuda._DeviceGuard(0):` we inject
    #    stream creation + helper.
    # 2. For every kernel launch we find the `stream0 = get_raw_stream(0)` 
    #    that precedes it (or the .run itself) and replace the launch with
    #    multi-stream version.
    # Because exact textual matching of the whole launch is fragile, we do a
    # conservative injection: before each kernel we emit wait + stream switch
    # comments + a runtime lookup, and we change `stream=stream0` to a
    # stream variable that we manage.

    # Simpler robust approach used here:
    # - Inject multi-stream setup once at the beginning of the first With.
    # - Replace every occurrence of `stream=stream0` that belongs to a known
    #   kernel with `stream=streams[STREAM_ID].cuda_stream` (or get_raw_stream
    #   of that stream).
    # - Before the .run of a kernel that has cross-stream reads we insert
    #   wait_event calls.

    # Collect injection points (line index → text to insert *before* that line)
    injections: Dict[int, List[str]] = defaultdict(list)

    # 1. Find the first With and inject setup right after its header
    first_with_lineno = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "call":
            for stmt in node.body:
                if isinstance(stmt, ast.With):
                    first_with_lineno = stmt.lineno
                    break
            break

    if first_with_lineno is None:
        raise RuntimeError("Could not find DeviceGuard With")

    # Inject the multi-stream setup just *before* the first With so that
    # `streams` and the helpers live in the `call` function scope.
    setup_lines = [
        "    # === Multi-stream support (auto-generated) ===",
        f"    _NUM_STREAMS = {num_streams}",
        "    streams = [torch.cuda.Stream() for _ in range(_NUM_STREAMS)]",
        "    # buffer_name -> (stream_idx, event)",
        "    _buffer_event = {}",
        "    def _wait_for_buffers(bufs, target_s):",
        "        for b in bufs:",
        "            if b in _buffer_event:",
        "                src_s, ev = _buffer_event[b]",
        "                if src_s != target_s:",
        "                    streams[target_s].wait_event(ev)",
        "    def _record_writes(bufs, s_idx):",
        "        ev = streams[s_idx].record_event()",
        "        for b in bufs:",
        "            _buffer_event[b] = (s_idx, ev)",
        "    # === end multi-stream setup ===",
        "",
    ]
    injections[first_with_lineno] = setup_lines

    # 2. For every kernel:
    #    - insert _wait_for_buffers *before* the kernel statement (at lineno)
    #    - insert _record_writes *after* the whole statement (at end_lineno)
    for k in kernels:
        if k.reads:
            reads_list = sorted(k.reads)
            # inject at the start of the kernel statement
            injections[k.lineno].insert(0, f"_wait_for_buffers({reads_list!r}, {k.stream})")

        if k.writes:
            writes_list = sorted(k.writes)
            # inject after the last line of the kernel statement
            injections[k.end_lineno].append(
                f"_record_writes({writes_list!r}, {k.stream})"
            )

    # 3. Also force every `stream=stream0` that appears on a kernel line to use the right stream.
    # We do a textual pass for that.

    print("Building rewritten source ...")
    new_lines: List[str] = []
    i = 0
    n = len(original)
    kernel_linenos = set(kernel_by_lineno.keys())
    # Lines that should receive post-injections (record_writes)
    post_injection_linenos = {k.end_lineno for k in kernels}

    while i < n:
        line = original[i]
        lineno = i + 1  # 1-based

        m = re.match(r"^(\s*)", line)
        base_indent = m.group(1) if m else "        "

        # ---- pre-injections (waits + setup) ----
        if lineno in injections:
            for extra in injections[lineno]:
                extra = extra.rstrip("\n")
                # setup block already has correct absolute indent
                if extra.startswith("    # ===") or extra.startswith("    _NUM") or \
                   extra.startswith("    streams") or extra.startswith("    # buffer") or \
                   extra.startswith("    _buffer") or extra.startswith("    def _") or \
                   extra.startswith("        for b") or extra.startswith("            if") or \
                   extra.startswith("                ") or extra.startswith("        ev =") or \
                   extra.startswith("            _buffer") or extra.startswith("    # === end") or \
                   extra == "":
                    new_lines.append(extra + "\n")
                else:
                    # only the _wait_for_buffers belong here as pre-injections
                    if extra.lstrip().startswith("_wait_for_buffers"):
                        new_lines.append(base_indent + extra.lstrip() + "\n")

        # ---- original line (possibly rewritten) ----
        if lineno in kernel_linenos:
            k = kernel_by_lineno[lineno]
            new_line = re.sub(
                r"stream\s*=\s*stream0",
                f"stream=streams[{k.stream}].cuda_stream",
                line
            )
            if new_line == line and "stream=" in line:
                new_line = re.sub(
                    r"stream\s*=\s*get_raw_stream\s*\(\s*0\s*\)",
                    f"stream=streams[{k.stream}].cuda_stream",
                    line
                )
            new_lines.append(new_line)
        else:
            if re.search(r"stream0\s*=\s*get_raw_stream\s*\(\s*0\s*\)", line):
                new_lines.append(f"{base_indent}# stream0 = get_raw_stream(0)  # disabled – using multi-stream\n")
            else:
                new_lines.append(line)

        # ---- post-injections (record_writes) after the kernel statement ----
        if lineno in post_injection_linenos and lineno in injections:
            for extra in injections[lineno]:
                extra = extra.rstrip("\n")
                if extra.lstrip().startswith("_record_writes"):
                    new_lines.append(base_indent + extra.lstrip() + "\n")

        i += 1

    # Final synchronisation before the return of call()
    # Find the return statement and inject before it
    for j in range(len(new_lines) - 1, -1, -1):
        if new_lines[j].lstrip().startswith("return "):
            indent = re.match(r"^(\s*)", new_lines[j]).group(1)
            sync = f"{indent}# Ensure all streams finished before returning results\n"
            sync += f"{indent}for _s in streams:\n"
            sync += f"{indent}    _s.synchronize()\n"
            new_lines.insert(j, sync)
            break

    print(f"Writing {out_path} ...")
    out_path.write_text("".join(new_lines))
    print(f"Done. New file size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    print("IMPORTANT notes for the rewritten model:")
    print("  1. The rewrite is a best-effort source transformation.")
    print("  2. Test numerical equivalence on a few real inputs before production use.")
    print("  3. Some kernels that share buffers in complicated reuse patterns may")
    print("     need extra manual event fences.")
    print("  4. Performance gain will be largest on the early wide levels (~95-way).")
    print("  5. You may want to increase/decrease NUM_STREAMS (currently 4).")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "./model.py")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "./model_multistream.py")
    rewrite(src, dst, num_streams=4)

# mbarrier
## Counters
A mbarrier maintains the following counters:

State                          | Description
-------------------------------| ---------------------------------------------------------------------------------------------------
Phase bit                      | 1-bit value (0 or 1) indicating the current phase
Pending arrival count          | How many arrivals are still needed for the current phase
Expected arrival count         | The value set by mbarrier.init. Used to reload the pending arrival count when a phase completes
Pending transaction-byte count | How many bytes of asynchronous transactions are still outstanding for the current phase

## Phase Completion
A phase completes only when both of the following conditions become true:
- Pending arrival count reaches 0, and
- Pending transaction-byte count (tx-count) reaches 0.

When that happens the hardware performs the following actions automatically:
- Flips the phase bit (0 → 1 or 1 → 0)
- Reloads the pending arrival count from the expected arrival count that was set at init
- Resets the transaction-byte count to 0

The barrier is now ready for the next phase. This automatic re-arming is why you almost never call mbarrier.init inside a hot loop.

## How to check whether a phase has completed or not
When you execute:

```ptx
mbarrier.try_wait.parity ... [mbar], 0;   // requesting parity 0
```

the hardware essentially asks:

> "Is the barrier's **current phase bit** different from 0?"

- If the phase bit is still **0** → phase 0 has **not** finished yet → return `false`
- If the phase bit is **1** → phase 0 **has** finished (the barrier already moved on) → return `true`

**Why this is safe**

The phase bit **only flips** when both completion conditions of the current phase are satisfied:

- Pending arrival count reaches 0, **and**
- Pending transaction-byte count reaches 0

Therefore, the moment the phase bit changes from 0 → 1, it is a reliable signal that phase 0 has fully completed. The instruction does not need to inspect the counters again; the flipped phase bit itself is the proof.


**Simple mental model**

```text
try_wait.parity(want):
    return (current_phase_bit != want)
```

## A complete example
Here is the complete picture for **re-using one mbarrier while issuing multiple TMAs**.

### Goal

- One mbarrier object lives for the whole kernel.
- Each pipeline iteration may issue **several** TMA copies.
- All copies that belong to the same iteration share the same phase.
- After the phase completes, the mbarrier is automatically re-armed and can be used again.

---

### 1. One-time initialization

```ptx
// Thread 0 only
mbarrier.init.shared.b64 [mbar], 1;   // expect 1 arrival (the leader)

mov.b32 phase, 0;                     // software phase tracker
```

**State after init**
- Phase bit = 0
- Expected arrival count = 1
- Pending arrival count = 1
- Pending transaction-byte count = 0

---

### 2. Reusable loop that issues multiple TMAs per iteration

```ptx
loop:
    // -------------------------------------------------
    // Producer (elected leader thread)
    // -------------------------------------------------
    @leader {
        // Total bytes of all TMAs that belong to this phase
        // e.g. tileA + tileB + tileC
        mbarrier.arrive.expect_tx.shared.b64 %state, [mbar], TOTAL_BYTES;

        // Issue several independent TMA copies.
        // All of them are linked to the same mbarrier.
        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [dstA], [tensorMapA, {x0, y0}], [mbar];

        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [dstB], [tensorMapB, {x1, y1}], [mbar];

        cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
            [dstC], [tensorMapC, {x2, y2}], [mbar];
    }

    // Make sure every thread sees that the TMAs were issued
    bar.sync 0;

    // -------------------------------------------------
    // Consumers wait for the whole phase to finish
    // -------------------------------------------------
wait_loop:
    mbarrier.try_wait.parity.shared.b64 %p, [mbar], phase;
    @!%p bra wait_loop;

    // All tiles (A, B, C) are now safely visible
    // ... use the data ...

    // Prepare for the next iteration (reuse the same mbarrier)
    xor.b32 phase, phase, 1;          // 0 ↔ 1
    // loop back
```

---

### 3. What happens inside one iteration

| Step | Action | Effect on mbarrier |
|------|--------|--------------------|
| 1 | `arrive.expect_tx(TOTAL_BYTES)` | Pending arrival count → 0<br>Pending tx-count → TOTAL_BYTES |
| 2 | Three TMA instructions issued | Each is tied to the same mbarrier via `.mbarrier::complete_tx::bytes` |
| 3 | Each TMA finishes | Hardware does `complete_tx` and subtracts the bytes of that TMA from the pending tx-count |
| 4 | Last TMA finishes | Pending tx-count reaches 0 |
| 5 | Both counters are zero | Phase completes → phase bit flips, pending arrival count reloaded to 1, tx-count cleared to 0 |
| 6 | `try_wait.parity(phase)` | Sees the flipped phase bit → returns true |
| 7 | `phase ^= 1` | Software is now ready for the next phase on the **same** mbarrier |

---

### 4. Key rules when multiple TMAs share one phase

- The value passed to `arrive.expect_tx` (or the sum of several `expect_tx` calls) **must equal the total number of bytes** that will be transferred by all TMAs of that phase.
- Under-counting → barrier never completes (deadlock).
- Over-counting → barrier may complete before all data has arrived (data race).
- All TMAs of the same phase must use the **same** mbarrier object.
- After the phase completes, the mbarrier is automatically ready for the next iteration; you do **not** call `mbarrier.init` again.

---

### Summary

One mbarrier + phase toggling lets you:

1. Issue any number of TMAs in a phase,
2. Wait once for all of them to finish,
3. Automatically re-arm the barrier,
4. Repeat the whole process for as many iterations as you need.

This is the standard pattern used in high-performance multi-stage TMA pipelines on Hopper and later GPUs.
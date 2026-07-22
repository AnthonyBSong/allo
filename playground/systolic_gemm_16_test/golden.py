"""Golden reference for systolic_gemm_16 (playground/test_allos/systolic.py).

Allo has no CPU lowering for `allo.grid_map`/`allo.stream_global` yet (see
mesh/TODO and playground/README.md's "grid_map in CPU simulator — upstream
Allo; until then use `systolic_cpu.py` as golden"), and this fork's `allo`
package needs its own from-source LLVM/MLIR build to run at all, which isn't
set up in this environment. So instead of executing through Allo's own
compiler, this is a direct Python transcription of the exact per-PE control
flow in `systolic.py`'s `pe()` — same branches, same `Stream.put`/`.get`
call sites, same accumulation order — run as cooperative generators over
explicit FIFO queues standing in for `Stream<T, DEPTH>`. It's the same
pattern `systolic_cpu.py` used for the design that predates this one.

Produces:
  - a golden C for a fixed, non-trivial random A/B pair
  - test_data.h: the same A/B/C as C++ literals, for test_harness.cu to
    diff the real systolic_gemm_16_kernel output against on a CUDA machine
"""

from collections import deque

import numpy as np

M, N, K = 4, 4, 4
P0, P1 = M + 2, N + 2  # 6x6 SPMW grid, matches systolic.py


def pe_program(i, j, A, B, C, fifo_A, fifo_B, progress):
    """Generator mirroring systolic.py's pe() body for one grid coordinate.

    Each `yield` stands in for one potential stall cycle: put() never
    blocks here (every cell receives exactly K=4 puts, matching Stream's
    depth-4 capacity, so it never needs to), get() re-yields until its
    fifo is non-empty -- the same "spin until ready" semantics as
    Stream<T,DEPTH>::get() in utils.cuh. `progress` is a 1-element list
    bumped on every actual put/get (not on a blocking re-yield) -- see
    `run_golden`'s deadlock check, which needs a signal that survives a
    round-robin pass draining a queue as fast as it fills it.
    """
    if i == 0 and j == 0:
        return  # corner halo, no compute
    if i == M + 1 and j == N + 1:
        return  # corner halo, no compute

    if j == 0:  # left edge: inject A downward
        if i != 0 and i != M + 1:
            for k in range(K):
                fifo_A[i][j + 1].append(A[i - 1, k])
                progress[0] += 1
                yield
        return

    if i == 0:  # top edge: inject B rightward
        if j != 0 and j != N + 1:
            for k in range(K):
                fifo_B[i + 1][j].append(B[k, j - 1])
                progress[0] += 1
                yield
        return

    if i == M + 1:  # bottom halo: drain B
        if j != 0 and j != N + 1:
            for k in range(K):
                while not fifo_B[i][j]:
                    yield
                fifo_B[i][j].popleft()
                progress[0] += 1
                yield
        return

    if j == N + 1:  # right halo: drain A
        if i != 0 and i != M + 1:
            for k in range(K):
                while not fifo_A[i][j]:
                    yield
                fifo_A[i][j].popleft()
                progress[0] += 1
                yield
        return

    # interior MAC PEs
    acc = np.float32(0.0)
    for k in range(K):
        while not fifo_A[i][j]:
            yield
        a = fifo_A[i][j].popleft()
        while not fifo_B[i][j]:
            yield
        b = fifo_B[i][j].popleft()
        prod = a * b
        acc = acc + prod
        fifo_A[i][j + 1].append(a)
        fifo_B[i + 1][j].append(b)
        progress[0] += 1
        yield
    C[i - 1, j - 1] = acc


def run_golden(A, B):
    C = np.zeros((M, N), dtype=np.float32)
    fifo_A = [[deque() for _ in range(P1)] for _ in range(P0)]
    fifo_B = [[deque() for _ in range(P1)] for _ in range(P0)]
    progress = [0]

    active = [
        pe_program(i, j, A, B, C, fifo_A, fifo_B, progress)
        for i in range(P0)
        for j in range(P1)
    ]

    # Deadlock check needs actual put/get counts, not queue occupancy: a
    # round-robin pass in dependency order can drain every queue it fills
    # within the same pass (see the row-major-vs-dataflow-direction note
    # above), so occupancy alone reads 0 every round even mid-progress.
    prev_progress = None
    rounds = 0
    while active:
        still_active = []
        for gen in active:
            try:
                next(gen)
                still_active.append(gen)
            except StopIteration:
                pass
        active = still_active
        rounds += 1
        if active and progress[0] == prev_progress:
            raise RuntimeError(
                f"golden PE simulation deadlocked after {rounds} rounds "
                f"({len(active)} PE(s) still active) -- some get() is "
                "spinning on a put() that will never come"
            )
        prev_progress = progress[0]
    return C, rounds


def emit_header(path, A, B, C):
    def literal(arr):
        rows = []
        for r in range(arr.shape[0]):
            vals = ", ".join(f"{float(x):.9g}f" for x in arr[r])
            rows.append(f"    {{ {vals} }}")
        return "{\n" + ",\n".join(rows) + "\n}"

    with open(path, "w") as f:
        f.write("// Auto-generated by golden.py -- do not hand-edit.\n")
        f.write("#pragma once\n\n")
        f.write(f"constexpr int TEST_M = {M};\n")
        f.write(f"constexpr int TEST_N = {N};\n")
        f.write(f"constexpr int TEST_K = {K};\n\n")
        f.write(f"static float test_A[{M}][{K}] = {literal(A)};\n\n")
        f.write(f"static float test_B[{K}][{N}] = {literal(B)};\n\n")
        f.write(f"static float test_C_expected[{M}][{N}] = {literal(C)};\n")


def main():
    rng = np.random.default_rng(seed=1234)
    # Non-trivial values on purpose: signed, non-integer, wide dynamic
    # range -- catches sign bugs and accumulation-order bugs that an
    # all-ones or all-zeros GEMM would hide.
    A = rng.uniform(-6.0, 6.0, size=(M, K)).astype(np.float32)
    B = rng.uniform(-6.0, 6.0, size=(K, N)).astype(np.float32)

    C_golden, rounds = run_golden(A, B)
    C_ref = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)

    print("A =\n", A)
    print("B =\n", B)
    print(f"\nPE-dataflow golden simulation converged in {rounds} rounds.")
    print("C (golden PE simulation) =\n", C_golden)
    print("C (numpy reference, A @ B) =\n", C_ref)

    if not np.allclose(C_golden, C_ref, rtol=1e-5, atol=1e-4):
        diff = np.abs(C_golden - C_ref)
        raise SystemExit(
            f"MISMATCH: golden PE simulation disagrees with numpy A @ B "
            f"(max abs diff {diff.max():.6g})"
        )
    print("\nPASS: golden PE-dataflow simulation matches A @ B.")

    out_path = "test_data.h"
    emit_header(out_path, A, B, C_golden)
    print(f"\nWrote {out_path} for test_harness.cu to compare against "
          f"once run on a CUDA machine.")


if __name__ == "__main__":
    main()

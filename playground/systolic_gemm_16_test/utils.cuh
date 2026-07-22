#pragma once
#include <cstdint>

// Memref<T, Dims...>  --  pointer-wrapping view with multi-dim operator()
// Maps directly to MLIR memref<DxDx...xT>.
template <typename T, int... Dims>
struct Memref {
  T* data;
  static constexpr int total = (Dims * ... * 1);

  template <typename... Idx>
  __host__ __device__ __forceinline__ T& operator()(Idx... idx) {
    return data[linearize(idx...)];
  }
  template <typename... Idx>
  __host__ __device__ __forceinline__ const T& operator()(Idx... idx) const {
    return data[linearize(idx...)];
  }

  __host__ __device__ __forceinline__ void fill(T val) {
    for (int i = 0; i < total; ++i) data[i] = val;
  }

private:
  template <typename... Idx>
  __host__ __device__ __forceinline__ static constexpr int linearize(Idx... idx) {
    constexpr int d[] = {Dims...};
    int idxs[] = {static_cast<int>(idx)...};
    int flat = 0, stride = 1;
    for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
      flat += idxs[i] * stride;
      stride *= d[i];
    }
    return flat;
  }
};

// MemrefLocal<T, Dims...>  --  stack-owning variant for local allocs
// Maps to memref.alloc() with shape (e.g. memref<4xi8>).
template <typename T, int... Dims>
struct MemrefLocal {
  static constexpr int total = (Dims * ... * 1);
  T storage[total];

  template <typename... Idx>
  __host__ __device__ __forceinline__ T& operator()(Idx... idx) {
    return storage[linearize(idx...)];
  }
  template <typename... Idx>
  __host__ __device__ __forceinline__ const T& operator()(Idx... idx) const {
    return storage[linearize(idx...)];
  }

  __host__ __device__ __forceinline__ void fill(T val) {
    for (int i = 0; i < total; ++i) storage[i] = val;
  }

private:
  template <typename... Idx>
  __host__ __device__ __forceinline__ static constexpr int linearize(Idx... idx) {
    constexpr int d[] = {Dims...};
    int idxs[] = {static_cast<int>(idx)...};
    int flat = 0, stride = 1;
    for (int i = sizeof...(Dims) - 1; i >= 0; --i) {
      flat += idxs[i] * stride;
      stride *= d[i];
    }
    return flat;
  }
};

// Stream<T, DEPTH>  --  blocking SPSC FIFO for allo.stream
//
// POD layout on purpose: no in-class member initializers, no user-declared
// constructors. That keeps it usable as a `__shared__` variable (CUDA
// disallows non-trivially-constructible types in shared memory) and avoids
// the implicit zero-init at declaration that nvcc would otherwise emit at
// every call site. Callers MUST invoke `.init()` once before the first
// `put`/`get` -- see `emit_top_megakernel` in codegen/src/emitter.rs and
// birrd_verify.cu for the canonical pattern.

template <typename T, int DEPTH>
struct Stream {
  T buf[DEPTH];
  int head;
  int tail;
  int count;

  __host__ __device__ void init() { head = 0; tail = 0; count = 0; }
  __host__ __device__ bool full()  const { return count == DEPTH; }
  __host__ __device__ bool empty() const { return count == 0; }

  __host__ __device__ void put(T v) {
#ifdef __CUDA_ARCH__
    while (*(volatile int*)&count >= DEPTH) { /* spin until consumer drains */ }
    buf[tail] = v;
    __threadfence_block();
    tail = (tail + 1) % DEPTH;
    atomicAdd(&count, 1);
#else
    buf[tail] = v;
    tail = (tail + 1) % DEPTH;
    ++count;
#endif
  }

  __host__ __device__ T get() {
#ifdef __CUDA_ARCH__
    while (*(volatile int*)&count <= 0) { /* spin until producer publishes */ }
    T v = buf[head];
    __threadfence_block();
    head = (head + 1) % DEPTH;
    atomicSub(&count, 1);
    return v;
#else
    T v = buf[head];
    head = (head + 1) % DEPTH;
    --count;
    return v;
#endif
  }
};
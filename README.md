# FastServe

A from-scratch neural network inference engine, built as a deliberate practice project in **inference engineering**.

FastServe is not a toy “one-off script”. It’s the backbone of a multi‑month journey: starting from a single matrix multiply and growing into a production‑style serving stack with quantization, custom kernels, compiler backends, and real benchmarking.

This repository captures that journey in code.

---

## Why This Exists

Most ML work starts at the top of the stack: Python, high‑level frameworks, cloud services. This project flips that on its head.

The goal is to:

- **Understand inference systems from the metal up** – caches, vector units, memory bandwidth, kernels.
- **Build real, measurable systems every day** – not just read papers.
- **Treat the engine as the resume** – the system you build should prove what you know.

FastServe is my lab notebook in code form. Every stage of the learning plan shows up here as a concrete capability: a faster kernel, a new backend, a better benchmark, a more realistic serving setup.

---

## High‑Level Roadmap

FastServe evolves across a few big phases (summarized from the learning plan):

1. **Foundations (Weeks 1–2)**
   - Cache‑aware and SIMD‑optimized matrix multiply on CPU.
   - First CNN inference engine using im2col + custom GEMM.
   - Early versions of FastServe (v0.x) with CLI, quantization, and pruning.

2. **Advanced Numerics (Weeks 3–4)**
   - INT8 and sub‑INT8 quantization, per‑channel scaling, block quantization.
   - Quantization‑aware training and mixed‑precision strategies.
   - Compression pipeline combining quantization, pruning, and distillation.

3. **Kernel Engineering (Weeks 5–8)**
   - CUDA kernels, shared‑memory tiling, Tensor Cores, Triton.
   - Fused ops (Conv+BN+ReLU, fused attention) and multi‑GPU support.
   - High‑end CPU backends (AVX‑512, AMX, NEON) and multi‑threading.

4. **Production Systems (Weeks 9–16)**
   - REST/gRPC serving, dynamic and adaptive batching.
   - Distributed inference, observability, fault tolerance.
   - CI/CD, Kubernetes deployment, multi‑model serving, security.

5. **Edge & Advanced Topics (Weeks 17–24)**
   - Mobile / edge deployment (TFLite, ONNX Runtime Mobile, TPUs).
   - NAS, dynamic networks, efficient architectures.
   - Custom compilation passes, caching, power and cost optimization.

6. **Integration & Mastery (Weeks 25–26+)**
   - Unifying everything into FastServe v3.x.
   - Picking a specialization: compilers, systems, hardware, or research.

You don’t have to follow this exact schedule to use FastServe, but the codebase is intentionally structured so that each phase has a visible footprint.

---

## Current Snapshot: FastServe v0.1

Right now, this directory contains the **first working inference engine**:

- A tiny CNN for MNIST (Conv → ReLU → MaxPool → Flatten → FC → Softmax).
- Hand‑rolled layers on top of a cache‑aware, AVX2‑accelerated matrix multiply.
- A simple CLI that loads weights and test images from disk and runs inference.

### What It Can Do Today

- Load a small CNN’s weights from a text file.
- Load grayscale digit images (MNIST‑style) and feed them through the network.
- Print predictions, confidence scores, ASCII art of the input, and timing.
- Run a synthetic benchmark loop to measure pure inference latency and throughput.

Example (run from `src/fastserve`):

```bash
make           # build the fastserve binary
./fastserve --model ../models/tiny_cnn_weights.txt \
            --images ../models/test_images.txt \
            --verbose
```

You’ll see something like:

- ASCII visualization of the digit.
- Class probabilities.
- Predicted label + correctness.
- Per‑image latency and overall throughput.

For a pure performance check:

```bash
./fastserve --model ../models/tiny_cnn_weights.txt --benchmark
```

This runs many forward passes on a dummy image and reports average latency and images/sec.

---

## Architecture (v0.1)

The current CPU backend is intentionally simple and explicit. There’s no hidden magic.

- `core/`
  - `tensor.h` – minimal tensor type aliases using nested `std::vector`.
  - `simd_matmul.h` – blocked, AVX2/FMA‑accelerated matrix multiply.

- `nn/`
  - `layers.h/.cpp` – `im2col`, `conv2d`, ReLU, MaxPool2D, Flatten, fully connected layer, softmax.
  - `model.h/.cpp` – `TinyCNN` class: holds weights, runs the full forward pass, tracks parameter counts.
  - `loader.h/.cpp` – utilities to load labeled images and print them as ASCII.

- Top‑level
  - `main.cpp` – CLI entrypoint: argument parsing, inference mode, benchmark mode.
  - `Makefile` – tiny build system for this stage.

The CNN itself is intentionally small but complete:

1. Input: `1 × 28 × 28` grayscale image.
2. Conv2D: `1 → 8` channels, `3×3`, stride 1, no padding.
3. ReLU.
4. MaxPool2D: `2×2` window, stride 2.
5. Flatten → 1352‑dimensional vector.
6. Fully connected: `1352 → 10` logits.
7. Softmax → probabilities over digits 0–9.

Weights and biases live in a plain text file (`models/tiny_cnn_weights.txt`) exported ahead of time. The C++ loader reads them into `TinyCNN`’s tensors.

---

## How This Fits the Bigger Mission

FastServe isn’t about re‑implementing PyTorch. It’s about earning intuition the hard way:

- When Conv2D is slow, you know exactly where the cycles go.
- When you add INT8 later, you already understand how the FP32 kernel works.
- When you deploy to GPU, you have a baseline CPU kernel you wrote yourself.

Each future version will be a small, deliberate step:

- **v0.2–v0.3** – integrate INT8 quantization and per‑channel scaling.
- **v0.4** – play with pruning and sparse formats (and learn why unstructured sparsity is tricky on CPU).
- **v0.5–v0.7** – add CUDA, fusion, Triton, and a richer benchmarking harness.
- **v1.x+** – serving infrastructure, compiler backends, distributed inference, observability.

The rule is simple: **no day ends without a measurable improvement**. That might be:

- Lower latency.
- Higher throughput.
- Less memory.
- Cleaner abstractions.
- Or just a deeper understanding encoded as working code.

---

## How to Use This Repo

If you want to learn along similar lines, you can:

1. **Build and run the current engine**
   - Use it as a baseline for your own experiments.

2. **Fork and extend**
   - Swap in your own model architecture.
   - Try different data layouts or kernels.
   - Add quantization, pruning, or a new backend.

3. **Treat it as a playbook**
   - The `DAILY_LEARNING_PLAN.md` at the root lays out a full 6‑month path.
   - You can adapt it to full‑time, part‑time, or accelerated schedules.

FastServe is intentionally small enough to understand, but serious enough to grow into a real system.

---

## Philosophy & Principles

A few rules I try to follow while building this:

- **Build first, optimize second** – correctness before cleverness.
- **Measure everything** – no “this feels faster” without numbers.
- **Real systems, not toy snippets** – every optimization plugs back into FastServe.
- **Explain it to future me** – code and docs should make sense when I come back months later.

If, months from now, this project has grown into a full inference stack with GPU backends, compiler integration, and production‑ready serving, it will be because these principles held.

---

## What’s Next

Short‑term, the focus is on:

- Cleaning up the CPU backend and adding baseline INT8 support.
- Building out a proper benchmarking harness (latency/throughput across sizes).
- Experimenting with simple pruning and sparse formats.

Longer‑term, the roadmap follows the learning plan:

- GPU kernels, fused ops, and Triton.
- Compiler backends (TensorRT, TVM, ONNX Runtime, XLA).
- Full serving system with batching, observability, and safe deployments.

If you’re reading this and building something similar, feel free to use this as a template, a reference, or just a nudge to start from the metal up.

The journey starts with a single matrix multiply.

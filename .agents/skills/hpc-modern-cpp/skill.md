---
name: hpc-modern-cpp
description: Use for performance-critical modern C++20 work, especially HPC kernels, TT-Metal/TT-Metalium, Tenstorrent dataflow, L1/DRAM placement, sharding, SIMD/SFPU code, memory-layout design, concurrency, benchmarking, or architectural refactors. Do not use for ordinary application code where performance and data movement are not primary concerns.
---

# HPC Modern C++ Engineering

Use this skill when designing, reviewing, optimizing, or rewriting performance-critical C++ code.

The goal is not merely to produce valid or stylistically modern C++. The goal is to produce code whose architecture matches the hardware, whose performance model is explicit, and whose correctness can be verified.

## Core priorities

Optimize in this order:

1. Correctness and preservation of mathematical semantics.
2. Elimination of unnecessary data movement and materialization.
3. Locality, ownership, and communication topology.
4. Parallel decomposition and synchronization.
5. Kernel/vector efficiency.
6. Low-level instruction and micro-optimization.

Do not optimize arithmetic while the design still performs avoidable DRAM passes, global copies, serialization, or redundant recomputation.

## Required workflow

Before editing code:

1. Read the relevant implementation, headers, tests, benchmarks, build files, and architecture notes.
2. Reconstruct the current dataflow:
   - where each object is allocated;
   - who owns it;
   - where it physically resides;
   - which core/thread/kernel reads and writes it;
   - when barriers occur;
   - how many times the full dataset crosses each memory boundary.
3. State the current bottleneck as a concrete cost:
   - bytes transferred;
   - number of full passes;
   - synchronization count;
   - temporary storage;
   - remote accesses;
   - redundant computation.
4. Identify invariants that must remain true.
5. Propose one coherent architecture. Do not present a collection of unrelated micro-patches as a redesign.
6. Implement the smallest complete slice that validates the architecture.
7. Compile, test, benchmark, and report what was actually measured.

When the existing abstraction enforces a poor dataflow, prefer a clean replacement over preserving it through adapters, compatibility layers, rollback paths, or duplicated backends.

## Performance model

For every nontrivial change, reason about:

- asymptotic work;
- bytes read and written;
- memory level used: registers, local SRAM/L1, remote L1, DRAM, host;
- access pattern: contiguous, strided, random, halo, broadcast;
- communication-to-computation ratio;
- number and scope of barriers;
- temporary-buffer footprint;
- expected core occupancy and load balance;
- vector width, alignment, and page/tile/stick granularity.

Use a compact table when useful:

| Resource | Before | After |
|---|---:|---:|
| Full DRAM reads | | |
| Full DRAM writes | | |
| L1 working set/core | | |
| Global barriers | | |
| Remote halo bytes | | |
| Arithmetic work | | |

Never claim a speedup solely from code appearance. Distinguish clearly between:

- measured result;
- analytical expectation;
- unverified hypothesis.

## Architecture rules

### Separate planning from execution

Keep host-side planning and device-side execution distinct.

Host/planner responsibilities:

- validate shapes and limits;
- choose cores, shards, windows, and execution mode;
- calculate offsets, lengths, halos, and ownership;
- select compile-time kernel variants;
- construct immutable execution metadata.

Kernel responsibilities:

- execute the already-decided dataflow;
- avoid policy decisions inside hot loops;
- use simple, explicit state;
- communicate only what the plan requires.

Do not make every element access dynamically decide between multiple memory sources when the source mode can be selected once per route, window, kernel, or program.

### Prefer ownership-based decomposition

Use an owner-computes model where possible:

- each output interval has one owner;
- writes are local to the owner;
- source body is local when practical;
- only bounded halo data is remote;
- no two workers write the same output region;
- route completion is synchronized only when a later dependency requires it.

Partition according to physical layout, not merely equal arithmetic counts.

### Keep intermediates near compute

Prefer:

```text
input DRAM -> local/sharded working set -> all dependent compute -> final DRAM
```

over:

```text
DRAM -> compute step -> DRAM -> compute step -> DRAM
```

If the full problem does not fit local memory, use a mathematically justified window, tile, or dependency-cone decomposition. Do not automatically spill every intermediate after every stage.

### Make execution modes coarse-grained

Good:

- `ResidentSharded`
- `Windowed`
- `DependencyConeStreamed`

Bad:

- per-value source switches;
- implicit cache coherence;
- mutable global source state;
- hidden fallback reads inside accessors;
- a circular buffer simultaneously acting as cache, queue, and authoritative storage.

Select the mode outside the hot path and make kernel contracts explicit.

## Modern C++ rules

Target the repository's configured standard; prefer C++20 where available.

Use:

- RAII for host-side resources;
- `enum class` for state and modes;
- narrow structs with explicit invariants;
- `std::span` for non-owning contiguous views;
- `std::array` for fixed-size data;
- `constexpr` and `consteval` for compile-time geometry and configuration;
- concepts or constrained templates only when they simplify contracts;
- `[[nodiscard]]` for values that must be checked;
- strong types when raw integers could mix addresses, lengths, offsets, core IDs, or bytes;
- explicit signed/unsigned conversions at validated boundaries.

Avoid in performance-sensitive paths:

- hidden heap allocation;
- `std::function`;
- virtual dispatch;
- exceptions as normal control flow;
- owning smart pointers when ownership is not shared;
- unnecessary reference counting;
- type erasure;
- recursive template machinery with no measured benefit;
- ranges pipelines that obscure generated work;
- containers whose allocation or traversal cost is unknown.

Do not use “modern C++” as a reason to hide hardware-relevant details. Zero-cost abstractions are acceptable only when their generated behavior is clear.

## Data-oriented design

Prefer structures that match execution:

- immutable configuration separated from mutable runtime state;
- compact metadata arrays;
- contiguous storage;
- structure-of-arrays when fields are consumed independently;
- fixed-capacity storage when the maximum is known;
- explicit alignment and padding;
- page/tile/stick counts stored separately from logical element counts.

Name units in identifiers when ambiguity exists:

```cpp
element_count
stick_count
page_bytes
l1_addr
dram_addr
source_offset_elements
```

Do not reuse a single integer field for multiple units or meanings.

## Hot-loop rules

Inside hot loops:

- hoist loop invariants;
- prefer compile-time specialization over repeated branching;
- batch DMA/NoC operations before one barrier;
- avoid one barrier per element/page when a batch barrier is sufficient;
- avoid repeated address reconstruction;
- make aliasing assumptions explicit;
- keep live state small;
- avoid dynamic allocation and general-purpose containers;
- preserve vector-friendly contiguous access;
- use unrolling only when register pressure and code size remain acceptable;
- do not add branches for rare cases when prologue/body/epilogue separation is cleaner.

Inspect generated assembly or device disassembly when instruction selection, spills, vectorization, or register pressure is uncertain.

## Memory and alignment

Treat alignment, page size, tile shape, stick width, and NoC granularity as correctness constraints, not comments.

For each transfer, establish:

- source alignment;
- destination alignment;
- transfer-size alignment;
- valid logical length;
- physical padded length;
- whether the final partial page is zero-filled, masked, or ignored.

Use checked helpers for rounding and size arithmetic. Prevent overflow before narrowing to device-width integers.

Do not assume logical contiguity implies physical contiguity in interleaved or sharded layouts.

## Synchronization and concurrency

Every barrier must have a named dependency.

For each synchronization point, be able to answer:

- which writes must become visible;
- which readers depend on them;
- whether the barrier is local, per-core, per-route, or global;
- whether double buffering or ownership removes the dependency;
- whether read and write NoCs can overlap.

Avoid atomics unless the algorithm genuinely requires shared updates. Prefer partitioning that eliminates write conflicts.

Do not remove a synchronization point without proving visibility and lifetime safety.

## Numerical correctness

Preserve the reference mathematical model.

For floating-point code:

- state the required format and accumulation precision;
- do not silently change FP32 to BF16/TF32/FP16;
- preserve operation ordering when reproducibility matters;
- document when reassociation or fused operations may change rounding;
- test NaN, infinity, denormal, zero, large magnitude, and boundary values when relevant;
- use tolerant comparisons with a justified absolute/relative tolerance.

Optimization is invalid if it changes stream shifts, valid intervals, boundary extension, coefficient order, or output canonicalization.

## Tenstorrent and TT-Metal guidance

When working with TT-Metal/TT-Metalium:

- reason explicitly about DRAM, local L1, remote L1, circular buffers, destination registers, and NoC traffic;
- use sharded L1 for resident intermediates when capacity permits;
- align shard ownership with output work;
- keep ordinary body accesses local and isolate halo traffic;
- batch `noc_async_read`/`noc_async_write` operations and minimize barriers;
- treat circular buffers primarily as producer-consumer transport, not as an implicit coherent cache;
- keep SFPU/SFPI kernels focused on arithmetic and register scheduling;
- do not compensate for poor host/dataflow architecture inside the SFPU body;
- account for CB footprint, runtime-reserved L1, kernel code, and safety margin before declaring a working set resident;
- use compile-time accessors or separate kernel variants when memory layouts differ materially.

For FP32 SFPU work:

- watch destination-register capacity and local-register pressure;
- minimize long-lived temporaries;
- avoid compiler spills;
- separate prologue, steady-state body, and epilogue;
- validate face/tile register mapping before optimizing shuffles.

## Repository evidence map for tt-wavelet

Before proposing or implementing a tt-wavelet optimization, inspect the repository sources in this order.

### Project semantics and mathematical model

Treat the project repository as the source of truth:

- `docs/`
  - mathematical definitions;
  - stream shifts and valid intervals;
  - lifting-step semantics;
  - horizontal stencil and shift/register mapping;
  - boundary and canonicalization rules.
- `wavelets/`
  - JSON lifting schemes;
  - actual predict/update/scale/swap sequences;
  - coefficient lengths;
  - shifts;
  - even/odd delays;
  - scheme-level numerical metadata.
- current implementation and tests
  - actual host planner;
  - route geometry;
  - reader/compute/writer contracts;
  - reference output behavior.

Do not design from one example JSON. Scan all scheme files and generate a summary containing at least:

- number of schemes;
- minimum/maximum predict-update count;
- coefficient-length distribution;
- shift range;
- swap count;
- whether scale-even and scale-odd are terminal;
- maximum derived dependency halo;
- schemes that violate common structural assumptions.

The JSON sequence is authoritative. `tap_size` is useful for validation and conservative padding, but dependency-cone intervals must be derived from the actual ordered steps, shifts, coefficient lengths, and swaps.

### Tenstorrent low-level documentation

Inspect the locally cloned `tt-isa-documentation/` repository when working on:

- SFPU instruction behavior;
- destination-register and local-register layout;
- tile/face/lane mapping;
- shuffle and shift instructions;
- synchronization or hardware status;
- instruction limitations not exposed clearly through SFPI.

Also inspect the repository's SFPI/SFPU documentation and examples before inventing register behavior.

Prefer SFPI or established TT instruction wrappers. Do not emit raw assembly unless the existing toolchain requires it and generated instructions have been verified.

### TT-Metal convolution references

Use TT-Metal convolution code as a reference for architecture and planning, especially:

- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_utils.cpp`
  - sharding selection;
  - padded shapes;
  - per-core blocking;
  - L1 usage estimation;
  - execution-path selection.
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv2d/conv2d_op_program_factory_common.cpp`
  - CB sizing;
  - globally allocated sharded CBs;
  - CB overlap;
  - transfer-cost modeling;
  - reader-splitting decisions.
- `tt-metal/ttnn/cpp/ttnn/operations/conv/conv_transpose2d/conv_transpose2d.cpp`
  - output-slice to input-slice mapping;
  - halo derivation;
  - L1-full versus DRAM-sliced execution;
  - slice-local execution followed by final DRAM write.
- `tt-metal/ttnn/cpp/ttnn/operations/sliding_window/`
  - halo and window planning;
  - slice geometry;
  - sharded data movement.

Use these files to learn patterns, not to force LWT into the conv2d implementation.

LWT predict/update is a small 1D shifted stencil with parity-stream dependencies and FP32 SFPU arithmetic. A generic conv2d path may introduce inappropriate im2col, weight preparation, tilization, matmul, channel blocking, and TTNN dispatch overhead.

Before reusing a convolution component, state exactly which part is being reused:

- planner mathematics;
- sharding helper;
- halo transfer pattern;
- CB allocation technique;
- data-movement kernel;
- compute primitive.

Do not call a high-level conv2d operator merely because lifting contains a convolution-like sum.

## Dependency-cone planning for lifting schemes

A dependency cone is valid for any finite lifting scheme composed of predict, update, scale, and swap steps.

For a required final interval, traverse the ordered scheme backward.

Let `E` and `O` be required logical intervals for the current even and odd streams.

For a forward predict step

```text
O' = O + P(E)
E' = E
```

with source offsets `[shift, shift + coefficient_count - 1]`, back-propagate:

```text
required_old_O = required_new_O
required_old_E = union(
    required_new_E,
    required_new_O + [shift, shift + coefficient_count - 1])
```

For a forward update step

```text
E' = E + U(O)
O' = O
```

back-propagate:

```text
required_old_E = required_new_E
required_old_O = union(
    required_new_O,
    required_new_E + [shift, shift + coefficient_count - 1])
```

For scale:

```text
required_old_stream = required_new_stream
```

Scale does not enlarge the cone.

For swap:

```text
swap(required_E, required_O)
```

The planner must compute these intervals from each JSON scheme and validate them against the existing forward geometry.

### Choosing between resident and cone-streamed execution

Use coarse execution modes:

1. `ResidentSharded`
   - use when the complete working set fits aggregate L1;
   - load/split once;
   - keep all intermediates in sharded L1;
   - write final streams once.

2. `ConeStreamed`
   - use when the complete signal does not fit;
   - partition final output into large chunks;
   - back-propagate each chunk through the scheme;
   - load only its initial even/odd dependency intervals;
   - execute all lifting steps locally;
   - write only the final interior.

Scheme length alone does not determine whether cone streaming is useful.

The key metric is:

```text
cone_overhead =
    (loaded_dependency_elements - final_output_elements)
    / final_output_elements
```

Short schemes usually have very small halos and are excellent cone-streaming candidates. Long alternating predict/update schemes remain suitable when coefficient vectors are short and chunks are large enough that the derived halo is small relative to the chunk.

Do not use `tap_size` alone to allocate every cone. Use exact backward-derived intervals and compare their size with the chosen chunk.

Terminal `scale-even` and `scale-odd` steps do not increase the cone. When numerically and architecturally safe, fold them into the final local step or final drain rather than materializing additional intermediate routes.


## tt-wavelet project invariants

For the Tenstorrent Fast Wavelet Transform project, preserve these architectural boundaries unless evidence requires changing them:

- scheme parsing remains generic across wavelet families;
- predict and update remain generic shifted finite stencils;
- scale remains a scalar operation;
- swap is preferably metadata/state exchange;
- step geometry owns logical shifts, source/base offsets, and output lengths;
- compute code should not hardcode a specific wavelet family;
- boundary behavior and canonical output semantics must match the reference implementation.

For LWT storage redesigns:

- eliminate route-by-route DRAM materialization first;
- prefer three-slot or similarly minimal resident storage over four full ping/pong copies;
- use explicit active-even, active-odd, and free-slot state;
- if the signal fits, keep all lifting intermediates in L1-sharded storage and write DRAM only at the final drain;
- if it does not fit, prefer a dependency-cone/windowed execution that loads original dependencies, executes all steps locally, and writes only final output;
- do not introduce a complicated per-blob L1/DRAM cache unless measurements prove that coarse execution modes are insufficient;
- preserve the lifting-step sequence and mathematical dependency graph even when storage and scheduling are completely rewritten.

A full remake is acceptable when it removes an architectural bottleneck. Do not preserve obsolete route, ping/pong, or stream abstractions merely because they already exist.

## API and code-quality rules

Public interfaces must make invalid states difficult to express.

Prefer:

```cpp
struct StepGeometry {
    uint32_t source_offset_elements;
    uint32_t base_offset_elements;
    uint32_t output_elements;
    uint32_t source_left_pad_elements;
};
```

over unrelated positional integer arrays.

Use assertions/fatal validation at host boundaries for:

- unsupported formats;
- overflow;
- invalid shape/layout combinations;
- insufficient L1 capacity;
- misalignment;
- out-of-range core assignments;
- inconsistent route geometry.

Comments should explain:

- invariants;
- physical layout;
- ownership;
- non-obvious synchronization;
- performance rationale;
- hardware constraints.

Do not comment syntax that is already obvious from the code.

## Refactoring policy

Do not perform broad stylistic refactors while changing performance architecture unless necessary for the design.

Delete dead abstractions after the replacement is validated. Avoid:

- parallel legacy and new paths with no concrete need;
- rollback implementations;
- speculative generic frameworks;
- compatibility wrappers around code that should be removed;
- renaming unrelated files;
- reformatting large untouched regions.

Keep diffs reviewable and architecture-focused.

## Testing requirements

A performance change needs both correctness tests and performance evidence.

Correctness tests should include:

- smallest valid input;
- lengths around stick/page/tile/shard boundaries;
- odd and even lengths;
- partial final pages;
- single-core and multi-core execution;
- shard boundaries and remote halos;
- all lifting step kinds;
- representative short and long filters;
- randomized comparison against a CPU/reference implementation;
- exact geometry checks for shifts, offsets, and output lengths.

For a new execution mode, compare it against the existing implementation before deleting the old path.

## Benchmark discipline

Use the same:

- device;
- clock/power mode;
- build type and compiler flags;
- input data and shape;
- core count;
- warmup policy;
- synchronization scope;
- timing boundaries.

Report at least:

- input size;
- active cores;
- warmup count;
- measured iterations;
- minimum;
- median or mean;
- variability;
- throughput;
- estimated DRAM bytes or full passes.

Do not include compilation, allocation, host-device setup, or validation in kernel timing unless the benchmark is explicitly end-to-end.

A regression is not acceptable merely because the new architecture is cleaner. Investigate where time moved.

## Expected response and implementation style

When using this skill, Codex should:

1. Briefly state the discovered bottleneck and intended architectural change.
2. List the critical invariants.
3. Implement rather than only describe, unless the user explicitly asks for design only.
4. Prefer a coherent rewrite when incremental modification preserves the bottleneck.
5. Show the important changed files and dataflow.
6. Report commands actually run.
7. Report tests and benchmarks honestly.
8. Identify remaining risks without inventing success.
9. Avoid asking for confirmation when the repository already provides enough context.
10. Never claim that code compiles, passes, or improves performance unless it was verified.

## Final review checklist

Before finishing, verify:

- [ ] Mathematical semantics are preserved.
- [ ] Ownership and lifetimes are explicit.
- [ ] Physical memory placement is known.
- [ ] Full DRAM passes are counted.
- [ ] L1/temporary footprint is bounded.
- [ ] Work partitions match shard ownership.
- [ ] Halo behavior is correct.
- [ ] Every barrier has a dependency.
- [ ] Inner loops avoid hidden allocation and policy branching.
- [ ] Integer narrowing and size arithmetic are checked.
- [ ] Boundary and partial-page cases are tested.
- [ ] Benchmarks use comparable conditions.
- [ ] Performance claims are labeled measured or hypothetical.
- [ ] Dead legacy abstractions are removed when safe.

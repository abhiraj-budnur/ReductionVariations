CUDA Parallel Reduction Techniques

This project demonstrates different approaches to implementing parallel reduction in CUDA using integer summation as the example workload.

The implementations explore how a reduction algorithm evolves from a basic shared-memory tree reduction to more optimized techniques such as sequential addressing, warp-level reduction, loop unrolling, and processing multiple elements per thread.

Table of Contents
Overview
Reduction Algorithm
Implementations
CUDA Optimization Techniques
Requirements
Compilation and Execution
Terminal Outputs
Comparison of Implementations
Known Issues
Future Improvements
Conclusion
Overview

Parallel reduction combines multiple input elements into a single result. Common examples include summation, maximum, minimum, and product operations.

For an input array 
𝑥
 containing 
𝑁
 elements, summation computes:

𝑆
=
∑
𝑖
=
0
𝑁
−
1
𝑥
𝑖

On a GPU, many threads cooperate to compute partial sums. These partial sums are progressively combined until the final result is obtained.

This project contains seven CUDA source files, each illustrating a different reduction technique:

File	Technique
FirstAdd.cu	Initial shared-memory reduction
Interleaved.cu	Interleaved addressing
InterleavedStrided.cu	Interleaved strided addressing
Sequentials.cu	Sequential-addressing tree reduction
UnrollWarp.cu	Warp-level unrolling
completelyUnrolled.cu	Template-based complete unrolling
MultiAdds.cu	Multiple elements per thread and grid-stride accumulation

The example inputs in most files contain the integers 1 through 10, whose expected sum is:

1
+
2
+
⋯
+
10
=
55

completelyUnrolled.cu uses a different test: 1,024 elements, all initialized to 1. Its expected sum is 1,024.

Reduction Algorithm

A typical CUDA reduction consists of the following stages:

Copy input elements from host memory to device memory.
Load elements into thread-local registers or shared memory.
Combine partial results within each thread block.
Write one partial result per block to global memory.
If multiple blocks are used, combine their partial results to obtain the final answer.
Copy the result back to host memory.

Shared memory enables threads within a block to exchange partial results efficiently. Synchronization is necessary whenever a thread may read data written by another thread.

Implementations
1. FirstAdd.cu — Initial Shared-Memory Reduction

This implementation loads input values into shared memory and performs a tree-style reduction.

Each thread loads two elements, adds them, and stores the partial sum in shared memory. The threads then combine partial sums through repeated reduction stages.

The implementation illustrates the basic structure of a CUDA reduction kernel:

Load data into shared memory.
Synchronize threads.
Combine partial sums.
Store the block result.

Learning objective: Understand how threads cooperate through shared memory to compute a partial sum.

The current implementation requires correction before it can reliably compute the expected result for its ten-element input.

2. Interleaved.cu — Interleaved Addressing

This implementation uses the condition:

cpp
if (tid % (s * 2) == 0)
{
    sdata[tid] += sdata[tid + s];
}


At each reduction stage, only threads whose IDs satisfy the condition participate.

The stride doubles at every iteration, progressively combining elements farther apart in shared memory.

Learning objective: Understand interleaved reduction and how thread participation changes across reduction stages.

This approach can cause divergent execution within a warp and uses a non-power-of-two block size in the provided example, so the current code does not correctly support its ten-element configuration.

3. InterleavedStrided.cu — Interleaved Strided Addressing

This version computes an explicit index for each participating thread:

cpp
int index = 2 * s * tid;

if (index < blockDim.x)
{
    sdata[index] += sdata[index + s];
}


Instead of testing the modulo of each thread ID, the kernel calculates the shared-memory position directly.

Learning objective: Compare implicit interleaved addressing with explicit strided indexing.

The indexing pattern can improve the clarity of the participating-thread logic, but the input length and reduction stages must still be compatible with the algorithm.

4. Sequentials.cu — Sequential-Addressing Reduction

This implementation starts with a stride equal to half the block size:

cpp
for (unsigned int s = blockDim.x / 2;
     s > 0;
     s >>= 1)
{
    if (tid < s)
    {
        sdata[tid] += sdata[tid + s];
    }

    __syncthreads();
}


At each stage, the number of participating threads is reduced by half.

This approach avoids the modulo operation used in the interleaved implementation.

Learning objective: Understand a conventional tree reduction with progressively smaller participating thread groups.

The reduction loop works as intended when the block size is a suitable power of two and the input is loaded safely.

5. UnrollWarp.cu — Warp-Level Unrolling

This implementation reduces the shared-memory values until the remaining work fits within one warp, then invokes a manually unrolled reduction sequence.

The helper function combines values at offsets 32, 16, 8, 4, 2, and 1.

The intention is to eliminate loop-control overhead during the final reduction stages.

Learning objective: Explore how warp-level reduction and loop unrolling can reduce synchronization and loop overhead.

The provided kernel does not store the final result to global memory. Its reduction logic and launch configuration also need correction before the result can be considered valid.

6. completelyUnrolled.cu — Complete Reduction Unrolling

This implementation uses a compile-time block size and template specialization to select reduction stages.

For the example configuration:

Input elements: 1,024.
Block size: 256 threads.
Elements loaded per thread: Up to two per block-processing iteration.
Expected result: 1,024.

The kernel combines two elements per thread where possible, reduces the partial sums in shared memory, and performs the final reduction using a manually unrolled sequence.

Learning objective: Understand how compile-time constants and loop unrolling can simplify control flow and enable compiler optimization.

The final result is accumulated on the host from the partial block sums.

7. MultiAdds.cu — Multiple Elements per Thread

This implementation aims to reduce global-memory and synchronization overhead by assigning multiple elements to each thread and accumulating them locally.

The intended approach is:

Calculate the initial input index for each thread.
Accumulate multiple input values into a thread-local sum.
Advance through the input using a grid-stride loop.
Store each thread's partial sum in shared memory.
Reduce the partial sums to one result per block.

Learning objective: Explore how multiple-element processing can reduce the number of reduction stages and improve memory access efficiency.

The current source contains inconsistent identifiers and needs correction before it can compile and be benchmarked.

CUDA Optimization Techniques
Shared memory

Shared memory allows threads in the same block to exchange intermediate results without repeatedly writing them to global memory.

Tree reduction

Tree reduction combines pairs of partial sums in stages. For a power-of-two number of values, the reduction requires approximately 
log
⁡
2
(
𝑁
)
 stages.

Warp-level reduction

A warp consists of 32 CUDA threads. Warp-level reduction exploits the execution and communication properties of a warp to reduce synchronization overhead.

Loop unrolling

Loop unrolling replaces iterative control flow with a sequence of operations. It can reduce branch and loop overhead, although the effect on performance depends on the compiler and GPU architecture.

Multiple elements per thread

Processing several values per thread can increase instruction-level parallelism and reduce the number of participating threads or blocks. It can also increase register pressure.

Grid-stride accumulation

A grid-stride loop allows a kernel to process inputs larger than the work initially assigned to each thread. It is especially useful for large arrays.

Requirements
NVIDIA GPU with CUDA support.
NVIDIA CUDA Toolkit and nvcc.
Compatible NVIDIA GPU driver.
C++ compiler supported by the installed CUDA Toolkit.

Check the environment:

bash
nvcc --version
nvidia-smi

Compilation and Execution

Each source file contains its own main() function, so compile each program independently.

The following commands assume that the source files have been corrected where necessary.

bash
nvcc -O3 -std=c++17 FirstAdd.cu -o first_add
nvcc -O3 -std=c++17 Interleaved.cu -o interleaved
nvcc -O3 -std=c++17 InterleavedStrided.cu -o interleaved_strided
nvcc -O3 -std=c++17 Sequentials.cu -o sequentials
nvcc -O3 -std=c++17 UnrollWarp.cu -o unroll_warp
nvcc -O3 -std=c++17 completelyUnrolled.cu -o completely_unrolled
nvcc -O3 -std=c++17 MultiAdds.cu -o multi_adds


Run each executable separately:

bash
./first_add
./interleaved
./interleaved_strided
./sequentials
./unroll_warp
./completely_unrolled
./multi_adds


The commands above are intended to be used after addressing the known issues documented below. The current source files are not all expected to compile or produce correct results as written.

Terminal Outputs

Actual terminal outputs will be added here after each implementation has been compiled, corrected, and executed.

1. FirstAdd.cu

Expected result after correcting the implementation: 55.

text
[Paste FirstAdd.cu terminal output here]

2. Interleaved.cu

Expected result after correcting the implementation: 55.

text
[Paste Interleaved.cu terminal output here]

3. InterleavedStrided.cu

Expected result after correcting the implementation: 55.

text
[Paste InterleavedStrided.cu terminal output here]

4. Sequentials.cu

Expected result after correcting the implementation: 55.

text
[Paste Sequentials.cu terminal output here]

5. UnrollWarp.cu

Expected result after correcting the implementation: 55.

text
[Paste UnrollWarp.cu terminal output here]

6. completelyUnrolled.cu

Expected result: 1024.

text
[Paste completelyUnrolled.cu terminal output here]

7. MultiAdds.cu

Expected result for the same ten-element example, if configured to use that input: 55.

text
[Paste MultiAdds.cu terminal output here]

Comparison of Implementations

The table below summarizes the intended techniques. It does not represent measured performance results.

Implementation	Reduction approach	Main concept
FirstAdd.cu	Shared-memory tree	Basic reduction structure
Interleaved.cu	Interleaved indexing	Conditional thread participation
InterleavedStrided.cu	Explicit strided indexing	Shared-memory access patterns
Sequentials.cu	Sequential addressing	Halving active threads per stage
UnrollWarp.cu	Warp-level unrolling	Reducing loop overhead
completelyUnrolled.cu	Template-based unrolling	Compile-time specialization
MultiAdds.cu	Multiple elements per thread	Local accumulation and grid-stride processing

Once valid timing results are available, the comparison can be expanded to include:

GPU model.
Input size.
Threads per block.
Kernel latency.
Effective bandwidth.
Correctness result.
Speedup relative to a selected baseline.

For meaningful performance comparisons, all implementations should use the same input size, GPU, compiler settings, and measurement methodology.

Known Issues

The current source code is intended for learning and requires fixes before its outputs should be treated as verified.

FirstAdd.cu: Uses ten threads and a reduction loop that stops before completing the full reduction. It also reads a second input element for each thread without verifying that the index is within bounds.
Interleaved.cu: The ten-thread block is not compatible with the assumed power-of-two tree structure. The reduction may read invalid shared-memory locations.
InterleavedStrided.cu: The same non-power-of-two limitation applies, and the reduction index must be checked carefully to prevent out-of-bounds shared-memory accesses.
Sequentials.cu: Uses a ten-thread block even though the standard halving reduction expects a suitable power-of-two block size. The host output array is also larger than necessary.
UnrollWarp.cu: The final reduction result is not written to g_odata, and the fixed warp-reduction offsets are incompatible with the ten-thread launch.
completelyUnrolled.cu: The grid-size calculation is appropriate for the supplied power-of-two workload, but the reduction logic depends on compatible block-size assumptions. CUDA API return values should also be checked.
MultiAdds.cu: Contains an undefined or incorrectly named variable (n instead of the parameter N) and an invalid output expression (s.data[0]). Its input-loading loop also needs bounds checks to avoid reading past the input array.

For robust implementations, use power-of-two block sizes where required, guard all global-memory accesses, ensure every reduction stage is valid, and check CUDA errors.

Future Improvements

Possible extensions include:

Add a CPU reference implementation to validate every reduction result.
Add CUDA error checking to all allocations, copies, and kernel launches.
Benchmark each kernel with CUDA events and warm-up iterations.
Test larger arrays that require multiple blocks.
Measure effective memory bandwidth and throughput.
Compare the different reduction strategies on the same GPU and workload.
Use Nsight Compute to analyze memory transactions, occupancy, and register usage.
Explore modern warp shuffle reductions using __shfl_down_sync().
Conclusion

This project demonstrates the evolution of CUDA parallel reduction, beginning with shared-memory tree reductions and progressing toward warp-level unrolling and multiple-element processing.

Each implementation highlights a different aspect of GPU programming: thread cooperation, memory access patterns, synchronization, compile-time specialization, and reduction efficiency.

After correctness issues are addressed, these programs can serve as a useful experimental foundation for understanding how CUDA reduction algorithms are developed and optimized.
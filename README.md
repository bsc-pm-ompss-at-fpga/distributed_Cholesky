# Distributed Cholesky

**Name**: Cholesky  
**Contact Person**: OmpSs@FPGA Team, ompss-fpga-support@bsc.es  
**License Agreement**: GPL 3.0  
**Platform**: OmpSs@FPGA+IMP+OMPIF

## Description

The Cholesky benchmark decomposes a Hermitian positive-definite square symmetric matrix into a lower-triangular matrix that solves the following equation:
$$A = LL^T$$
$A$ is the input matrix, $L$ is the output.
The decomposed matrix $L$ multiplied by its transposed $L^T$ results in the original input $A$.
This implementation uses single precision floating-point to represent the matrix elements, stored in column-major order.
There are two reason why it uses column-major.
First, we use the `potrf`from the LAPACK library which is implemented in Fortran, thus it expects column-major.
Second, this layout is useful for the FPGA implementation as we can optimize better the code, more details in the respective section.

### Parallelization with tasks

To parallelize Cholesky using tasks, we distribute the matrix into multiple square blocks, and each task operates at the block level.
There are 4 kernels in total:
```C++
static const int BS = ...;

#pragma oss task inout(A)
void potrf(float A[BS][BS]);

#pragma oss task input(A) inout(B)
void trsm(floag A[BS][BS], float B[BS][BS]);

#pragma oss task input(A) input(B) inout(C)
void gemm(float A[BS][BS], float B[BS][BS], float C[BS][BS]);

#pragma oss task input(A) inout(B)
void syrk(float A[BS][BS], float B[BS][BS]);

void cholesky_blocked(const int nt, float *A[nt][nt])
{
   for (int k = 0; k < nt; k++) {
      potrf( A[k][k] );
      for (int i = k+1; i < nt; i++) {
         trsm( A[k][k],
               A[k][i] );
      }
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            gemm( A[k][i],
                  A[k][j],
                  A[j][i] );
         }
         syrk( A[k][i],
               A[i][i] );
      }
   }
}
```

You will see that this code is not exactly copied from the `cholesky.c` file, it is simplified to make it more readable and easier to understand.
`cholesky_blocked` takes as input two parameters, the number of tiles (blocks) `nt` and an array of pointers `A`.
All kernels have as parameters two-dimensional arrays of a fixed size `BS` which is the block size.
Therefore, they expect the block input to be consecutive in memory.
Array `A` is a matrix of pointers to the blocks.
`potrf` is the Cholesky decomposition of a single block, defined in the LAPACK library.
The others, `trsm`, `gemm` and `syrk` are defined in he BLAS spec.
Short description found in BLAS webpage:
* `trsm`: solving triangular matrix with multiple right hand sides
* `gemm`: matrix matrix multiply
* `syrk`: symmetric rank-k update to a matrix

The following animation shows on the right the task graph generated by a 4x4 block matrix (any block size).
On the left, the 4x4 block matrix, the colored block is the output (the `inout` parameter), and the grey highlited block are the inputs (the `input` parameters).

![test](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/574a499b-5a01-4be8-af61-393bf13b31a0)

And here is the final task graph:

![cholesky_animation](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/3a9d5861-ea18-4236-ab6c-85a07256479e)

### Parallelization with IMP

The idea is simple: assign each block to a rank, and the task is executed by the rank that has the `inout` block assigned to it.
If there are input blocks assigned to other ranks, IMP will take care of moving data.
However, in this benchmark there is a problem with this straight-forward implementation that comes when a single block is copied multiple times on the same rank.
There is a cache implementation of IMP that memorizes previous copies between blocks, and optimizes the ones that hit in the cache.
The amount of optimization depends on several cache parameters, like size and replacement policy.
In this repository there is another implementation that optimizes all copies in the user code.
The reasoning behind these optimizations are explained in a later section.
Asuming that a block is only sent once, the following image shows the task graph of a 3x3 block matrix, and the same task graph when distributing it in two ranks.

![cholesky_imp (1)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/bf45b536-0262-4e15-bbac-b43ac0fe65e3)

The rightmost figure is the sequence of sends/receives between rank 0 (left column of matrices) and 1 (right column of matrices).
The matrix numbers indicate the rank assigned to each block, and the blocks present in both ranks.
We use a cyclic distribution for both rows and columns, which takes the formula $(i+j)\\%n$ where $n$ is the number of ranks, and $i,j$ the block matrix coordinates.
After trying other distributions, this one seems to get better performance by balancing the workload of every rank.
At the top of the figure, we have the rank assignment for every block.
After that, the sequence of sends and receives in the same order they appear in the middle graph.
All ranks have the same matrix allocated, though they only have the contents of the blocks assigned to them.
Blank positions mean there is no valid data in that block.
In the first step, we can see the block $(0,0)$ moves from rank 0 to rank 1 which was initially blank.
Rank 1 needs the contents of that block because it is an input dependence of the `trsm` task (labeled 1 in the graphs) at block $(1,0)$.
The block generated by the `trsm` of rank 1 is needed later by a `syrk` task on rank 0 at block $(1,1)$ (third row in the send/recv sequence).
The same logic applies for the rest of send/recv pairs of the example.

### Memory optimization

As you have seen in the last figure and in the code, Cholesky only reads and writes the lower triangle of the matrix because this one is symmetric.
Since the matrix dimensions must be square, the size increases very fast, and with this implementation we allocate the whole matrix on every rank.
A small optimization that we do is only allocate the useful part, i.e. the lower triangle of the block matrix.
This is not exactly the lower triangle of the whole matrix, because on the blocks of the diagonal there are elements of the upper triangle, but we save close to 2x memory by doing this.

![cholesky_imp (3)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/efcf416d-58ef-4010-a3a1-6446dc61c286)

This image show how we store the matrix in memory.
First there is the original matrix, the numbers are coordinates to each element, and the colors represent the block each element belongs to.
Just below that, the linear representation in memory stored in column-major order.
Then, we can see the three blocks that are actually used by Cholesky, thus, we only store that part in memory.
Besides only storing the needed blocks, these are stored consecutive in memory in column-major order for elements in a block, and for different blocks.
This is needed because as we explained earlier, each kernel expects to have the block consecutive in memory.
Also, loading an entire block from memory is faster because we can do a single copy instead of doing one per row.

## FPGA implementation

### POTRF



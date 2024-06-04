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

#pragma oss task in(A) inout(B)
void trsm(floag A[BS][BS], float B[BS][BS]);

#pragma oss task in(A) in(B) inout(C)
void gemm(float A[BS][BS], float B[BS][BS], float C[BS][BS]);

#pragma oss task in(A) inout(B)
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

#### Copy optimizations

As we explained before, this implementation optimizes copies in the user code.
It is easier to do this way because the user has the knowledge of how the tasks are created, and how the blocks are distributed in memory.
The latter is essential to get the optimal number of send/receives.
Lets go in order for every possible copy on each kernel:

##### POTRF

There are no input blocks, so we will never have communication with this kernel.

##### TRSM

There is one input block, $(k,k)$, which is the block produced by a `potrf` task, and one output block $(i,k)$ (the code presented at the beggining has these coordinates swapped because of the column-major order).
We know that $i$ is increased by one on each iteration of the `trsm` loop, and since $k$ is constant for this loop, blocks of consecutive iterations are assigned to consecutive ranks.
If the number of blocks in one dimension is greater than the number of ranks, we know that the same rank will execute at least two tasks with the same $(k,k)$ block.
We can optimize the second and any other copy of this block, by only doing IMP when $i-(k+1) < n$ where $n$ is the number of ranks.
I.e. if this condition is true, we give one data owner with the $(k,k)$ block.
If not, we give 0 data owners, ensuring that there's no communication.
Here is a visual example with 3 ranks on a 6x6 block matrix (only showing the first column):

![cholesky_imp (2)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/1084518a-a0a5-44b1-b631-a13421fda697)

The numbers on each position of the column represent the rank assigned to that block.
$i-(k+1)$ is the iteration count starting from 0.
It tells how many `trsm` tasks have already been created.
We can see that rank 1 repeats after creating 3 tasks, because there are 3 ranks and the distribution is cyclic.
This same reasoning applies to every rank, so we know that after creating $n$ tasks, whichever is the rank that has the $(k,k)$ block, the loop will have already created tasks for each rank.
Thus, every rank has the $(k,k)$, and we don't have to repeat the copy.

##### GEMM

This one is a little more tricky, but we use the same reasoning than before.
In fact, all optimizations are based on the fact that the same block is sent to consecutive ranks, so after $n$ tasks of that block, we can stop communication.
There are two input blocks, always produced by previous `trsm` tasks, $(i,k)$ and $(j,k)$, and one output block $(i,j)$.

First, lets look at the first input dependence over block $(i,k)$, which is the easiest to understand.
This is identical to the `trsm` case, but loop variable $j$ traverses rows instead of columns.
Block $(i,k)$ is constant for the `gemm` loop, since it modifies variable $j$, used in the output block $(i,j)$.
Thus, after $n$ iterations, the block $(i,k)$ is already sent to every rank.
We can see this in the following image:

![cholesky_imp (2)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/3245d043-d5a9-4c7b-a84b-8bea6dc4d157)

Its the same case, but there is an extra variable $j$.
Also, we can be sure that block $(i,k)$ is not sent before the `gemm` loop, because for the same $k$, $i$ only increases and thus never repeats the same row.
The $k$ variable also increases without repeating, so on different $k$ the input block also changes and thus we never repeat.
In conclusion, the condition to enable communication is $j-(k+1) < n$.

The second dependence is over block $(j,k)$, which changes on every iteration.
However, the condition ends being $i-(k+1) < n$.
This is because the value $i-(k+1)$ tells us how many tasks on previous $i$ iterations have the block $(i,k)$ as input.
We can also see it as the distance between the input and output blocks according to the cyclic formula $(i+j)\\%n$.
These tasks include `gemm` and `syrk`.
The value of the condition is constant for the $j$ loop because the distance between the input and output blocks is also constant.

![cholesky_imp](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/cd869471-c07d-4e93-a8ac-d16a283ac048)

Here we see all the tasks that have as input the same block of the second dependence of a `gemm` task for a 4x4 block matrix with 4 ranks (showing only the relevant blocks).
First, that block is the first dependence of the `gemm` tasks for $i-1$ (if any), as well as the input block for the `syrk` task of $i-1$.
After that, the block $(j,k)$ appears as the second dependence of all `gemm` with the same $j$.
The number of tasks created since the first appearance is in fact $i-(k+1)$, and although the coordinates travel on the two dimensions, each task is assigned to consecutive ranks because the assignment is cyclic on both dimensions.
I.e. after $n$ tasks, the ranks start repeating, hence the condition we introduced earlier $i-(k+1) < n$.

In summary, since we know that $i > j$ for all $i$, we can combine both conditions to decide the number of data owners of a `gemm` task.
* If $i-(k+1) < n$, we know that $j-(k+1) < n$, so both dependencies may communicate (2 data owners).
* If $i-(k+1) >= n$ and $j-(k+1) < n$, only the first dependence may communicate (1 data owner).
* if $i-(k+1) >= n$ and $j-(k+1) >= n$ there's no communication (0 data owners).

##### SYRK

This one is as easy as `trsm`.
`syrk` has one input block $(i,k)$ and one output block $(i,i)$.
Before a `syrk` task on row $i$ of any $k$, there are $i-(k+1)$ `gemm` tasks with the same $(i,k)$ input block, so the condition is the same $i-(k+1) < n$.
You can see that in the last figure, but for completeness here is an example of a 5x5 block matrix with 3 ranks.

![cholesky_imp (1)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/fdd4b5aa-2258-4337-af64-cad3451b8d98)

### Memory distribution

The idea we use is very simple, but complex to implement: Every rank has allocated in memory a proportional part of the matrix.
I.e. if the matrix is $x$ bytes, we want that every rank only allocates $x/nranks$ bytes.
This is simple with a square matrix and a more friendly data decomposition, but we are doing a cyclic distribution on both rows and columns over a symmetrical square matrix.
Before going into details, lets go step by step.
First, lets see a regular Cholesky matrix stored by columns in memory.

![cholesky_imp-Page-3](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/cedba3e6-b39d-40d2-9ce1-f2a8500b7e53)

The numbers represent matrix coordinates $(i,j)$ and the colors represent the block they belong to.
In this figure we have a 4x4 matrix divided in blocks of 2x2.
As you can see, the blocks are not consecutive in memory, as the distance between two rows is 4.
Another fact that we have to take into account in Cholesky is that the algorithm only uses the lower triangular matrix (it is symmetric).
Therefore, the first step is so store the matrix by blocks, and only store blocks that belong to the lower triangle.

![cholesky_imp-Page-3](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/e01750b6-9f85-44a4-9938-749190ff39a9)

This memory layout is much better for the Cholesky algorithm.
Every block is stored consecutive in memory by columns.
Therefore, with a single pointer to the block, we can access it as a small isolated matrix of 2x2 without knowing the real dimensions of the original matrix.
Now, lets see how every block would be assigned to each rank.
From now on, we only represent blocks in the figures, so each square is a full block, not an element of the matrix.

![cholesky_imp-Page-5](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/8cc96839-e6b2-4ac5-b0e8-839ff0f8e0cf)

The numbers represent the rank owner of that block, as well as the colors.
The first question is how to allocate the blocks on each rank.
I.e. after the first block, what should go next, the  block on the right, bottom, left?
In the figure it is not obvious because there are only 3 blocks.
But the general idea is to store one anti-diagonal of blocks after the other, from top-left to bottom-right.
Then, each anti-diagonal store it from bottom-left to top-right.

![cholesky_imp-Page-5 (1)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/5ba07c23-5a09-45fd-ad20-5aba2a3e3ca5)

This figure represents a bit better the memory layout of the previous example with 3 ranks.
Now, the numbers on each block represent the *block* coordinates, and the color represents the rank owner.
As you can see, in memory each rank only stores the blocks owned by itself, with the aforementioned order.
This memory layout is very useful because it allow us to get the memory address of any block from the block coordinates with a generic formula.
For example, for rank 0, with just some arithmetic operations we can learn that block $(3,0)$ is in address $1$ in memory, or that block $(3,3)$ is in address $3$.

For that, we start with how to calculate the number of elements in a given anti-diagonal $d$.
The answer is $d/2 + 1$ when $d > n$, being $n$ the number of blocks in a single dimension of the matrix, and with integer division, i.e. truncating the decimals.

![cholesky_imp-Page-5 (2)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/05f4453a-5068-4e2b-8a32-899b42df57bf)

This is a bigger example.
The numbers represent anti-diagonal $d$, as well as the colors.
Anti-diagonals that satisfy the condition $d < n$ have light colors and black font, while the rest have darker colors and white font.
You can see that every 2 anti-diagonals, it grows by 1 block.
However, this only works for half of the matrix.
After that, if you use the same formula you get more blocks.
In fact, you can visualize how many blocks you are counting.

![cholesky_imp-Page-5 (3)](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/2e2c7197-4e41-42c5-acd7-44ad459c872d)

One solution to this problem is to subtract the extra part we don't want to count.
In this case, the number of extra blocks we are counting grows by 1 between consecutive anti-diagonals.
Therefore, the resulting formula is $d/2 - (d-n+1)$.
In summary, we have:
* $d/2 + 1$ if $d <= n$ and truncating the decimals.
* $d/2 + 1 - (d-n+1)$ if $d > n$ and truncating the decimals.

Now, we can use this formula to calculate how many blocks there are stored from anti-diagonal $0$ to $d$.
I.e. the memory offset of anti-diagonal $d+1$.
For the moment, we assume there is only one rank, so all blocks are allocated with the anti-diagonal order.
For example, we want to know that including anti-diagonal $2$, there are $3$ blocks in memory, thus the first block of anti-diagonal $4$ has address $3$.
In the case of anti-diagonal $7$, we want to know there are $17$ blocks, thus the first block of anti-diagonal $8$ has address $17$.
Again, we will split this calculation in two.
First, lets see how it would look in the first half of the matrix, when $d < n$:

$$\sum_{i=0}^{d} i/2+1$$

In this summatory, $d$ is included, so we can simplify the formula like:

$$(\sum_{i=0}^{d} i)/2 + d+1$$

And then, remove the summatory and simplify the formula:

$$\frac{d*(d+1)}{4} + d+1$$

However, this simplification introduces a new problem.
This fraction doesn't truncate the decimals of the divisions inside the summatory.
I.e. even when truncating the division by 4, the formula is still counting decimals that we don't want to include.
For example, when $d=5$, the formula would give us $1 + 1.5 + 2 + 2.5 + 3 + 3.5 = 13.5$, but the actual result is $12$.
Therefore, we have to subtract those $0.5$ units from the result.
We call this the correction factor.
The number of $0.5$ units we add is equal to $(d+1)/2$, therefore the quantity we have to subtract is $(d+1)/4$.
Here we can also do integer division and truncate the result, because we only care about the number of 1s added to the final result, and the trailing $0.5$ is already removed by the original formula itself.
Finally, we have the formula:

$$\frac{d*(d+1)}{4} + d+1 - \frac{d+1}{4}$$

Now, we can go for the final formula when $d > n$.
Lets go back some steps with the summatory:

$$(\sum_{i=0}^{d} i/2+1) - (\sum_{i=0}^{d-n} i+1)$$

Result after simplification is:

$$\frac{d*(d+1)}{4} + d+1 - \frac{d+1}{4} - (\frac{(d-n)*(d-n+1)}{2} + (d-n+1))$$

The only part left is taking into account there are multiple ranks.
We apply the same idea, however we have to change some parts of the formula.
Imagine the previous example with 3 ranks, on rank 1 we would have:

![cholesky_imp-Page-5](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/3a584b9e-78e3-4ffa-acd2-7ddb7916544b)

In this case, we can see that the summatory doesn't start at anti-diagonal 0, it starts at the rank index $r$, and instead of jumping from one anti-diagonal to the next one, it is jumping by the number of ranks $s$.
In this example, anti-diagonal $4$ would be stored locally in anti-diagonal $1$ of rank 1, and thus we would have to sum $(1/2+1)+(4/2+1)$.
Again, starting only on the first half of the triangle, we would have the summatory like this:

$$\sum_{i=0}^{d} \frac{r+i*s}{2} + 1$$

This time, $d$ refers to the local anti-diagonal, which can be calculated from the global anti-diagonal by dividing with the total number of ranks $d/s$.
You can see that we get the same formula as the beggining if $s=1$ and $r=0$.
However, the correction factor now depends also on $r$ and $s$.
In the case of rank $1$, the extra $0.5$ that we don't want to count would be added like this in the summatory: $0.5 + 0 + 0.5 + 0 + 0.5 + 0 + 0.5$...
Now the $0.5$ starts in the first unit of the summatory, which changes slightly the correction factor.
This is because when $d+1$ is odd, we are counting an extra $0.5$, which may generate an integer unit not counted in the old correction factor $(d+1)/4$.
For example, when $d=2$ the correction factor would be $(2+1)/4 = 0$ but there are two $0.5$.
To correct this, we have to adjust the formula to $(d+2)/4$.
This depends on if $r$ and $s$ are even or odd, resulting in 4 different possibilities:

* If $r$ is even and $s$ is even, we are only adding numbers that are multiple of 2, so the correction factor is $0$.
* If $r$ is even and $s$ is odd, we have the first explained case and the correction factor is $(d+1)/4$.
* If $r$ is odd and $s$ is even, we are always adding numbers that are not multiple of 2, so all add a $0.5$. The correction factor is $(d+1)/2$.
* If $r$ is odd and $s$ is odd, we have the second explained case and the correction factor is $(d+2)/4$.

From now on, we call the correction factor $cf$ in the formulas.
After removing the summatory and simplifying, we would have the final formula:

$$\frac{2* r*(d+1) + s* d*(d+1)}{4} + d+1 - cf$$

We only have one case left, when $d > n$ being $d$ the global anti-diagonal.
In the previous case, the initial offset of the summatory was easy to get because it was the rank $r$ itself.
However, for the second summatory is no as straightforward.
This depends on who is the owner of the first anti-diagonal that is greater than $n$.
In the case of the first summatory it is always $0$ because that's how we designed the data decomposition.
However, the *mirror* triangle (the one we want to subtract) first anti-diagonal depends on the number of blocks in a single dimension of the matrix.
In the example there are $6$ blocks and $3$ ranks, and since one is multiple of the other, the offset matches, but if there are $7$ blocks, the first anti-diagonal greater than $n$ would belong to rank $1$.
Therefore, we have to find the distance between the owner of the first anti-diagonal of the *mirror* triangle and rank $r$.
This owner is $n \bmod s$, so we have two cases:

* If $r >= n \bmod s$, distance is $r - (n \bmod s)$
* If $r < n \bmod s$, distance is $r+s - (n \bmod s)$

## FPGA implementation

### POTRF

The `potrf` kernel FPGA implementation is not very interesing.
It doesn't give much margin for optimizations due to the access patterns and dependencies between iterations.
There are pipelining directives, but the initiation interval can't reach 1.
To give a little boost, the loops are manually unrolled by a factor of 2.
However, the total execution time of `potrf` compared to the rest is very small so we didn't dedicate much efforts in optimizing the kernel.

### TRSM

The `trsm` solves a triangular magtrix equation, storing the result in block $B$, and $A$ is transposed.
There is a main loop over $k$.
Inside, first an $i$ loop that is pipelined and unrolled by a configurable factor (in the code is unrolled to be executed in 4 iterations).
Here we get II 1 because the $B$ block is partitioned cyclically by columns (since is column-major), and the loop iterates over consecutive columns.
In this case, column-major order helps because if it were row major, we would have to change partitioning and thus slow-down loading the block from memory, as well as storing the result.
The code loads the block as fast as possible with a data bus that can bring multiple data elements per cycle, but for that we need a cylic partitioning so we can store the data in BRAMs in the same cycle.
With a column partition, positions from consecutive column would be stored in the same memory bank, and it would not be possible to store them in parallel.

The second loop is a nested loop.
It is pipelined with a factor of 4 (it could be 1 but this way we reduce resource usage of this kernel), and the inner loop is also unrolled implicitly by the pipeline directive.
Since `tmp_row` and $B$ (it's called row but in fact it's a column) are partitioned, we can access to half of the elements in parallel to achieve II=2.

### GEMM

This is a regular matrix multiply in column-major order with the $B$ matrix transposed.
The middle loop is pipeline with a factor of 2 (could be 1 but this way we reduce resources), this implies an unroll of the innermost loop and flattening with the outermost.
We get the desired II because of the array partitioning and because the loop order doesn't add any dependence between consecutive iterations of the inner and middle loops (kij order).
Consequently, we calculate 128 elements per cycle.
Again, this is possible because the matrices are column-major.
Without transposing $B$, it is possible to use a cylic partition and use efficiently the memory port.
But in this case $B$ is transposed, so with row-major we would have the same conflict as with `trsm`.

### SYRK

This one is very similar to `gemm`, but it multiplies a symmetric matrix by its transpose.
Also, the code only updates the lower-triangular half of the block.
However, you will see the innermost $j$ loop has full range from 0 to `ts` (block size).
Instead of starting at $i$, the code executes all iterations and multiplies by 0 when $j < i$.
This is useful to get pipelining in the middle $i$ loop, because the $j$ loop has a constant trip count and we can unroll it (needed for pipelining).
To reduce resources, the II is 4, so we calculate 64 elements per cycle.

## How to compile

Sadly, there is no official support in the clang compiler for OMPIF and IMP, so we have to split manually the FPGA and the host part. The host code is under `src`.
In `cholesky.c` you will find the definitions of all kernels, the FPGA code and a call to OpenBLAS or MKL depending on some defines.
That code is not used, it is there because the compiler needs a definition of the functions.
In execution time, the code will search for the FPGA accelerators, which are implemented in hls under the `hls` directory.

So, first you need an OmpSs@FPGA installation with broadcaster support for both in Nanos6 and clang. Also, you need the AIT version that implements Ethernet subsystem and OMPIF support. For the moment these versions are not official nor public, you can ask for access to ompss-fpga-support@bsc.es.

In summary, to compile the host executable run `make`.
To generate the bitstream run `make ait`
By default, it generates the bitstream from the HLS phase, but you can change the start and finish phases with the Makefile variables `TO_STEP` and `FROM_STEP`.
Besides that, there are other variables:
* FPGA_CLOCK: Frequency in MHz at which the accelerators will run.
* FPGA_MEMPORT_WIDTH: Data bit-width of the memory port for all the accelerators. More bit-width may provide more bandwidth (depending on the FPGA memory path), at the cost of more resource usage. **IMPORTANT** This variable is intended to be used in the task pragma. Since there is no compiler support, if you want to change the port width you have to modify the hls code on each kernel individually. By default it is 256.
* BLOCK_SIZE: Size of one of the dimensions of the block matrices, which are square. **IMPORTANT** The block size affects both the host and hls code, so if you modify the variable, you have to apply the changes manually in all hls codes.
* SYRK_NUM_ACCS: Number of `syrk` accelerators.  **IMPORTANT** If you want to change this variable, change the `num_instances` field of the `ait_extracted.json` file. Usually this file is generated by clang, but since we do not have that support with OMPIF or IMP, we have to modify the file manually.
* GEMM_NUM_ACCS: Number of `gemm` accelerators. **IMPORTANT** Same as SYRK_NUM_ACCS.
* TRSM_NUM_ACCS: Number of `trsm` accelerators. **IMPORTANT** Same as SYRK_NUM_ACCS.
* POTRF_SMP: This variable must be always 0 for this version of the benchmark. It is used to offload the `potrf` kernel to the CPU from the FPGA, but this is not supported in the distributed version with many FPGAs.
* FPGA_GEMM_II: The inititation interval of the `gemm` kernel. Setting it to 1 achieves best performance but it uses the most number of resources because you fully unroll the innermost loop, which depends on the block size (an entire block column). By default it is 2, which reduces performance by 2x but also reduces resources by the same amount approximately. **IMPORTANT** This variable is intended to be used in the FPGA code of `cholesky.c`, but since we don't use it, you have to change the corresponsing variable in `omp_gemm.cpp`.
* FPGA_OTHER_II: The initiation interval of the `trsm` and `syrk` kernels. The same reasoning with the FPGA_GEMM_II applies with this variable for `trsm` and `syrk`. We separate both because the most critical kernel is `gemm`, so we prefer giving more resources to that kernel, and use lower II for the others. **IMPORTANT** The same as with FPGA_GEMM_II, you have to change the corresponding variable in `omp_trsm.cpp` and `omp_syrk.cpp`.

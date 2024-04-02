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

void cholesky_blocked(const int nt, float A[nt][nt])
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

![test](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/574a499b-5a01-4be8-af61-393bf13b31a0)

![cholesky_animation](https://github.com/bsc-pm-ompss-at-fpga/distributed_Cholesky/assets/17345627/3a9d5861-ea18-4236-ab6c-85a07256479e)



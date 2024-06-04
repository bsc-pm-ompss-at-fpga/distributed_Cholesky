/*
* Copyright (c) 2020, BSC (Barcelona Supercomputing Center)
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the <organization> nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY BSC ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <assert.h>

#include "cholesky.h"
#include "cholesky.fpga.h"

#include <nanos6/distributed.h>
#include <nanos6/debug.h>

void smp_potrf(int s, type_t *A) {
   static const char L = 'L';
   int info;
   potrf(&L, &s, A, &s, &info);
}

#if defined(OPENBLAS_IMPL) || defined(POTRF_SMP)
#pragma oss task copy_deps inout([ts*ts]A)
#else
//#pragma oss task device(fpga) copy_deps inout([ts*ts]A)
#endif
void omp_potrf(type_t *A)
{
#if defined(OPENBLAS_IMPL) || defined(POTRF_SMP)
   static const char L = 'L';
   int info;
   potrf(&L, &ts, A, &ts, &info);
#else
   #pragma HLS inline
   #pragma HLS array_partition variable=A cyclic factor=FPGA_PWIDTH/64
   for (int j = 0; j < ts; ++j) {
      type_t tmp = A[j*ts + j];
      for (int k = 0; k < j; ++k) {
         #pragma HLS pipeline II=1
         type_t Akj = A[k*ts + j];
         tmp -= Akj*Akj;
      }

      A[j*ts + j] = sqrtf(tmp);

      for (int i = j + 1; i < ts; ++i) {
         type_t tmp = A[j*ts + i];
         for (int k = 0; k < j; ++k) {
            #pragma HLS pipeline II=1
            tmp -= A[k*ts + i]*A[k*ts + j];
         }
         A[j*ts + i] = tmp/A[j*ts + j];
      }
   }
#endif
}

#ifdef OPENBLAS_IMPL
#pragma oss task in([ts*ts]A) inout([ts*ts]B)
#else
//#pragma oss task device(fpga) num_instances(TRSM_NUMACCS) copy_deps in([ts*ts]A) inout([ts*ts]B)
#endif
void omp_trsm(const type_t *A, type_t *B)
{
#ifdef OPENBLAS_IMPL
   trsm(CBLAS_MAT_ORDER, CBLAS_RI, CBLAS_LO, CBLAS_T, CBLAS_NU,
      ts, ts, 1.0, A, ts, B, ts);
#else
   #pragma HLS inline
   #pragma HLS array_partition variable=A cyclic factor=FPGA_PWIDTH/64
   #pragma HLS array_partition variable=B cyclic factor=ts/FPGA_OTHER_II
   #pragma HLS array_partition variable=tmp_row cyclic factor=ts/(2*FPGA_OTHER_II)
   type_t tmp_row[ts];

   for (int k = 0; k < ts; ++k) {
      type_t temp = 1. / A[k*ts + k];
      for (int i = 0; i < ts; ++i) {
         #pragma HLS unroll factor=ts/FPGA_OTHER_II
         #pragma HLS pipeline II=1
         //Sometimes Vivado HLS doesn't achieve II=1 because it detects
         //some false dependence on B, this fixes the issue. Same for the other loop
         #pragma HLS DEPENDENCE variable=B inter false
         B[k*ts + i] = tmp_row[i] = temp * B[k*ts + i];
      }

      for (int j = k + 1; j < ts; ++j) {
         #pragma HLS pipeline II=FPGA_OTHER_II
         #pragma HLS DEPENDENCE variable=B inter false
         for (int i = 0; i < ts; ++i) {
            B[j*ts + i] -= A[k*ts + j] * tmp_row[i];
         }
      }
   }
#endif
}

#ifdef OPENBLAS_IMPL
#pragma oss task in([ts*ts]A) inout([ts*ts]B)
#else
//#pragma oss task device(fpga) num_instances(SYRK_NUMACCS) copy_deps in([ts*ts]A) inout([ts*ts]B)
#endif
void omp_syrk(const type_t *A, type_t *B)
{
#ifdef OPENBLAS_IMPL
   syrk(CBLAS_MAT_ORDER, CBLAS_LO, CBLAS_NT,
      ts, ts, -1.0, A, ts, 1.0, B, ts);
#else
   #pragma HLS inline
   #pragma HLS array_partition variable=A cyclic factor=ts/FPGA_OTHER_II
   #pragma HLS array_partition variable=B cyclic factor=ts/FPGA_OTHER_II

   for (int k = 0; k < ts; ++k) {
      for (int i = 0; i < ts; ++i) {
         #pragma HLS pipeline II=FPGA_OTHER_II
         for (int j = 0; j < ts; ++j) {
            //NOTE: Instead of reduce the 'i' iterations, multiply by 0
            B[i*ts + j] += -A[k*ts + i] * (j < i ? 0 : A[k*ts + j]);
         }
      }
   }
#endif
}

#ifdef OPENBLAS_IMPL
#pragma oss task in([ts*ts]A, [ts*ts]B) inout([ts*ts]C)
#else
//#pragma oss task device(fpga) num_instances(GEMM_NUMACCS) copy_deps in([ts*ts]A, [ts*ts]B) inout([ts*ts]C)
#endif
void omp_gemm(const type_t *A, const type_t *B, type_t *C)
{
#ifdef OPENBLAS_IMPL
   gemm(CBLAS_MAT_ORDER, CBLAS_NT, CBLAS_T,
      ts, ts, ts, -1.0, A, ts, B, ts, 1.0, C, ts);
#else
   #pragma HLS inline
   #pragma HLS array_partition variable=A cyclic factor=ts/(2*FPGA_GEMM_II)
   #pragma HLS array_partition variable=B cyclic factor=FPGA_PWIDTH/64
   #pragma HLS array_partition variable=C cyclic factor=ts/FPGA_GEMM_II
   #ifdef USE_URAM
   #pragma HLS resource variable=A core=XPM_MEMORY uram
   #pragma HLS resource variable=B core=XPM_MEMORY uram
   #endif

   for (int k = 0; k < ts; ++k) {
      for (int i = 0; i < ts; ++i) {
         #pragma HLS pipeline II=FPGA_GEMM_II
         for (int j = 0; j < ts; ++j) {
            C[i*ts + j] += A[k*ts + j] * -B[k*ts + i];
         }
      }
   }
#endif
}

#ifdef OPENBLAS_IMPL
#pragma oss task inout([nt*nt*ts*ts]A)
#else
#pragma oss task device(broadcaster) inout(A[0])
#endif
void cholesky_blocked(const int nt, type_t* A)
{
   for (int k = 0; k < nt; k++) {

      // Diagonal Block factorization
      omp_potrf( A + (k*nt + k)*ts*ts );

      // Triangular systems
      #ifdef OPENBLAS_IMPL
      for (int i = k+1; i < nt; i++) {
      #else
      // Create in inverse order because Picos wakes up ready
      // chain tasks in that order
      for (int i = nt-1; i >= k+1; i--) {
      #endif
         omp_trsm( A + (k*nt + k)*ts*ts,
                   A + (k*nt + i)*ts*ts );
      }

      // Update trailing matrix
      for (int i = k + 1; i < nt; i++) {
         for (int j = k + 1; j < i; j++) {
            omp_gemm( A + (k*nt + i)*ts*ts,
                      A + (k*nt + j)*ts*ts,
                      A + (j*nt + i)*ts*ts );
         }
         omp_syrk( A + (k*nt + i)*ts*ts,
                   A + (i*nt + i)*ts*ts );
      }
   }
   #pragma oss taskwait
}

static inline size_t get_total_blocks(int nt, int r, int s)
{
   int cf;
   int n = (2*nt-1)/s + (r < (2*nt-1)%s ? 1 : 0) - 1;
   int globaln = r + n*s;
   if (s%2 == 0 && r%2 == 0) {
      cf = 0;
   }
   else if (s%2 == 0 && r%2 == 1) {
      cf = (n+1)/2;
   }
   else if (s%2 == 1 && r%2 == 0) {
      cf = (n+1)/4;
   } else { //s%2 == 1 && r%2 == 1
      cf = (n+2)/4;
   }
   int rem = n - (nt/s + (r < nt%s ? 1 : 0));
   int remstart = r >= nt%s ? r-nt%s : r+s - nt%s;
   int remblocks = (rem+1)*remstart + s*((rem*(rem+1))/2) + (rem+1);
   int nblocks = (s*n*(n+1) + 2*(n+1)*r)/4 + (n+1) - cf - (globaln >= nt ? remblocks : 0);
   return nblocks;
}

static void triangular_to_linear(type_t* triangular, type_t* linear, int n, int nt)
{
   int diag_offset = 0;
   for (int d = 0; d < 2*nt-1; ++d) {
      int blocks = d/2 + 1 - (d >= nt ? d-nt + 1 : 0);
      int i = d >= nt ? nt-1 : d;
      int j = d >= nt ? d-nt+1 : 0;
      type_t* block_array;
      block_array = triangular + diag_offset*ts*ts;
      diag_offset += blocks;
      for (int b = 0; b < blocks; ++b) {
         const type_t* triangular_b = block_array + b*ts*ts;
         scatter_block(n, triangular_b, linear + j*ts*n + i*ts); //linear matrix is column major
         if (i != j) scatter_transposed_block(n, triangular_b, linear + i*ts*n + j*ts);
         --i, ++j;
      }
   }
}

void initialize_matrix_blocked_lower(const int n, type_t *matrix, int parse)
{
   // This code was designed for an MPI application, every rank would allocate and initialize its own
   // part of the matrix. In this version, the CPU allocates and initializes the whole matrix.
   // If the CPU doesn't have enough memory for the full matrix, a simple solution is to allocate just the space
   // for the FPGA with more blocks, and initialize the part corresponding to each FPGA by changing this
   // rank and size parameters.
   int rank = 0;
   int size = 1;

   //ISEED is INTEGER array, dimension (4)
   //On entry, the seed of the random number generator; the array
   //elements must be between 0 and 4095, and ISEED(4) must be odd.
   //On exit, the seed is updated.
   //int ISEED[4] = {0,0,0,1};
   const int intONE=1;

   const int nt = n/ts;

   const int tb = get_total_blocks(nt, rank, size);

#ifdef VERBOSE
   if (rank == 0 && !parse)
      printf("Initializing matrix with random values ...\n");
#endif

   const long long msize = tb*ts*ts;
   int cpus = nanos6_get_num_cpus();
   const long long bsize = msize/cpus < 1024 ? 1024 : (msize/cpus > 2147483648ull ? 2147483648ull : msize/cpus);

   for (long long i = 0; i < msize; i += bsize) {
      #pragma oss task firstprivate(i)
      {
         int ISEED[4] = {0, i >> 32, i & 0xFFFFFFFF, 1};
         int final_size = msize-i > bsize ? bsize : msize-i; 
         larnv(&intONE, &ISEED[0], &final_size, matrix + i);
      }
   }
   #pragma oss taskwait

   type_t a = (type_t)n;
   int diag_offset = 0;
   for (int d = rank; d < 2*nt-1; d += size) {
      int blocks = d/2 + 1 - (d >= nt ? d-nt + 1 : 0);
      for (int b = 0; b < blocks; b++) {
         //#pragma oss task firstprivate(diag_offset, b, blocks, d)
         {
         type_t* bmat = matrix + (diag_offset + b)*ts*ts;
         for (int j = 0; j < ts; ++j) {
            for (int i = (d%2 == 0 && b == blocks-1 ? j : 0); i < ts; ++i) {
               bmat[j*ts + i] *= 2;
               if (d%2 == 0 && b == blocks-1) {//diagonal block
                  if (i == j)
                     bmat[j*ts + i] += a;
                  else
                     bmat[i*ts + j] = bmat[j*ts + i];
               }
            }
         }
         }
      }
      diag_offset += blocks;
   }
}

// Robust Check the factorization of the matrix A2
static int check_factorization(int N, type_t *A1, type_t *A2, int LDA, char uplo)
{
#ifdef VERBOSE
   printf ("Checking result ...\n");
#endif

   char NORM = 'I', ALL = 'A', UP = 'U', LO = 'L', TR = 'T', NU = 'N', RI = 'R';
   type_t alpha = 1.0;
   type_t const b = 2.0;
#ifdef USE_DOUBLE
   const int t = 53;
#else
   const int t = 24;
#endif
   type_t const eps = pow_di( b, -t );

   type_t *Residual = (type_t *)malloc(N*N*sizeof(type_t));
   type_t *L1       = (type_t *)malloc(N*N*sizeof(type_t));
   type_t *L2       = (type_t *)malloc(N*N*sizeof(type_t));
   type_t *work     = (type_t *)malloc(N*sizeof(type_t));

   memset((void*)L1, 0, N*N*sizeof(type_t));
   memset((void*)L2, 0, N*N*sizeof(type_t));

   lacpy(&ALL, &N, &N, A1, &LDA, Residual, &N);

   /* Dealing with L'L or U'U  */
   if (uplo == 'U'){
      lacpy(&UP, &N, &N, A2, &LDA, L1, &N);
      lacpy(&UP, &N, &N, A2, &LDA, L2, &N);
      trmm(CBLAS_MAT_ORDER, CBLAS_LF, CBLAS_UP, CBLAS_T, CBLAS_NU,
         N, N, alpha, L1, N, L2, N);
   } else {
      lacpy(&LO, &N, &N, A2, &LDA, L1, &N);
      lacpy(&LO, &N, &N, A2, &LDA, L2, &N);
      trmm(CBLAS_MAT_ORDER, CBLAS_RI, CBLAS_LO, CBLAS_T, CBLAS_NU,
         N, N, alpha, L1, N, L2, N);
   }

   /* Compute the Residual || A -L'L|| */
   for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
         Residual[j*N+i] = L2[j*N+i] - Residual[j*N+i];
      }
   }

   type_t Rnorm = lange(&NORM, &N, &N, Residual, &N, work);
   type_t Anorm = lange(&NORM, &N, &N, A1, &N, work);

   printf("==================================================\n");
   printf("Checking the Cholesky Factorization \n");
#ifdef VERBOSE
   printf("-- Rnorm = %e \n", Rnorm);
   printf("-- Anorm = %e \n", Anorm);
   printf("-- Anorm*N*eps = %e \n", Anorm*N*eps);
   printf("-- ||L'L-A||_oo/(||A||_oo.N.eps) = %e \n",Rnorm/(Anorm*N*eps));
#endif

   const int info_factorization = isnan(Rnorm/(Anorm*N*eps)) ||
      isinf(Rnorm/(Anorm*N*eps)) || (Rnorm/(Anorm*N*eps) > 60.0);

   if ( info_factorization ){
      fprintf(stderr, "\n-- Factorization is suspicious ! \n\n");
   } else {
      printf("\n-- Factorization is CORRECT ! \n\n");
   }

   free(work);
   free(L2);
   free(L1);
   free(Residual);

   return info_factorization;
}

int main(int argc, char* argv[])
{
   char *result[3] = {"n/a","sucessful","UNSUCCESSFUL"};

   if ( argc != 4 ) {
      fprintf( stderr, "USAGE:\t%s <matrix size> <check> <parse>\n", argv[0] );
      return 1;
   }
   const unsigned long long int  n = atoi(argv[1]); // matrix size
   int check    = atoi(argv[2]); // check result?
   int parse    = atoi(argv[3]);
   const unsigned long long int nt = n / ts; // number of tiles
   if ( n % ts != 0 ) {
      fprintf( stderr, "ERROR:\t<matrix size> is not multiple of <block size>\n" );
      return 1;
   }

   int nranks = nanos6_dist_num_devices();
   if (nranks <= 0) {
      fprintf(stderr, "No devices found!\n");
      return 1;
   }

   // Allocate matrix
   type_t * matrix;
   // Allocate blocked matrix
   type_t *Ab;
   type_t *recv_buffer;
   const size_t s = ts * ts * sizeof(type_t);
   const size_t tb = get_total_blocks(nt, 0, 1);
   Ab = malloc(tb*s);
   recv_buffer = malloc(2*nt*s);
   if (Ab == NULL || recv_buffer == NULL) {
      fprintf(stderr, "Could not allocate matrix\n");
      return 1;
   }

   double tIniStart = wall_time();

   // Init matrix
   initialize_matrix_blocked_lower(n, Ab, parse);

   type_t * original_matrix = NULL;
   if ( check == 1 ) {
      // Allocate matrix
      original_matrix = (type_t *) malloc(n * n * sizeof(type_t));
      if (original_matrix == NULL) {
         fprintf(stderr, "Could not allocate original matrix\n");
         return 1;
      }
      triangular_to_linear(Ab, original_matrix, n, nt);
      //print_matrix(original_matrix, n);
   }

   const double tEndStart = wall_time();

#ifdef VERBOSE
   if (!parse)
      printf ("Executing ...\n");
#endif

   unsigned int max_tb = get_total_blocks(nt, 0, nranks);
   for (int i = 1; i < nranks; ++i) {
      unsigned int blocks = get_total_blocks(nt, i, nranks);
      if (blocks > max_tb)
         max_tb = blocks;
   }
   nanos6_dist_map_address(Ab, max_tb*s);
   nanos6_dist_map_address(recv_buffer, 2*nt*s);

   nanos6_dist_memcpy_info_t* memcpy_infos = (nanos6_dist_memcpy_info_t*)malloc((2*nt-1)*sizeof(nanos6_dist_memcpy_info_t));
   if (memcpy_infos == NULL) {
      fprintf(stderr, "Could not allocate memcpy infos\n");
      return 1;
   }

   const double tBeginCopy = wall_time();
   
   unsigned int diag_offset = 0;
   unsigned int *r_diag_offset = (unsigned int*)malloc(nranks*sizeof(unsigned int));
   memset(r_diag_offset, 0, nranks*sizeof(unsigned int));
   for (int d = 0; d < 2*nt-1; ++d) {
      int blocks = d/2 + 1 - (d >= nt ? d-nt + 1 : 0);
      int r = d%nranks;
      memcpy_infos[d].size = blocks*s;
      memcpy_infos[d].sendOffset = diag_offset*s;
      memcpy_infos[d].recvOffset = r_diag_offset[r]*s;
      memcpy_infos[d].devId = r;
      r_diag_offset[r] += blocks;
      diag_offset += blocks;
   }
   nanos6_dist_memcpy_vector(Ab, 2*nt-1, memcpy_infos, NANOS6_DIST_COPY_TO);
   const double tEndCopy = wall_time();
   const double tElapsedCopy = tEndCopy-tBeginCopy;
   free(r_diag_offset);
   free(memcpy_infos);

#ifdef VERBOSE
   fprintf(stderr, "Copy bandwidth %fMB/s\n", tb*s/1e6/tElapsedCopy);
#endif

   const double tIniExec = wall_time();
   //Performance execution
   cholesky_blocked(nt, Ab, recv_buffer);
   #pragma oss taskwait
   const double tEndExec = wall_time();

#ifdef VERBOSE
   if (!parse)
      printf("Done\n");
#endif

   double tIniToLinear = 0;
   double tEndToLinear = 0;
   double tIniCheck = 0;
   double tEndCheck = 0;

   if ( check == 1 ) {
      tIniToLinear = wall_time();
      unsigned int diag_offset = 0;
      unsigned int *r_diag_offset = (unsigned int*)malloc(nranks*sizeof(unsigned int));
      memset(r_diag_offset, 0, nranks*sizeof(unsigned int));
      for (unsigned int d = 0; d < 2*nt-1; ++d) {
         unsigned int blocks = d/2 + 1 - (d >= nt ? d-nt + 1 : 0);
         unsigned int r = d%nranks;
         nanos6_dist_memcpy_from_device(r, Ab, blocks*s, r_diag_offset[r]*s, diag_offset*s);
         r_diag_offset[r] += blocks;
         diag_offset += blocks;
      }
      free(r_diag_offset);
      matrix = (type_t*)malloc(n*n*sizeof(type_t));
      if (matrix == NULL) {
          fprintf(stderr, "Could not allocate auxiliar matrix\n");
          return 1;
      }
      triangular_to_linear(Ab, matrix, n, nt);
      tEndToLinear = wall_time();
      tIniCheck = tEndToLinear;
      const char uplo = 'L';
      if ( check_factorization(n, original_matrix, matrix, n, uplo) ) check = 10;
      free(original_matrix);
      free(matrix);
      tEndCheck = wall_time();
   }

   // Print results
   if (parse) {
      printf("%e\n", tEndExec-tIniExec);
   }
   else {
   float gflops = (float)n/1e3;
   gflops = (gflops*gflops*gflops/3.f)/(tEndExec - tIniExec);
   printf( "==================== RESULTS ===================== \n" );
   printf( "  Benchmark: %s (%s)\n", "Cholesky", "OmpSs" );
   printf( "  Elements type: %s\n", ELEM_T_STR );
#ifdef VERBOSE
   printf( "  Matrix size:           %llux%llu\n", n, n);
   printf( "  Block size:            %dx%d\n", ts, ts);
#endif
   printf( "  Init. time (secs):     %f\n", tEndStart    - tIniStart );
   printf( "  Execution time (secs): %f\n", tEndExec     - tIniExec );
   printf( "  Convert linear (secs): %f\n", tEndToLinear - tIniToLinear );
   printf( "  Checking time (secs):  %f\n", tEndCheck    - tIniCheck );
   printf( "  Performance (GFLOPS):  %f\n", gflops );
   printf( "================================================== \n" );
   }
   nanos6_dist_unmap_address(Ab);

   // Free blocked matrix
   free(Ab);

   return check == 10 ? 1 : 0;
}

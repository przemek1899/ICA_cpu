/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <cula_lapack_device.h>
#include "pca.cuh"
#include <fstream>
#include <iostream>

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))
#define imax(X, Y)  ((X) > (Y) ? (X) : (Y))
#define NUM_COMPONENTS = 20;

void checkStatus(culaStatus status)
{
    char buf[256];
    if(!status)
        return;
    culaGetErrorInfoString(status, culaGetErrorInfo(), buf, sizeof(buf));
    printf("%s\n", buf);
    culaShutdown();
    exit(EXIT_FAILURE);
}

int getRound(int m, int n){

	if (m % n == 0)
		return m;
	else
		return (m/n) * n + n;
}

__device__ int deviceGetRound(int m, int n){

	if (m % n == 0)
		return m;
	return (m/n) * n + n;
}

/*

colsign2 functions

*/

__global__ void colsign2(nifti_data_type* coeff, int rows, int cols, nifti_data_type * intermediate_results, int m_colsign, int n_colsign){

	extern __shared__ nifti_data_type Ash[];
	int tid = threadIdx.x;

	Ash[tid] = 0.0;
	int x_global_index = threadIdx.x + blockIdx.x * blockDim.x;// + i*gridDim.x;
	if (blockIdx.y < cols && x_global_index < rows){
		int global_data_index = x_global_index + blockIdx.y * rows;

		// find max(abs)
		// shared memory version
		Ash[tid] = coeff[global_data_index];
		__syncthreads();

		int result;
		for (unsigned int s = blockDim.x/2; s>0; s>>=1){
			if (tid < s){
				//Ash[tid] += Ash[tid+s];
				nifti_data_type a = Ash[tid]; nifti_data_type b = Ash[tid+s];
				result = (fabs(a)-fabs(b))>0;
				Ash[tid] = result * a + fabs((nifti_data_type)result - 1) * b;
			}
			__syncthreads();
		}
			
		if (tid == 0){

			intermediate_results[blockIdx.x + blockIdx.y*gridDim.x] = Ash[0];
			//(r == 0)*(-1) + (r > 0);
		}
	}
}

__global__ void get_colsign(nifti_data_type *intmed_results, int rows, int cols, nifti_data_type * coeff, int m_coeff, int n_coeff, nifti_data_type* maxFindResults_d){

	extern __shared__ nifti_data_type Ash[];

	int tid = threadIdx.x;
	
	Ash[tid] = 0.0;
	if (blockIdx.x < cols){
		Ash[tid] = intmed_results[tid + rows*blockIdx.x];
	}

	// find max(abs)
	int result;
	for (unsigned int s = blockDim.x/2; s>0; s>>=1){
		if (tid < s){
			//Ash[tid] += Ash[tid+s];
			nifti_data_type a = Ash[tid]; nifti_data_type b = Ash[tid+s];
			result = (fabs(a)-fabs(b))>0;
			Ash[tid] = result * a + fabs((nifti_data_type)result - 1) * b;
		}
		__syncthreads();
	}

	int r = Ash[0] >= 0;
	int sign = (r == 0)*(-1) + (r > 0);
	if (tid == 0 && blockIdx.x < cols){
		maxFindResults_d[blockIdx.x] = sign;
	}

	// do tego momentu jest dobrze

	//jeden blok - jedna kolumna (w macierzy coeff, wynikowej U z svd, która siedzi w tablicy A)
	if (blockIdx.x < cols){
		int iter = deviceGetRound(m_coeff, blockDim.x) / blockDim.x; // m / d³ugoœæ bloku
		for (unsigned i=0; i < iter; i++){
			//int index = tid + blockDim.x*blockIdx.x + i * gridDim.x;
			int rowIndex = tid + blockDim.x * i;
			if (rowIndex < m_coeff){
				coeff[rowIndex + blockIdx.x*m_coeff] *= sign;
			}
		}
	}
}


void print_matrix_data(float * Matrix, int m, int n, int print_to_shell, int write_to_file, const char * filename){

	if (write_to_file && print_to_shell){
		std::ofstream file_data;
		file_data.open(filename);

		for(int i=0; i < n; i++){
			std::cout << Matrix[i] << std::endl;
			file_data << Matrix[i] << "\n";
		}

		file_data.close();
	}
	else if(write_to_file){
		std::ofstream file_data;
		file_data.open(filename);

		if (n == 0){
			for(int j=0; j<m; j++){
				file_data << Matrix[j] << "\n";
			}
		}
		else{
			for(int i=0; i < n; i++){
				for(int j=0; j<m; j++){
					file_data << Matrix[i*m +j] << " ";
				}
				file_data << "\n";
			}
		}
		file_data.close();
	}
	else if(print_to_shell){
		for(int i=0; i < n; i++){
			std::cout << Matrix[i] << std::endl;
		}
	}

	return;
}

__device__ inline double __shfl_down_double(double var, unsigned int srcLane, int width=32) {
  int2 a = *reinterpret_cast<int2*>(&var);
  a.x = __shfl_down(a.x, srcLane, width);
  a.y = __shfl_down(a.y, srcLane, width);
  return *reinterpret_cast<double*>(&a);
}

__device__ inline double __sfhl_down_asm_double(double var, unsigned int delta, int width=32){

	int lo, hi;
	asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r" (hi) : "d"(var));

	// shuffle the 32b registers
	lo = __shfl_down(lo, delta, width);
	hi = __shfl_down(hi, delta, width);

	// recreate the 64b number
	asm volatile("mov.b64 %0, {%1, %2};" : "=d"(var) : "r"(lo), "r"(hi));

	return var;
}

__device__ __inline__ double shfl_asm_double(double x, int laneId){

	// Split the double number into 2 32b registers
	int lo, hi;
	asm volatile("mov.b64 {%0, %1}, %2;" : "=r"(lo), "=r" (hi) : "d"(x));

	// shuffle the 32b registers
	lo = __shfl(lo, laneId);
	hi = __shfl(hi, laneId);

	// recreate the 64b number
	asm volatile("mov.b64 %0, {%1, %2};" : "=d"(x) : "r"(lo), "r"(hi));

	return x;
}

__inline__ __device__ double warpDoubleReduce(double val){
	
	for (int offset = warpSize/2; offset > 0; offset /= 2){
		val += __shfl_down_double(val, offset);
	}
	return val;
}

__inline__ __device__ nifti_data_type blockReduceMuColumn(nifti_data_type val){
	
  static __shared__ nifti_data_type shared[32]; // Shared mem for 32 partial sums
  int lane = threadIdx.x % warpSize;
  int wid = threadIdx.x / warpSize;

  val = warpDoubleReduce(val);

  if (lane==0) shared[wid]=val;
  __syncthreads();

  //read from shared memory only if that warp existed
  val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
  if (wid==0) val = warpDoubleReduce(val); //Final reduce within first warp

  return val;
}

__global__ void mu_shuffle(nifti_data_type * A, int m, int n, int iter, nifti_data_type* MU){

	// na razie zakladamy, ze jeden blok mo¿e przykryæ ca³¹ kolumnê, czyli liczba elementów w kolumnie <= 1024

	static __shared__ nifti_data_type shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	int tid = threadIdx.x;
	int difference = blockDim.x - m;
	int grid_index = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i=0; i<iter; i++){
		int columnIndex = blockIdx.x + i * gridDim.x;
		int globalDataIndex = grid_index - blockIdx.x * difference + i * gridDim.x * m;
		
		if (columnIndex < n){
			// do reduction by shuffle instructions in warps
			nifti_data_type val = 0.0;
			if (tid < m){
				val = A[globalDataIndex];
			}
			__syncthreads(); 
			
			val = warpDoubleReduce(val);
			if (lane==0) shared[wid]=val;
			__syncthreads(); 
			
			//read from shared memory only if that warp existed
			val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

			if (wid==0) val = warpDoubleReduce(val); //Final reduce within first warp

			nifti_data_type mean = val / m;
			if (tid < m){
				A[globalDataIndex] -= mean;
			}
			if (tid == 0 ){
				MU[columnIndex] = mean;
			}
		}
	}
}

/*	MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU MU

	w tej wersji funkcji get_mu oblicza tablice jak ponizej:

						M = 163840 (to jest problem - du¿a liczba kolumn)
		_________________________________________________________
		|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
		|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
  N=121 |___|___|___|___|___|___|___|___|___|___|___|___|___|___|
		|___|___|___|___|___|___|___|___|___|___|___|___|___|___|
		|___|___|___|___|___|___|___|___|___|___|___|___|___|___|

		Ka¿dy BLOK wykonuje redukcjê sumy dla JEDNEJ KOLUMNY

		PROBLEM: liczba kolumn jest zbyt du¿y aby "przykryæ" ca³¹ tablicê blokami za jednym razem - dlatego potrzebne s¹ iteracje
		ALE: co jeœli liczba wierszy (bêdzie wiêksza od 128 albo np. 1024 ??) - to do przemyœlenia

*/

__global__ void get_mu(nifti_data_type * A, int m, int n, int iter, nifti_data_type* MU){

	// tutaj jest zalozenie z m < n (np. m=121, n=163840)
	// every block has an array (shared memory) of ceil((threads_per_blocks / warpSize)) elements
	
	extern __shared__ nifti_data_type Ash[];
	int tid = threadIdx.x;
	int difference = blockDim.x - m;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	//#pragma unroll
	for (int i=0; i<iter; i++){
		
		int columnIndex = blockIdx.x + i * gridDim.x;
		int globalDataIndex = index - blockIdx.x * difference + i * gridDim.x * m;
		if (columnIndex < n){
			
			Ash[tid] = 0.0; // initialize all to zeros (padding the rest of elements which are not part of array
			// each thread loads one element from global memory to shared memory
			if (tid < m){
				Ash[tid] = A[globalDataIndex];
			}
			__syncthreads();

			// do reduction in shared memory
			for (unsigned int s = blockDim.x/2; s>0; s>>=1){
				if (tid < s){
					Ash[tid] += Ash[tid+s];
				}
				__syncthreads();
			}
			
			nifti_data_type mean = Ash[0] / m;
			if (tid < m){
				A[globalDataIndex] -= mean;
			}
			if (tid == 0 ){
				MU[columnIndex] = mean;
			}
		}
	}
	
}

void runPCA(nifti_data_type * A, int m, int n){

	// int DOF = n - 1;
		
	/* Initialize CULA */
	checkCudaErrors(cudaSetDevice(0));
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);
	
	// for m >= n
	char jobu = 'O';  // n > m ? 'S' : 'O';
	char jobvt = 'N'; // n > m ? 'O' : 'S'; bylo S

	// for n > m 
	if (n > m){
		jobu = 'N';
		jobvt = 'O';
	}
	
	const int NCOMPONENTS = 20;
	int lda = m;
    int ldu = m;
    int ldvt = n;
	int min = imin(m,n);
	int max = imax(m,n);
	cudaEvent_t start, stop;
	float elapsedTime;

	nifti_data_type *S, *U, *VT;
	nifti_data_type *A_dev, *AT_dev, *MU_dev, *S_dev, *U_dev, *VT_dev;

	// allocation of memory
	checkCudaErrors(cudaMalloc(&A_dev, m*n*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMalloc(&AT_dev, m*n*sizeof(nifti_data_type))); // AT do transpozycji macierzy
	checkCudaErrors(cudaMalloc(&MU_dev, m*sizeof(nifti_data_type))); // an array only for check the results, need to be removed in final version
		
	S = (nifti_data_type*) malloc(min * sizeof(nifti_data_type));
	checkCudaErrors(cudaMalloc(&S_dev, min * sizeof(nifti_data_type)));

	if (jobu != 'O' && jobu != 'N'){
		U = (nifti_data_type*) calloc(ldu*m, sizeof(nifti_data_type));
		checkCudaErrors(cudaMalloc(&U_dev, ldu*m*sizeof(nifti_data_type)));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		VT = (nifti_data_type*) malloc(ldvt*n*sizeof(nifti_data_type));
		checkCudaErrors(cudaMalloc(&VT_dev, ldvt*n*sizeof(nifti_data_type)));
	}


	// copy data from host to device
	checkCudaErrors(cudaMemcpy(A_dev, A, m*n*sizeof(nifti_data_type), cudaMemcpyHostToDevice));

	// transpozycja macierzy A w celu obliczenia mu
	status = culaDeviceSgeTranspose(m, n, A_dev, m, AT_dev, n);
    checkStatus(status);

	//printf("Calculete transpose-only time: %f ms\n", elapsedTime);
	
	// ---------- MU calculations -----------------------
	int shared_mem_size = getRound(min, 32)*sizeof(nifti_data_type);
	int threadsPerBlock = 128;
	int numBlocks = 65535;
	int iter = getRound(m, numBlocks) / numBlocks;
	printf("shared mem size %d, iter %d\n", shared_mem_size, iter);

	
	//get_mu<<<numBlocks, threadsPerBlock, shared_mem_size>>>(AT_dev, n, m, iter, MU_dev);
	mu_shuffle<<<numBlocks, threadsPerBlock>>>(AT_dev, n, m, iter, MU_dev);
	checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

	// transpozycja macierzy AT po obliczenia mu
	status = culaDeviceSgeTranspose(n,  m, AT_dev, n, A_dev, m);
    checkStatus(status);

	nifti_data_type *MU = (nifti_data_type*) malloc(max*sizeof(nifti_data_type));
	checkCudaErrors(cudaMemcpy(MU, MU_dev, max*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	print_matrix_data(MU, 1, m, 0, 1, "mu_shuffle.txt");
	free(MU);

	// ------------- SVD -----------------------------
	// coeff = U_dev (m x min)
	
	checkCudaErrors(cudaEventCreate(&start));	checkCudaErrors(cudaEventCreate(&stop));	checkCudaErrors(cudaEventRecord(start, 0));

    status = culaDeviceSgesvd(jobu, jobvt, m, n, A_dev, lda, S_dev, U_dev, ldu, VT_dev, ldvt);
    checkStatus(status);
		
	checkCudaErrors(cudaEventRecord(stop, 0));	checkCudaErrors(cudaEventSynchronize(stop));	
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Calculate svd time: %f ms\n", elapsedTime);
	//checkCudaErrors(cudaMemcpy(S, S_dev, min*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	//print_matrix_data(S, min, 0, 0, 1, "S_matrix.txt");

	// -------------- colsign ---------------------------------
	threadsPerBlock = 512;
	int blocks_per_column = getRound(m, threadsPerBlock) / threadsPerBlock;
	int grid_x = getRound(NCOMPONENTS, 32);
	dim3 grid(blocks_per_column, grid_x);
	shared_mem_size = threadsPerBlock*sizeof(nifti_data_type);
	printf("shared mem size %d, iter %d\n", shared_mem_size);

	nifti_data_type * intermediate_results;
	checkCudaErrors(cudaMalloc(&intermediate_results, blocks_per_column*NCOMPONENTS*sizeof(nifti_data_type)));

	colsign2<<<grid, threadsPerBlock, shared_mem_size>>>(A_dev, m, NCOMPONENTS, intermediate_results, blocks_per_column, NCOMPONENTS);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//nifti_data_type * intermediate_results_h = (nifti_data_type*) malloc(new_cols*blocks_per_column*sizeof(nifti_data_type));
	//checkCudaErrors(cudaMemcpy(intermediate_results_h, intermediate_results, new_cols*blocks_per_column*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	//print_matrix_data(intermediate_results_h, blocks_per_column, new_cols, 0, 1, "intermed_res.txt");
	//free(intermediate_results_h);

	// ------------------- get_colsign ----------------------------------------------------
	nifti_data_type* maxFindResults = (nifti_data_type*) malloc(NCOMPONENTS*sizeof(nifti_data_type));
	nifti_data_type* maxFindResults_d;
	checkCudaErrors(cudaMalloc(&maxFindResults_d, NCOMPONENTS*sizeof(nifti_data_type)));

	dim3 grid2(grid_x, 1);
	shared_mem_size = blocks_per_column*sizeof(nifti_data_type);
	get_colsign<<<grid2, blocks_per_column, shared_mem_size>>>(intermediate_results, blocks_per_column, NCOMPONENTS, A_dev, m, NCOMPONENTS, maxFindResults_d);
	
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	nifti_data_type* coeff = (nifti_data_type*) malloc(m*NCOMPONENTS*sizeof(nifti_data_type));
	checkCudaErrors(cudaMemcpy(coeff, A_dev, m*NCOMPONENTS*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	print_matrix_data(coeff, m, NCOMPONENTS, 0, 1, "coeff_mat.txt");

	checkCudaErrors(cudaMemcpy(maxFindResults, maxFindResults_d, NCOMPONENTS*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(maxFindResults_d));
	print_matrix_data(maxFindResults, NCOMPONENTS, 0, 0, 1, "max_results.txt");
	free(maxFindResults);

	// free memory
	free(S);
	free(coeff);
	checkCudaErrors(cudaFree(intermediate_results));
	checkCudaErrors(cudaFree(A_dev));
	checkCudaErrors(cudaFree(S_dev));
	checkCudaErrors(cudaFree(MU_dev));
	checkCudaErrors(cudaFree(AT_dev));

	if (jobu != 'O' && jobu != 'N'){
		free(U);
		checkCudaErrors(cudaFree(U_dev));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		free(VT);
		checkCudaErrors(cudaFree(VT_dev));
	}

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	//checkCudaErrors(cudaDeviceReset()); // dla debuggera
	return;
}
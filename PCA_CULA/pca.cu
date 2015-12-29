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

sign_convention1 function searches for an element in a column (row) with the biggest absolute value,
than wirtes each found element to an array which later is used by another kernel program

*/

__global__ void sign_convention1(nifti_data_type* coeff, int rows, int cols, nifti_data_type * intermediate_results, int m_colsign, int n_colsign){

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

__global__ void sign_convention2(nifti_data_type *intmed_results, int rows, int cols, nifti_data_type * coeff, int m_coeff, int n_coeff, nifti_data_type* maxFindResults_d){

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
	/*
	if (tid == 0 && blockIdx.x < cols){
		maxFindResults_d[blockIdx.x] = sign;
	}*/

	if (blockIdx.x < cols){
		int iter = deviceGetRound(m_coeff, blockDim.x) / blockDim.x; // m / d�ugo�� bloku
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

__global__ void center_data_shuffle(nifti_data_type * A, int m, int n, int iter, nifti_data_type* MU){

	// an assumption is that one row is no longer than 1024 (nframes <= 1024)

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
				//MU[columnIndex] = mean;
			}
		}
	}
}


__global__ void center_data(nifti_data_type * A, int m, int n, int iter, nifti_data_type* MU){

	// m < n (np. m=121, n=163840)
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
			/*
			if (tid == 0 ){
				MU[columnIndex] = mean;
			}*/
		}
	}
	
}

void runPCA(nifti_data_type * A, int m, int n, int ncomponents, nifti_data_type* coeff_result){

		
	/* Initialize CULA */
	checkCudaErrors(cudaSetDevice(0));
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);

	cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

	int maxThreads_x = deviceProp.maxThreadsDim[0];
	
	// for m >= n
	char jobu = 'O';  // n > m ? 'S' : 'O';
	char jobvt = 'N'; // n > m ? 'O' : 'S';

	// for n > m 
	if (n > m){
		jobu = 'N';
		jobvt = 'O';
	}
	
	int lda = m;
    int ldu = m;
    int ldvt = n;
	int min = imin(m,n);
	int max = imax(m,n);
	cudaEvent_t start, stop;
	float elapsedTime;

	nifti_data_type *S, *U, *VT;
	nifti_data_type *A_dev, *AT_dev, *MU_dev, *S_dev, *U_dev, *VT_dev, *intermediate_results;

	// allocation of memory
	checkCudaErrors(cudaMalloc(&A_dev, m*n*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMalloc(&AT_dev, m*n*sizeof(nifti_data_type))); // array AT for transpose matrix
	//checkCudaErrors(cudaMalloc(&MU_dev, m*sizeof(nifti_data_type))); // an array only for checking the results, need to be removed in final version
		
	//S = (nifti_data_type*) malloc(min * sizeof(nifti_data_type));
	checkCudaErrors(cudaMalloc(&S_dev, min * sizeof(nifti_data_type)));

	if (jobu != 'O' && jobu != 'N'){
		U = (nifti_data_type*) calloc(ldu*m, sizeof(nifti_data_type));
		checkCudaErrors(cudaMalloc(&U_dev, ldu*m*sizeof(nifti_data_type)));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		VT = (nifti_data_type*) malloc(ldvt*n*sizeof(nifti_data_type));
		checkCudaErrors(cudaMalloc(&VT_dev, ldvt*n*sizeof(nifti_data_type)));
	}

	checkCudaErrors(cudaMemcpy(A_dev, A, m*n*sizeof(nifti_data_type), cudaMemcpyHostToDevice));

	// ---------- centring the data -----------------------
	//int threadsPerBlock = 128;
	int threadsPerBlock = getRound(min, 32);
	if (threadsPerBlock > maxThreads_x){
		threadsPerBlock = maxThreads_x;
	}
	//int shared_mem_size = getRound(min, 32)*sizeof(nifti_data_type);
	int shared_mem_size = threadsPerBlock*sizeof(nifti_data_type);
	int numBlocks = 65535;
	int iter = getRound(m, numBlocks) / numBlocks;
	//printf("shared mem size %d, iter %d\n", shared_mem_size, iter);

	// transpose matrix for function to centring the data
	status = culaDeviceSgeTranspose(m, n, A_dev, m, AT_dev, n);
    checkStatus(status);
	
	center_data<<<numBlocks, threadsPerBlock, shared_mem_size>>>(AT_dev, n, m, iter, MU_dev);
	checkCudaErrors(cudaDeviceSynchronize()); checkCudaErrors(cudaGetLastError());

	// trasponse matrix again to restore to its initial shape
	status = culaDeviceSgeTranspose(n,  m, AT_dev, n, A_dev, m);
    checkStatus(status);

	// ------------- SVD -----------------------------
	// coeff = U_dev (m x min)
	
	//checkCudaErrors(cudaEventCreate(&start));	checkCudaErrors(cudaEventCreate(&stop));	checkCudaErrors(cudaEventRecord(start, 0));

    status = culaDeviceSgesvd(jobu, jobvt, m, n, A_dev, lda, S_dev, U_dev, ldu, VT_dev, ldvt);
    checkStatus(status);
		
	//checkCudaErrors(cudaEventRecord(stop, 0));	checkCudaErrors(cudaEventSynchronize(stop));
	//checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	//printf("Calculate svd time: %f ms\n", elapsedTime);

	// -------------- sign convention on the coefficients ---------------------------------
	threadsPerBlock = 512;
	int blocks_per_column = getRound(m, threadsPerBlock) / threadsPerBlock;
	int grid_x = getRound(ncomponents, 32);
	dim3 grid(blocks_per_column, grid_x);
	shared_mem_size = threadsPerBlock*sizeof(nifti_data_type);
	//printf("shared mem size %d, iter %d\n", shared_mem_size);

	checkCudaErrors(cudaMalloc(&intermediate_results, blocks_per_column*ncomponents*sizeof(nifti_data_type)));

	sign_convention1<<<grid, threadsPerBlock, shared_mem_size>>>(A_dev, m, ncomponents, intermediate_results, blocks_per_column, ncomponents);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	// ------------------- run second kernel ----------------------------------------------------
	//nifti_data_type* maxFindResults = (nifti_data_type*) malloc(ncomponents*sizeof(nifti_data_type));
	nifti_data_type* maxFindResults_d;
	//checkCudaErrors(cudaMalloc(&maxFindResults_d, ncomponents*sizeof(nifti_data_type)));

	dim3 grid2(grid_x, 1);
	shared_mem_size = blocks_per_column*sizeof(nifti_data_type);
	sign_convention2<<<grid2, blocks_per_column, shared_mem_size>>>(intermediate_results, blocks_per_column, ncomponents, A_dev, m, ncomponents, maxFindResults_d);
	
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//nifti_data_type* coeff = (nifti_data_type*) malloc(m*ncomponents*sizeof(nifti_data_type));
	checkCudaErrors(cudaMemcpy(coeff_result, A_dev, m*ncomponents*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	//print_matrix_data(coeff, m, ncomponents, 0, 1, "coeff_mat.txt");

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	//checkCudaErrors(cudaMemcpy(maxFindResults, maxFindResults_d, ncomponents*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	//checkCudaErrors(cudaFree(maxFindResults_d));
	//print_matrix_data(maxFindResults, ncomponents, 0, 0, 1, "max_results.txt");
	//free(maxFindResults);

	// free memory
	//free(S);
	//free(coeff);
	checkCudaErrors(cudaFree(intermediate_results));
	checkCudaErrors(cudaFree(A_dev));
	checkCudaErrors(cudaFree(S_dev));
	//checkCudaErrors(cudaFree(MU_dev));
	checkCudaErrors(cudaFree(AT_dev));

	if (jobu != 'O' && jobu != 'N'){
		free(U);
		checkCudaErrors(cudaFree(U_dev));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		free(VT);
		checkCudaErrors(cudaFree(VT_dev));
	}

	//checkCudaErrors(cudaEventDestroy(start));
	//checkCudaErrors(cudaEventDestroy(stop));

	//checkCudaErrors(cudaDeviceReset());
	return;
}
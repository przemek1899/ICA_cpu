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

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

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


__device__ __inline__ double shfl_double(double x, int laneId){

	// Split the double number into 2 32b registers
	int lo, hi;
	asm volatile("mov.b32 {%0, %1}, %2;" : "=r"(lo), "=r" (hi) : "d"(x));

	// shuffle the 32b registers
	lo = __shfl(lo, laneId);
	hi = __shfl(hi, laneId);

	// recreate the 64b number
	asm volatile("mov.b64 %0, {%1, %2};" : "=d"(x) : "r"(lo), "r"(hi));

	return x;
}

__global__ void mu_shuffle(nifti_data_type * A, int m, int n, int iter){

}

__global__ void test_shuffle_reduce() {

	int laneId = threadIdx.x & 0x1f;
	int value = 31 - laneId;

	// Use XOR to perform butterfly shuffle
	for(unsigned int i=16; i>=1; i/=2){
		value += __shfl_xor(value, i, 32);
	}
	// "value" now contains the sum across all threads 
	printf("Thread %d final value = %d\n", threadIdx.x, value);
}


__global__ void get_mu(nifti_data_type * A, int m, int n, int iter){

	// in this version thera are not yet weights, not needed now
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

			int mean = Ash[0] / m;
			if (tid < m){
				A[globalDataIndex] -= mean;
			}
		}
	}
	
}

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

void runPCA(nifti_data_type * A, int m, int n){

	// int DOF = n - 1;
		
	/* Initialize CULA */
	checkCudaErrors(cudaSetDevice(0));
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);
	
	// for m >= n
	char jobu = 'O';  // n > m ? 'S' : 'O';
	char jobvt = 'S'; // n > m ? 'O' : 'S';

	// for n > m 
	if (n > m){
		jobu = 'S';
		jobvt = 'O';
	}

	int lda = m;
    int ldu = m;
    int ldvt = n;
	int min = imin(m,n);	
	int i;

	nifti_data_type *S, *U, *VT;
	nifti_data_type *A_dev, *AT_dev, *MU_dev, *S_dev, *U_dev, *VT_dev;

	checkCudaErrors(cudaMalloc(&A_dev, m*n*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMemcpy(A_dev, A, m*n*sizeof(nifti_data_type), cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	float elapsedTime;

	// transpozycja macierzy A w celu obliczenia mu
	
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	checkCudaErrors(cudaMalloc(&AT_dev, m*n*sizeof(nifti_data_type)));
	status = culaDeviceDgeTranspose(m, n, A_dev, m, AT_dev, n);
    checkStatus(status);

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Calculete transpose-only time: %f ms\n", elapsedTime);
	checkCudaErrors(cudaFree(AT_dev));

	/* obliczanie wartoœci mu */
	//checkCudaErrors(cudaMalloc(&MU_dev, n*sizeof(nifti_data_type)));

	int shared_mem_size = getRound(m, 32)*sizeof(nifti_data_type);
	int threadsPerBlock = 128;
	int numBlocks = 65535;
	int iter = getRound(n, numBlocks) / numBlocks;

	

	//get_mu<<<numBlocks, threadsPerBlock, shared_mem_size>>>(A_dev, m, n, iter);

	checkCudaErrors(cudaGetLastError());
	/*
	//sprawdzenie wartoœci - kopiowanie do cpu - to w przyszlosci zostanie usuniête
	//nifti_data_type *MU = (nifti_data_type*) malloc(n*sizeof(nifti_data_type));
	//checkCudaErrors(cudaMemcpy(MU, MU_dev, n*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	
	// reading & writing mu array - print_matrix_data(MU, n, 0, 1, "mu_file.txt");

	free(MU);
	/* koniec obliczania wartoœci mu */

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

	/* Perform singular value decomposition CULA */
    

    status = culaDeviceDgesvd(jobu, jobvt, m, n, A_dev, lda, S_dev, U_dev, ldu, VT_dev, ldvt);
    checkStatus(status);

	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	checkCudaErrors(cudaMemcpy(S, S_dev, min*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	
	// reading S Matrinx - print_matrix_data(S, min, 1, 1, "Smatrix.txt")

	//free memory
	free(S);
	checkCudaErrors(cudaFree(A_dev));
	checkCudaErrors(cudaFree(S_dev));
	//checkCudaErrors(cudaFree(MU_dev));

	if (jobu != 'O' && jobu != 'N'){
		free(U);
		checkCudaErrors(cudaFree(U_dev));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		free(VT);
		checkCudaErrors(cudaFree(VT_dev));
	}

	//checkCudaErrors(cudaDeviceReset()); // dla debuggera
	
	printf("Calculete mu-only time: %f ms\n", elapsedTime);
	return;
}


/*
#define TILE_DIM    16
#define BLOCK_ROWS  16

__global__ void transpose_matrix(nifti_data_type * odata, nifti_data_type * idata, int height, int width){

	__shared__ nifti_data_type tile[];

	int y_diff = deviceGetRound(height, blockDim.y);

	int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
    }

    __syncthreads();

	
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
    {
        odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
    }
}
*/
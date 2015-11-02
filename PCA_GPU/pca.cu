
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

__global__ void pca_gpu(float* tab, int n){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		tab[i] = i*i;
	}

}

void runPCA(void){

	checkCudaErrors(cudaSetDevice(0));

	//initialize cusolverDn
	cusolverDnHandle_t handle = NULL;
	checkCudaErrors(cusolverDnCreate(&handle));

	int m = 64;
	int n = 64;
	float *dev_A;
	//allocate memory
    //checkCudaErrors(cudaMalloc(&dev_A, m*n*sizeof(float)));
    //checkCudaErrors(cudaMalloc(&dev_C, m*m*sizeof(float)));
	/*
	// copy data from cpu to gpu memory
    checkCudaErrors(cudaMemcpy(dev_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice));
	*/

	cudaEvent_t start, stop;
	float elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	// call kernel function here
	//pca_gpu<<<64, 64>>>(dev_A, m*n);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	
	//float * c;
	//c = (float *) malloc(m*n*sizeof(float));

	//copy results from gpu memory to cpu
	//checkCudaErrors(cudaMemcpy(c, dev_A, m*n*sizeof(float), cudaMemcpyDeviceToHost));
	
	//free gpu memory
	//checkCudaErrors(cudaFree(dev_A));
	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	//free(c);
	
	checkCudaErrors(cusolverDnDestroy(handle));
	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
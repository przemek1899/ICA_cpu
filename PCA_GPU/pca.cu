
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "pca.cuh"

__global__ void pca_gpu(float* tab, int n){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		tab[i] = i*i;
	}

}

void runPCA(nifti_data_type * data, int xyzv){

	checkCudaErrors(cudaSetDevice(0));

	//initialize cusolverDn
	cusolverDnHandle_t handle = NULL;
	cusolverDnCreate(&handle); //sprawdzac checkCudaErrors

	//allocate memory
	nifti_data_type * dev_A;
    checkCudaErrors(cudaMalloc(&dev_A, xyzv*sizeof(nifti_data_type)));
    
	// copy data from cpu to gpu memory
    checkCudaErrors(cudaMemcpy(dev_A, data, xyzv*sizeof(nifti_data_type), cudaMemcpyHostToDevice));
	

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

	//copy results from gpu memory to cpu
	//checkCudaErrors(cudaMemcpy(c, dev_A, m*n*sizeof(float), cudaMemcpyDeviceToHost));
	
	//free gpu memory
	checkCudaErrors(cudaFree(dev_A));
	cusolverDnDestroy(handle); //sprawdzac checkCudaErrors
	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	//free(c);
	
	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
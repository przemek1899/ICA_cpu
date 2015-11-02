
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"

__global__ void pca_gpu(){

}

void runPCA(void){

	checkCudaErrors(cudaSetDevice(0));
	/*
	float *dev_A, *dev_C;
	//allocate memory
    checkCudaErrors(cudaMalloc(&dev_A, m*n*sizeof(float)));
    checkCudaErrors(cudaMalloc(&dev_C, m*m*sizeof(float)));

	// copy data from cpu to gpu memory
    checkCudaErrors(cudaMemcpy(dev_A, A, m*n*sizeof(float), cudaMemcpyHostToDevice));
	*/

	cudaEvent_t start, stop;
	float elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	// call kernel function here

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));


	/*
	//copy results from gpu memory to cpu
	checkCudaErrors(cudaMemcpy(C, dev_C, m*m*sizeof(float), cudaMemcpyDeviceToHost));

	//free gpu memory
    checkCudaErrors(cudaFree(dev_C));
    checkCudaErrors(cudaFree(dev_A));
    */

	checkCudaErrors(cudaDeviceReset()); // dla debuggera

	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
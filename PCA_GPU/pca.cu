
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "pca.cuh"

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

__global__ void pca_gpu(float* tab, int n){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		tab[i] = i*i;
	}

}

void checkCuSolverErrors(cusolverStatus_t code){

	if(code){
		fprintf(stderr, "Cuda solver error code %d\n", static_cast<unsigned int>(code));
		cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
	}
}

void runPCA(nifti_data_type * data, int m, int n){

	if (m < n){
		fprintf(stderr, "rows parameter (m) is smaller than columns parameter (n)\n");
		exit(EXIT_FAILURE);
	}

	checkCudaErrors(cudaSetDevice(0));

	//initialize cusolverDn
	cusolverDnHandle_t handle = NULL;
	cusolverDnCreate(&handle); //sprawdzac checkCudaErrors

	//allocate memory
	nifti_data_type * dev_A;
    checkCudaErrors(cudaMalloc(&dev_A, m*n*sizeof(nifti_data_type)));
    
	// copy data from cpu to gpu memory
    checkCudaErrors(cudaMemcpy(dev_A, data, m*n*sizeof(nifti_data_type), cudaMemcpyHostToDevice));

	// calculate the size needed for pre-allocated buffer
	// xy - numer of rows, zv - number of columns
	int Lwork;
	checkCuSolverErrors(cusolverDnSgesvd_bufferSize(handle, m, n, &Lwork));

	//prepare arguments for cusolver svd
	char jobu = 'A';
	char jobvt = 'A';
	int *devInfo; checkCudaErrors(cudaMalloc(&devInfo, sizeof(int)));
	int lda = m; // leading dimension is equal to m ?? (or n ??)
    int ldu = m;
    int ldvt = n;

	// below there are some notes from the cuda toolkit cusolver documentation
	// Note that the routine returns V H , not V.
	// Remark 1: gesvd only supports m>=n.  VEEEEEEEEERY IMPORTANT !!!!!!!!!!!!!!!!!!!!!
	// Remark 2: gesvd only supports jobu='A' and jobvt='A' and returns matrix U and V H .
	// rwork - needed for data types C,Z

	printf("m = %d, n = %d, Lwork = %d\n", m, n, Lwork);

	nifti_data_type * S, *U, *VT, *Work, *rwork;
	checkCudaErrors(cudaMalloc(&S, imin(m,n)*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMalloc(&U, ldu*m*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMalloc(&VT, ldvt*n*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMalloc(&Work, Lwork*sizeof(nifti_data_type)));
	//checkCudaErrors(cudaMalloc(&rwork, 5*imin(m,n)*sizeof(nifti_data_type)));

	// do we really need rwork??
	// run cusolver svd
	printf("before run cusolver svd\n");
	checkCuSolverErrors(cusolverDnSgesvd(handle, jobu, jobvt, m, n, dev_A, lda, S, U, ldu, VT, ldvt, Work, Lwork, rwork, devInfo));
	int h_devInfo;
	cudaDeviceSynchronize();
	checkCudaErrors(cudaMemcpy(&h_devInfo, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
	printf("devInfo %d\n", h_devInfo);

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
	//nifti_data_type * diagonalMatrix = (nifti_data_type *) malloc(imin(m,n)*sizeof(nifti_data_type));
	//checkCudaErrors(cudaMemcpy(diagonalMatrix, S, imin(m,n)*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	//int k = imin(m,n);
	//free(diagonalMatrix);
	
	//free gpu memory
	checkCudaErrors(cudaFree(dev_A));
	checkCudaErrors(cudaFree(S));
	checkCudaErrors(cudaFree(U));
	checkCudaErrors(cudaFree(VT));
	checkCudaErrors(cudaFree(Work));
	checkCudaErrors(cudaFree(rwork));
	checkCudaErrors(cudaFree(devInfo));

	cusolverDnDestroy(handle); //sprawdzac checkCudaErrors
	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	//free(c);
	
	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
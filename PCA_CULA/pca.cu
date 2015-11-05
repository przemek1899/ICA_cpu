
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <cula_lapack.h>
#include "pca.cuh"

#define imin(X, Y)  ((X) < (Y) ? (X) : (Y))

__global__ void pca_gpu(float* tab, int n){

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n){
		tab[i] = i*i;
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

	if (m < n){
		fprintf(stderr, "rows parameter (m) is smaller than columns parameter (n)\n");
		exit(EXIT_FAILURE);
	}

	culaStatus status;
	checkCudaErrors(cudaSetDevice(0));

	//prepare arguments for cusolver svd
	char jobu = 'O';
	char jobvt = 'S';
	int lda = m; // leading dimension is equal to m ?? (or n ??)
    int ldu = m;
    int ldvt = n;

	nifti_data_type *S = (nifti_data_type*) malloc(imin(m,n) * sizeof(nifti_data_type));
    nifti_data_type *U = (nifti_data_type*) malloc(ldu*m* sizeof(nifti_data_type));
    nifti_data_type *VT = (nifti_data_type*) malloc(ldvt*n* sizeof(nifti_data_type));

	/* Initialize CULA */
    status = culaInitialize();
    checkStatus(status);

	/* Perform singular value decomposition CULA */
    printf("Performing singular value decomposition using CULA ... ");

	cudaEvent_t start, stop;
	float elapsedTime;

	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

    status = culaSgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt);
    checkStatus(status);

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
	
	//free host memory
	free(S);
	free(U);
	free(VT);

	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	
	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
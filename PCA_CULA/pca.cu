
/*
 * PCA Principal Component Analysis on raw data
 * This implementation bases on matlab pca implementation
 */

#include "cuda_runtime.h"
#include "helper_cuda.h"
#include "device_launch_parameters.h"
#include <cula_lapack.h>
#include "pca.cuh"
#include <fstream>

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

	int min = imin(m,n);

	nifti_data_type *S, *U, *VT;
	S = (nifti_data_type*) malloc(min * sizeof(nifti_data_type));
	if (jobu != 'O' && jobu != 'N'){
		U = (nifti_data_type*) calloc(ldu*m, sizeof(nifti_data_type));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		VT = (nifti_data_type*) malloc(ldvt*n* sizeof(nifti_data_type));
	}

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

	// read result data
	// reading S diagonal
	
	ofstream S_file;
	S_file.open("Smatrix.txt");

	printf("\nElements of diagonal matrix S are the following: ");
	int i, j;
	for(i=0; i < min; i++){
		// printf("%f ", S[i]);
		S_file << S[i] << "\n";
	}

	S_file.close();
	/*
	// reading first n columns of U matrix
	printf("\nReading the first min(m,n)=%d columns of matrix U from the matrix A\n", imin(m,n));
	for(i=0; i < m; i++){
		for(j=0; j < min; j++){
			printf("%f ", A[i + j*m]);
		}
		printf("\n");
	}
	
	// reading first n rows of VT matrix
	printf("Printing matrix VT\n");
	for(i=0; i < min; i++){
		for(j=0; j < n; j++){
			printf("%f ", VT[i*min + j]);
		}
		printf("\n");
	}
	*/

	//free host memory
	free(S);
	if (jobu != 'O' && jobu != 'N'){
		free(U);
	}
	if (jobvt != 'O' && jobvt != 'N'){
		free(VT);
	}

	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	
	printf("Kernel-only time: %f ms\n", elapsedTime);

	return;
}
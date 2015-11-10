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

	nifti_data_type *S, *U, *VT;
	S = (nifti_data_type*) malloc(min * sizeof(nifti_data_type));
	if (jobu != 'O' && jobu != 'N'){
		U = (nifti_data_type*) calloc(ldu*m, sizeof(nifti_data_type));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		VT = (nifti_data_type*) malloc(ldvt*n* sizeof(nifti_data_type));
	}

	/* Initialize CULA */
	checkCudaErrors(cudaSetDevice(0));
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);

	/* Perform singular value decomposition CULA */
    printf("Performing singular value decomposition using CULA ... ");

	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

    status = culaDgesvd(jobu, jobvt, m, n, A, lda, S, U, ldu, VT, ldvt);
    checkStatus(status);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	
	int i, j;
	// read result data
	// reading S diagonal
	
	//std::ofstream S_file;
	//S_file.open("Smatrix.txt");

	printf("\nElements of diagonal matrix S are the following: ");
	
	for(i=0; i < min; i++){
		printf("%f ", S[i]);
		//S_file << S[i] << "\n";
	}

	//S_file.close();
	
	
	// reading first n columns of U matrix
	/*
	std::ofstream U_file;
	U_file.open("Umatrix.txt");

	printf("\nReading the first min(m,n)=%d columns of matrix U from the matrix A\n", imin(m,n));
	for(i=0; i < m; i++){
		for(j=0; j < min; j++){
			printf("%f ", A[i + j*m]);
		}
		printf("\n");
	}

	U_file.close();
	*/
	
	// reading first n rows of VT matrix
	/*
	std::ofstream VT_file;
	VT_file.open("VTmatrix_sample.txt");

	printf("Printing matrix VT\n");
	for(i=0; i < min; i++){
		for(j=0; j < n; j++){
			printf("%f ", VT[i*min + j]);
			VT_file << VT[i*min + j] << " ";
		}
		printf("\n");
		VT_file << "\n";
	}
	VT_file.close();
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
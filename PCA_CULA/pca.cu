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

__global__ void get_mu(nifti_data_type * A, nifti_data_type * MU, int m, int n, int iter){

	// in this version thera are not yet weights, not needed now

	extern __shared__ nifti_data_type Ash[];

	int tid = threadIdx.x;

	int difference = blockDim.x - m;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

#pragma unroll
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

			// write results to global memory
			if (tid == 0){
				MU[columnIndex] = Ash[0];
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
	int i, j; //do ró¿nych iteracji

	nifti_data_type *S, *U, *VT;
	nifti_data_type *A_dev, *MU_dev, *S_dev, *U_dev, *VT_dev;

	checkCudaErrors(cudaMalloc(&A_dev, m*n*sizeof(nifti_data_type)));
	checkCudaErrors(cudaMemcpy(A_dev, A, m*n*sizeof(nifti_data_type), cudaMemcpyHostToDevice));

	/* obliczanie wartoœci mu */
	checkCudaErrors(cudaMalloc(&MU_dev, n*sizeof(nifti_data_type)));

	int shared_mem_size = getRound(m, 32);
	int threadsPerBlock = 128;
	int numBlocks = 65535;
	int iter = getRound(n, numBlocks);
	printf("shared memory size is %d and iter %d\n", shared_mem_size, iter);

	get_mu<<<numBlocks, threadsPerBlock, shared_mem_size>>>(A_dev, MU_dev, m, n, iter);

	checkCudaErrors(cudaGetLastError());

	//sprawdzenie wartoœci - kopiowanie do cpu - to w przyszlosci zostanie usuniête
	nifti_data_type *MU = (nifti_data_type*) malloc(n*sizeof(nifti_data_type));
	checkCudaErrors(cudaMemcpy(MU, MU_dev, n*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));
	
	std::ofstream mu_file;
	mu_file.open("mu_file.txt");

	for (i=0; i<n; i++){
		mu_file << MU[i] << "\n";
	}

	mu_file.close();

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

	/* Initialize CULA */
	checkCudaErrors(cudaSetDevice(0));
    culaStatus status;
    status = culaInitialize();
    checkStatus(status);

	// copy data to 

	/* Perform singular value decomposition CULA */
    printf("Performing singular value decomposition using CULA ... ");

	cudaEvent_t start, stop;
	float elapsedTime;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

    status = culaDeviceDgesvd(jobu, jobvt, m, n, A_dev, lda, S_dev, U_dev, ldu, VT_dev, ldvt);
    checkStatus(status);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	// transfer data from gpu to host memory
	checkCudaErrors(cudaMemcpy(S, S_dev, min*sizeof(nifti_data_type), cudaMemcpyDeviceToHost));

	// read result data
	// reading S diagonal
	
	//std::ofstream S_file;
	//S_file.open("Smatrix.txt");

	printf("\nElements of diagonal matrix S are the following: ");
	
	for(i=0; i < min; i++){
		printf("%f \n", S[i]);
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

	//free memory
	
	free(S);
	checkCudaErrors(cudaFree(A_dev));
	checkCudaErrors(cudaFree(S_dev));
	checkCudaErrors(cudaFree(MU_dev));

	if (jobu != 'O' && jobu != 'N'){
		free(U);
		checkCudaErrors(cudaFree(U_dev));
	}
	if (jobvt != 'O' && jobvt != 'N'){
		free(VT);
		checkCudaErrors(cudaFree(VT_dev));
	}

	checkCudaErrors(cudaDeviceReset()); // dla debuggera
	
	printf("Kernel-only time: %f ms\n", elapsedTime);
	return;
}
#include "helper_timer.h"
#include "pca.cuh"
#include <fslio.h>
#include <fstream>
	

int main(int argc, char * argv[] ){

	//Read nifti data
	/*
	FSLIO *fslio;
    void *buffer;
	short x, y, z, v;
	fslio = FslInit();
    buffer = FslReadAllVolumes(fslio,"/home/pteodors/openfmri/ds105/sub001/BOLD/task001_run001/bold.nii.gz");
    if (buffer == NULL) {
		fprintf(stderr, "\nError opening and reading\n");
		exit(1);
	}
	signed short *bf = (signed short *) buffer;
	FslGetDim(fslio, &x, &y, &z, &v);
	int nvol = x*y*z*v;

	int m = x*y*z; // x*y*z
	int n = v; // = v
	int mn = m*n;
	int i, j;
	if (argc==3){
		m = atoi(argv[1]);
		n = atoi(argv[2]);
	}
	else if(argc == 2){
		m = atoi(argv[1]);
	}

	// some sample data for testing
	/*
	double sample_data [4][5] = {{1.0, 0.0, 0.0, 0.0, 2.0}, {0.0, 0.0, 3.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 4.0, 0.0, 0.0, 0.0}};
	m = 5; n = 4;
	int sample_size = m*n;
	

	// testing version
	nifti_data_type *data = (nifti_data_type*) malloc(sizeof(nifti_data_type)*sample_size);
	int j;
	for(i=0; i < n; i++){
		for(j=0; j < m; j++){
			data[i*m + j] = sample_data[i][j];
		}
	}
	*/

	if (argc < 3){
		printf("You must specify name values of m and n\n");
		return 0;
	}

	int m = atoi(argv[1]);
	int n = atoi(argv[2]);
	int mn = m*n;
	char filename[80];
	sprintf(filename, "/home/pteodors/matlab_pca/%dx%d.bin", m, n);
	
	float* A = (float*) malloc(sizeof(float)*mn);
	FILE * pA;
	const unsigned int num_bytes = mn;
	pA = fopen(filename, "rb");
	int i;
	unsigned int read_bytes = 0;
	while(num_bytes - read_bytes){
		read_bytes = fread((void*)&A[read_bytes], sizeof(float), num_bytes-read_bytes, pA);
	}
	
	fclose(pA);

	nifti_data_type *data = (nifti_data_type*) malloc(sizeof(nifti_data_type)*mn);
	

	// prepare data format
	for(i=0; i < mn; i++){
		//data[i] = (nifti_data_type) bf[i];
		data[i] = (nifti_data_type) A[i];
	}
	free(A);
	//FslClose(fslio);

	int ncomponents = 20;
	nifti_data_type* coeff = (nifti_data_type*) malloc(m*ncomponents*sizeof(nifti_data_type));
	
	int interations = 20;
	int i;
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	for(i=0; i<interations; i++){
		runPCA(data, m, n, ncomponents, coeff); //x*y*z, v
	}

	sdkStopTimer(&timer);
	printf("Processing time: %f ms\n", sdkGetTimerValue(&timer)/(float)interations);
	sdkDeleteTimer(&timer);

	free(data);
	free(coeff);
	return 0;
}

//#include "helper_timer.h"
#include "pca.cuh"
#include <fslio.h>
	

int main(int argc, char * argv[] ){

	//Read nifti data
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
	int i;
	if (argc==3){
		m = atoi(argv[1]);
		n = atoi(argv[2]);
	}
	else if(argc == 2){
		m = atoi(argv[1]);
	}

	// some sample data for testing
	float sample_data [4][5] = {{1.0f, 0.0f, 0.0f, 0.0f, 2.0f}, {0.0f, 0.0f, 3.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 4.0f, 0.0f, 0.0f, 0.0f}};
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

	/* production version
	nifti_data_type *data = (nifti_data_type*) malloc(sizeof(nifti_data_type)*mn);
	for(i=0; i<mn; i++){
		data[i] = bf[i];
	}
	*/
	FslClose(fslio);
	//the end of reading data

	printf("po zamkniecu fslio\n");

	/*
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);*/
	
	//runPCA(data, m, n); //x*y*z, v
	//testing version
	runPCA(sample_data, m, n);

	printf("po runPCA\n");
	//sdkStopTimer(&timer);
	//printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
	//sdkDeleteTimer(&timer);

	free(data);
	return 0;
}

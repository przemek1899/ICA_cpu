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

	int m = 81920; // x*y*z
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
	nifti_data_type *data = (nifti_data_type*) malloc(sizeof(nifti_data_type)*mn);
	for(i=0; i<mn; i++){
		data[i] = bf[i];
	}
	FslClose(fslio);
	//the end of reading data

	printf("po zamkniecu fslio\n");
	/*
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);*/
	
	// keep in mind that first parameter below (x*y*z which is m) must be equal or bigger than the second one (v which is n), m >= n
	runPCA(data, m, n); //x*y*z, v

	printf("po runPCA\n");
	//sdkStopTimer(&timer);
	//printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
	//sdkDeleteTimer(&timer);

	free(data);
	return 0;
}

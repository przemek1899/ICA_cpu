//#include "helper_timer.h"
#include "pca.cuh"
#include <fslio.h>
	



int main(){

	//Read nifti data
	FSLIO *fslio;
    void *buffer;
	short x, y, z, v;
	fslio = FslInit();
    buffer = FslReadAllVolumes(fslio,"/home/pteodors/ds105/sub001/BOLD/task001_run001/bold.nii.gz");
    if (buffer == NULL) {
		fprintf(stderr, "\nError opening and reading\n");
		exit(1);
	}
	signed short *bf = (signed short *) buffer;
	FslGetDim(fslio, &x, &y, &z, &v);
	int nvol = x*y*z*v;
	float *data = (float *) malloc(sizeof(float)*nvol);
	//petla do przemyslania - moze na gpu?
	int i;
	for(i=0; i<nvol; i++){
		data[i] = bf[i];
	}
	FslClose(fslio);
	//the end of reading data


	/*
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);*/
	
	runPCA();

	//sdkStopTimer(&timer);
	//printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
	//sdkDeleteTimer(&timer);

	free(data);
	return 0;
}
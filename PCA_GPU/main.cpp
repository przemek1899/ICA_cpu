//#include "helper_timer.h"
#include "pca.cuh"

int main(){
	/*
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);*/
	
	runPCA();

	//sdkStopTimer(&timer);
	//printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
	//sdkDeleteTimer(&timer);

	return 0;
}
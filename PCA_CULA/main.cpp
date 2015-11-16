#include "helper_timer.h"
#include "pca.cuh"
#include <fslio.h>
#include <fstream>
	

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

	// production version 
	std::ofstream column;
	column.open("Column.txt");
	
	nifti_data_type *data = (nifti_data_type*) malloc(sizeof(nifti_data_type)*mn);

	// w przypadku transpozycji macierzy, zmieniamy tylko przepisywanie danych z bf do data oraz wartoœci m i n
	// kopiowanie danych dla transpozycji macierzy
    // dwie wersje transpozycji macierzy

	// 1 wersja z kolejnymi odczytami 
	for(i = 0; i < n; i++){
		for(j = 0; j < m; j++){
			data[n*j + i] = (nifti_data_type) bf[i*m + j];
		}
	}
	int temp = n;
	//n = m;
	//m = temp;
	
	// 2 wersja z kolejnymi zapisami
	/*
	for(i = 0; i < mn; i++){
		int index = 
		data[i] = (nifti_data_type) bf[];
	}
	

	// kopiowanie danych bez transpozycji maciezrzy
	for(i=0; i < mn; i++){
		data[i] = (nifti_data_tfype) bf[i];
		if (i < v){
			printf("%d:  %f\n", i+1, data[i]);
		}
	}
	column.close();
	*/

	FslClose(fslio);

	
	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);
	sdkStartTimer(&timer);
	
	runPCA(data, m, n); //x*y*z, v

	sdkStopTimer(&timer);
	printf("Processing time: %f ms\n", sdkGetTimerValue(&timer));
	sdkDeleteTimer(&timer);

	free(data);
	return 0;
}

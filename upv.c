/* ----------------------------------------------------------------------
 *
 * compile example (consider -pedantic or -Wall):
 *
 * gcc -o upv upv.c -I../include -L../lib -lfslio -lniftiio -lznz -lz -lm
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <fslio.h>

int main(int argc, char *argv[]){
	
	/*Reading*/
        FSLIO *fslio;
        void *buffer;
        int nvols;
	short dt, x, y, z, v;
        
	/*fslio = FslOpen("/home/przemek/Pulpit/ds105/sub001/BOLD/task001_run001/bold.nii.gz","rb");

	size_t t = FslGetVolSize(fslio);
	printf("vol size %d\n", t);
	FslGetDataType(fslio, &dt);
	printf("typ danych: %d\n", dt);
	FslGetDim(fslio, &x, &y, &z, &v);
	printf("rozmiar %dx%dx%dx%d\n", x, y, z, v);
	/*
          ... can now access header info via the FslGet calls ...
          ... allocate room for buffer ...
	
        //FslReadVolumes(fslio,buffer,nvols);
	
	*/
	fslio = FslInit();
        buffer = FslReadAllVolumes(fslio,"/home/przemek/Pulpit/ds105/sub001/BOLD/task001_run001/bold.nii.gz");
        if (buffer == NULL) {
                fprintf(stderr, "\nError opening and reading\n");
                exit(1);
        }
	double ****scaled = FslGetBufferAsScaledDouble(fslio);
	
	signed short *bf = (signed short *) buffer;

	int i=0;
	if (argc == 2){
		i = atoi(argv[1]);
		printf("argc %d\n", i);
		printf("voxel nr %d: %d\n", i, bf[i]);
	}

	if (argc == 5){
		int i = atoi(argv[1]);
		int j = atoi(argv[2]);
		int k = atoi(argv[3]);
		int t = atoi(argv[4]);	
		// 40 64 64, 40 64 121, 64 64 121
		int dim1 = 40;   
		int dim2 = 64; 
		int dim3 = 64;
		int coordinate = (i + j*dim1 + k*dim1*dim2 + t*dim1*dim2*dim3);
		// (i + j*dim[1] + k*dim[1]*dim[2])
		printf("voxel nr %d (%d,%d,%d,%d): %d\n", coordinate, i, j, k, t, bf[coordinate]);
		printf("voxel nr (%d,%d,%d,%d): %f\n", i, j, k, t, scaled[t][k][j][i]);
	}

	/*
	signed short max = 0;
	int total_len = 40*64*64*121;
	for(i=0; i<total_len; i++){
		if(bf[i] > max){
			max = bf[i];
		}
	}
	printf("max: %d\n", max);*/

        FslClose(fslio);
	return 0;
}


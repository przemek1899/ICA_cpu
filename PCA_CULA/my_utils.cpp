#include "my_utils.h"
#include <iostream>
#include <fstream>

void print_matrix_data(double * Matrix, int n, int print_to_shell, int write_to_file, const char * filename){

	if (write_to_file && print_to_shell){
		std::ofstream file_data;
		file_data.open(filename);

		for(int i=0; i < n; i++){
			std::cout << Matrix[i] << std::endl;
			file_data << Matrix[i] << "\n";
		}

		file_data.close();
	}
	else if(write_to_file){
		std::ofstream file_data;
		file_data.open(filename);

		for(int i=0; i < n; i++){
			file_data << Matrix[i] << "\n";
		}

		file_data.close();
	}
	else if(print_to_shell){
		for(int i=0; i < n; i++){
			std::cout << Matrix[i] << std::endl;
		}
	}

	return;
}

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
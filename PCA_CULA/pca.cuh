
#ifndef PCA_H
#define PCA_H

typedef float nifti_data_type;

void runPCA(nifti_data_type* A, int m, int n, int ncomponents, nifti_data_type* coeff_result);

#endif
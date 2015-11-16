pca: main.o pca.o my_utils.o
	g++ main.o pca.o my_utils.o -L/usr/local/cuda/lib64 -lcudart -L/home/pteodors/cula/lib64 -lcula_lapack -liomp5 -lcublas -L/home/pteodors/nifticlib-2.0.0/lib -lfslio -lniftiio -lznz -lz -o pca

main.o: main.cpp
	g++ -c main.cpp -I/home/pteodors/nifticlib-2.0.0/include -I/usr/local/cuda/samples/common/inc -o main.o

pca.o: pca.cu
	nvcc -arch=sm_50 -c pca.cu -I/usr/local/cuda/samples/common/inc -I/home/pteodors/cula/include -o pca.o

my_utils.o: my_utils.cpp
	g++ -c my_utils.cpp -o my_utils.o
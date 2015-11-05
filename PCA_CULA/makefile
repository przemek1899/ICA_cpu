pca: main.o pca.o
	g++ main.o pca.o -L/usr/local/cuda/lib64 -lcudart -L/home/pteodors/cula/lib64 -lcula_lapack -liomp5 -lcublas -L/home/pteodors/nifticlib-2.0.0/lib -lfslio -lniftiio -lznz -lz -o pca

main.o: main.cpp
	g++ -c main.cpp -I/home/pteodors/nifticlib-2.0.0/include -o main.o

pca.o: pca.cu
	nvcc -c pca.cu -I/usr/local/cuda/samples/common/inc -I/home/pteodors/cula/include -o pca.o



#include<cuda.h>
#include"gettime.c"
#include"tensor.c"
#include<assert.h>
#include"fft.h"
#include"svd.h"
#include"check.h"
int main(int arc,char** argc){
	int m,n,k,l;
	printf("input the size of tensor(m*n*k*l):");
	scanf("%d%d%d%d",&m,&n,&k,&l);
	printf("m=%d,n=%d,k=%d,l=%d\n",m,n,k,l);
	float* T=(float*)malloc(m*n*k*l*sizeof(float));
	float* S=(float*)malloc(sizeof(float)*(m<n)?m:n);
	cuComplex* U=(cuComplex*)malloc(sizeof(cuComplex)*m*((m<n)?m:n));
	cuComplex* V=(cuComplex*)malloc(sizeof(cuComplex)*n*((m<n)?m:n));
	cufftComplex* A=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* B=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* C=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* D=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	cufftComplex* E=(cufftComplex*)malloc(m*n*k*l*sizeof(cufftComplex));
	if(!T||!A||!C||!D||!S||!U||!V){
		printf("host memory allocation failed");
		exit(-1);
	}
	double time1,time2,time3;
	time1=gettime();
	T=createtensor(m,n,k,l,T);
	printtensor(m,n,k,l,T);
	A=createctensor(m,n,k,l,A);
    printctensor(m,n,k,l,A);
    C=tensor_scalar(m,n,k,l,A,C);
    printctensor(m,n,k,l,C);
   svd(m*n,k*l,(cuComplex*)C,U,V,S);
    D=tensor_scalar_transpose(m,n,k,l,C,D);
    printctensor(m,n,k,l,D);
    E=tensor_scalartotensor(m,n,k,l,C,E);
    printctensor(m,n,k,l,E);
    fft(m,n,k,l,A,B);
    printctensor(m,n,k,l,B);
	time2=gettime();
	time3=time2-time1;
	printf("time:%.6f",time3);
    checkkernel();
	free(A);
	free(B);
	free(C);
	free(D);
	free(E);
	free(S);
	free(U);
	free(V);
	return 0;
}

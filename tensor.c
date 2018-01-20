#include"tensor.h"
float* createtensor(int m,int n,int k,int l,float* T){
	srand((unsigned)time(NULL));
	for(int i=0;i<m*n*k*l;i++){
		T[i]=(float)rand()/(RAND_MAX/100);
	}
	return T;
}
cufftComplex* createctensor(int m,int n,int k,int l,cufftComplex* T){
	srand((unsigned)time(NULL));
	for(int i=0;i<m*n*k*l;i++){
		T[i].x=(float)rand()/(RAND_MAX/100);
		T[i].y=(float)rand()/(RAND_MAX/100);
	}
	return T;
}
void printtensor(int m,int n, int k,int l,const float*T){
	for(int h=0;h<l;h++){
		for(int tube=0;tube<k;tube++){
			for(int i=0;i<m;i++){
				for(int j=0;j<n;j++){
					printf("%f\n",T[j+i*n+tube*m*n+h*m*n*k]);
				}
			}
		}
	}
}
void printctensor(int m,int n, int k,int l,const cufftComplex* T){
	for(int h=0;h<l;h++){
		for(int tube=0;tube<k;tube++){
			for(int i=0;i<m;i++){
				for(int j=0;j<n;j++){
					printf("%f%s%fi\n",T[j+i*n+tube*n*m+h*m*n*k].x,
							T[j+i*n+tube*n*m+h*m*n*k].y<0?"":"+",
							T[j+i*n+tube*n*m+h*m*n*k].y);
				}
			}
		}
	}
}
cufftComplex* tensor_scalar_transpose(int m,int n,int k,int l,const cufftComplex* t,cufftComplex* T){
    for(int q=0;q<m;q++){
    	for(int p=0;p<n;p++){
    		for(int i=0;i<l;i++){
    			for(int j=0;j<k;j++){
    				T[l*k*n*q+l*k*p+l*j+i].x=t[l*k*n*q+l*k*p+i*k+j].x;
    				T[l*k*n*q+l*k*p+l*j+i].y=0-t[l*k*n*q+l*k*p+i*k+j].y;
    			}
    		}
    	}
    }
	return T;
}
cufftComplex* matview_transpose(int m,int n,int k,int l,cufftComplex* t,cufftComplex* M){
	for(int q=0;q<l;q++){
		for(int p=0;p<k;p++){
			for(int i=0;i<m;i++){
				for(int j=0;j<n;j++){
					M[m*n*k*q+m*n*p+j*m+i].x=t[j+i*n+p*m*n+m*n*k*q].x;
					M[m*n*k*q+m*n*p+j*m+i].y=0-t[j+i*n+p*m*n+m*n*k*q].y;
				}
			}
		}
	}
	return M;
}
cufftComplex* tensor_scalar(int m,int n,int k,int l,const cufftComplex* t,cufftComplex* s){
	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			for(int p=0;p<l;p++){
				for(int q=0;q<k;q++){
					s[i*l*n*k+j*l*k+p*k+q].x=t[m*n*k*p+m*n*q+n*i+j].x;
					s[i*l*n*k+j*l*k+p*k+q].y=t[m*n*k*p+m*n*q+n*i+j].y;
				}
			}
		}
	}
	return s;
}
cufftComplex* tensor_scalartotensor(int m,int n,int k,int l,cufftComplex* s,cufftComplex* t){
	for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				for(int p=0;p<l;p++){
					for(int q=0;q<k;q++){
						t[m*n*k*p+m*n*q+n*i+j].x=s[i*l*n*k+j*l*k+p*k+q].x;
						t[m*n*k*p+m*n*q+n*i+j].y=s[i*l*n*k+j*l*k+p*k+q].y;
					}
				}
			}
     	}
	return t;
}

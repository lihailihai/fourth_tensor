#include"svd.h"
#include<cusolverDn.h>
void svd(int m,int n,cuComplex* t,cuComplex* U,cuComplex* V,float* S){
     cusolverDnHandle_t handle;
     gesvdjInfo_t params=NULL;
     int* info=NULL;
     int echo=0;
     int lda=m;
     int ldu=m;
     int ldv=n;
     int lwork=0;
     cuComplex* work=NULL;
     float* s;
     cuComplex* u;
     cuComplex* v;
     cusolverStatus_t status=CUSOLVER_STATUS_SUCCESS;
     status=cusolverDnCreate(&handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnCreateGesvdjInfo(&params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     cudaError_t stat1=cudaSuccess;
     cudaError_t stat2=cudaSuccess;
     cudaError_t stat3=cudaSuccess;
     cudaError_t stat4=cudaSuccess;
     stat1=cudaMalloc((void**)&info,sizeof(int));
     stat2=cudaMalloc((void**)&u,sizeof(cuComplex)*m*m);
     stat3=cudaMalloc((void**)&v,sizeof(cuComplex)*n*n);
     stat4=cudaMalloc((void**)&s,sizeof(float)*((m<n)?m:n));
     if(
    		 stat1!=cudaSuccess||
    		 stat2!=cudaSuccess||
    		 stat3!=cudaSuccess||
    		 stat4!=cudaSuccess){
    	 printf("cuda malloc error\n");
    	 exit(-1);
     }
     if(cusolverDnCgesvdj_bufferSize(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 m,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 &lwork,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj_bufferSize failed\n");
    	 exit(-1);

     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("synchronize failed");
    	 exit(-1);
     }
     stat1=cudaMalloc((void**)&work,sizeof(cuComplex)*lwork);
     assert(stat1==cudaSuccess);
     if(cusolverDnCgesvdj(
    		 handle,
    		 CUSOLVER_EIG_MODE_VECTOR,
    		 echo,
    		 m,
    		 n,
    		 t,
    		 lda,
    		 s,
    		 u,
    		 ldu,
    		 v,
    		 ldv,
    		 work,
    		 lwork,
    		 info,
    		 params)!=CUSOLVER_STATUS_SUCCESS){
    	 printf("cusolverDnCgesvdj err\n");
    	 return;
     }
     if(cudaDeviceSynchronize()!=cudaSuccess){
    	 printf("cuda synchronize err\n");
    	 return;
     }
     stat1=cudaMemcpy(U,u,sizeof(cuComplex)*ldu*m,cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(V,v,sizeof(cuComplex)*ldv*n,cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     stat1=cudaMemcpy(S,s,sizeof(float)*((m<n)?m:n),cudaMemcpyDeviceToHost);
     assert(stat1==cudaSuccess);
     status=cusolverDnDestroy(handle);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     status=cusolverDnDestroyGesvdjInfo(params);
     assert(status==CUSOLVER_STATUS_SUCCESS);
     stat1=cudaFree(u);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(v);
     assert(stat1==cudaSuccess);
     stat1=cudaFree(s);
     assert(stat1==cudaSuccess);
}

/*
 * tensor.h
 *
 *  Created on: Jan 11, 2018
 *      Author: haili
 */

#ifndef TENSOR_H_
#define TENSOR_H_
#include<stdio.h>
#include<assert.h>
#include<stdlib.h>
#include<time.h>
#include<cufft.h>
typedef struct csr_t{
	cufftComplex* row_array;
	cufftComplex* col_array;
	cufftComplex* data_array;
}csr;
float* createtensor(int m,int n,int k,int l,float* T);
cufftComplex* createctensor(int m,int n,int k,int l,cufftComplex* T);
void printtensor(int m,int n,int k,int l,const float* T);
void printctensor(int m,int n,int k,int l,const cufftComplex* T);
cufftComplex* tensor_scalar_transpose(int m,int n,int k,int l,const cufftComplex* t,cufftComplex* M);
cufftComplex* matview_transpose(int m,int n,int k,int l,cufftComplex* t,cufftComplex* T);
cufftComplex* tensor_scalar(int m,int n,int k,int l,const cufftComplex* t,cufftComplex* s);
cufftComplex* tensor_scalartotensor(int,int,int,int,cufftComplex*,cufftComplex*);
#endif /* TENSOR_H_ */

/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to call CUBLAS library
 * functions both from the HOST code and from the DEVICE code
 * running on the GPU (the latter is available only for the compute
 * capability >= 3.5). The single-precision matrix-matrix
 * multiplication operation, SGEMM, will be performed 3 times:
 * 1) once by calling a method defined in this file (simple_sgemm),
 * 2) once by calling the cublasSgemm library routine from the HOST code
 * 3) and once by calling the cublasSgemm library routine from
 *    the DEVICE code.
 */

/* Includes, system */
//#include <stdio.h>
//#include <stdlib.h>
#include <string.h>
//
/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Includes, cuda helper functions */
#include <helper_cuda.h>

#include <stdio.h>
#include <time.h>
#include <string>
#include <iostream>
#include "device_launch_parameters.h"

#include<stdlib.h>
#include<sys/mman.h>
#include<unistd.h>
#include<fcntl.h>

#include <sys/stat.h>
using namespace std;

using  std::string;

time_t StringToDatetime(string str)
{
    char *cha = (char*)str.data();             // 将string转换成char*。
    tm tm_;                                    // 定义tm结构体。
    int year, month, day, hour, minute, second;// 定义时间的各个int临时变量。
    sscanf(cha, "%d-%d-%d %d:%d:%d", &year, &month, &day, &hour, &minute, &second);// 将string存储的日期时间，转换为int临时变量。
    tm_.tm_year = year - 1900;                 // 年，由于tm结构体存储的是从1900年开始的时间，所以tm_year为int临时变量减去1900。
    tm_.tm_mon = month - 1;                    // 月，由于tm结构体的月份存储范围为0-11，所以tm_mon为int临时变量减去1。
    tm_.tm_mday = day;                         // 日。
    tm_.tm_hour = hour;                        // 时。
    tm_.tm_min = minute;                       // 分。
    tm_.tm_sec = second;                       // 秒。
    tm_.tm_isdst = 0;                          // 非夏令时。
    time_t t_ = mktime(&tm_);                  // 将tm结构体转换成time_t格式。
    return t_;                                 // 返回值。
}


//void file_input(char* filename)
//{
//	filesize=file_size2(filename);
//
//	int i,f;
//    FILE *fp;
//	//注意这里是open打开不是fopen!!!!
//	f = open("recond.dat",O_RDWR);
//		//获得磁盘文件的内存映射
//	mapped = (Recond *) mmap(0 , NumReconds * sizeof(Byte) , PROT_READ|PROT_WRITE, MAP_SHARED, f, 0);
//	mapped[43].iNum = 999;
//		sprintf(mapped[43].sName,"Recond-%d",mapped[43].iNum);
//		//将修改同步到磁盘中
//	msync((void *)mapped,NumReconds*sizeof(recond),MS_ASYNC);
//		//关闭内存映射
//	munmap((void *)mapped,NumReconds*sizeof(recond));
//	close(f);
//}

/* Main */

int file_size2(char* filename)
{
    struct stat statbuf;
    stat(filename,&statbuf);
    int size=statbuf.st_size;
    return size;
}

struct rezult{
	char  *input;
	long length;
};

rezult fileinput(string filename){
	char *p;
	int len = filename.length();
	p=(char *)malloc((len+1)*sizeof(char));
	filename.copy(p,len,0);

	int fd;
	char *mapped_mem;
	int flength = 1024;
	void * start_addr = 0;
	fd = open(p, O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
	flength = file_size2(p);
    //write(fd, '\n', 1); /* 在文件最后添加一个空字符，以便下面printf正常工作 */
	//printf("s %d",flength);
	mapped_mem = (char *)mmap(start_addr, flength, PROT_READ, //允许读
	MAP_PRIVATE, //不允许其它进程访问此内存区域
	fd, 0);
	/* 使用映射区域. */
	close(fd);
	char * d_input = NULL;
	checkCudaErrors(cudaMallocManaged((void **)&d_input,flength*sizeof(char)));
	cudaMemcpy(d_input,mapped_mem,flength*sizeof(char),cudaMemcpyHostToDevice);
	munmap(mapped_mem, flength);
	rezult rezults;
	rezults.input=d_input;
	rezults.length=flength;
	return rezults;
}

__host__ __device__ int leng(char* des){
   int len=0;
   while(true){
//	  printf("ok \n");
	  if((char)*(des+len)!='\0')
		len=len+1;
	  else
		break;
   }
   return len;
}

__global__ void finddes(char* value,long length,char* des,char *output,int* length_up,int* length_down)
{ long tid = blockIdx.x * blockDim.x + threadIdx.x;
  long pitch=blockDim.x*gridDim.x;
  long row_num=length;
  int  len=leng(des);
  for(long i=tid;i<row_num;i=i+pitch)
     {  int num=0;
	    for(int j=0;j<len;j++)
	      {if((char)(value[i+j])==(char)(des[j]))
	    	  {num++;}
	       else
	         break;
	      }

	    if(num==len)
	    { if (i-(*length_down*len)>0 and i+(*length_up*len)<row_num)
	       { int l=0;
	         for(int j=i-(*length_up*len);j<i+(*length_up*len);j++)
	           { output[l]=value[j];
	             l++;
	           }
	         output[l+1]='\0';
	       }
	       else
	       output[0]='#';
	    }
     }
};


int main(int argc, char **argv)
  {
//	struct tm tm_time;
//	strptime("2015-12-10 15:18:10.000000", "%Y-%m-%d %H:%M:%S", &tm_time);
//	printf("%ld \n", mktime(&tm_time));
//	printf("------------------------------------- \n");
//
//	char szBuf[256] = {0};
//	time_t timer = StringToDatetime("2015-12-10 15:18:10.000000");
//	strftime(szBuf, sizeof(szBuf), "%Y-%m-%d %H:%M:%S",localtime(&timer));
//	printf("%s \n", szBuf);
//
//	strptime("2015-12-10 15:18:10", "%Y-%m-%d %H:%M:%S", &tm_time);
//	printf("%ld \n", mktime(&tm_time));
//	printf("------------------------------------- \n");
//	return 0;
	rezult re;
	re=fileinput("/lf/2017.11.14/total/G_CFMY_1_002FQ001.txt");

	char* h_des="2016-11-22 6:0:16.046000";
	char* des;
	checkCudaErrors(cudaMallocManaged((void **)&des,100*sizeof(char)));
	cudaMemcpy(des,h_des,leng(h_des)*sizeof(char),cudaMemcpyHostToDevice);
	printf("len:=%d",leng(h_des));

    int* des_length_up;
	checkCudaErrors(cudaMallocManaged((void **)&des_length_up,sizeof(int)));
	*des_length_up=100;

    int* des_length_down;
	checkCudaErrors(cudaMallocManaged((void **)&des_length_down,sizeof(int)));
	*des_length_up=100;
    /////////////////////////////////////////////////////////////

	char * d_input =re.input;
	char * h_output = NULL;
	h_output=(char *)malloc(re.length*sizeof(char));
	cudaMemcpy(h_output,d_input,re.length*sizeof(char),cudaMemcpyDeviceToHost);

    //gpu
	int GPU_N;
	cudaGetDeviceCount(&GPU_N);
	printf("gpu:= %d \n",GPU_N);
	long pitch=(long)(re.length/GPU_N);
	long last_pitch=re.length-pitch*GPU_N;

	long *cut_num=new long[GPU_N];//每一段的结束位置
	for(int i=0;i<GPU_N;i++)
	{if(i!=GPU_N-1 and i!=0)
	   {int j=1;
		while(true)
		  {if ((char)h_output[*(cut_num+i-1)+pitch+j]!='\n')
			  j=j+1;
		   else
			  break;
		  }
		*(cut_num+i)=*(cut_num+i-1)+pitch+j;
	   }
	   else
	   { if(i==GPU_N-1)
		 {*(cut_num+i)=re.length;}
		 else
		 {int j=0;
			while(true)
			  {if ((char)h_output[pitch+j]!='\n')
				  j=j+1;
			   else
				  break;
			  }
			*(cut_num+i)=pitch+j;
		 }
	   }
	}

	for(int i=0;i<GPU_N;i++)
	{printf("第%ld个结束位置 \n",*(cut_num+i));
	 printf("第一个字符%c\n",(char)h_output[*(cut_num+i)+1]);
	}

	////////////需要使用到的参数配置//////
	char* output;
	checkCudaErrors(cudaMallocManaged((void **)&output,10000*sizeof(char)));

    /////////////////////////////////////////////////////////////
	for(long i=0;i<GPU_N;i++)
		{ if(i==0)
		  {cudaSetDevice(i);
		   finddes<<<224,256>>>(d_input,*(cut_num),des,output,des_length_up,des_length_down);}
		   else{
		   cudaSetDevice(i);
		   finddes<<<224,256>>>(d_input+*(cut_num+i-1)+1,*(cut_num+i)-*(cut_num+i-1),des,output,des_length_up,des_length_down);
		  }
		}
	checkCudaErrors(cudaDeviceSynchronize());

//	for(int i=0;i<1000;i++)
//	{if(h_output[i]=='\n')
//		{printf("换行");
//		 printf("\n");}
// 	  else
//	     printf("%c",h_output[i]); /* 为了保证这里工作正常，参数传递的文件名最好是一个文本文件 */
//	}
    printf("结果输出：＝＝＝＝");
	for(int i=0;i<leng(output);i++)
	{printf("%c",output[i]);}
	printf("a");
  }

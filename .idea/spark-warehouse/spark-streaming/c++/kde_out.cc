/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include<iostream>

using namespace tensorflow;
using namespace std;

REGISTER_OP("KdeOut")
      .Input("ekf_op_in: float")
      .Input("ekf_shape: int32")//2个shape长度
      .Output("ekf_op_out: float")
      .Attr("p_value: float")
      .Attr("pitch: float");

using namespace tensorflow;

class KdeOutOp : public OpKernel {
  public:
	  explicit KdeOutOp(OpKernelConstruction* context):OpKernel(context){
			OP_REQUIRES_OK(context,context->GetAttr("pitch", &pitch_N));
			OP_REQUIRES_OK(context,context->GetAttr("p_value",&p_value_N));
	  }

  float normal_probability_density(const Tensor& y,float x){
      float pi=3.141592654;
      auto y_flat = y.flat<float>();
      int tensor_len=y_flat.size();
      float h=std::pow(1.0/tensor_len,0.2);
      float result_out=0.0;
      for(int i=0;i<tensor_len;i++){
    	 result_out= result_out+std::exp(-1*std::pow(x-y_flat(i),2)/(std::pow(h,2)*2.0))/(std::pow(pi*2.0,0.5)*h);
      }
      return (float)result_out/tensor_len;
  }

  float min_T(const Tensor& y){
	  auto y_flat = y.flat<float>();
	  int tensor_len=y_flat.size();
	  float min_l=1000000.0;
	  for(int i=0;i<tensor_len;i++){
		  if (min_l>y_flat(i)){
			  min_l=y_flat(i);
		  }
	  }
	  return min_l;
  }

  float max_T(const Tensor& y){
	  auto y_flat = y.flat<float>();
	  int tensor_len=y_flat.size();
	  float max_l=-1000000;
	  for(int i=0;i<tensor_len;i++){
		  if (max_l<y_flat(i)){
			  max_l=y_flat(i);
		  }
	  }
	  return max_l;
  }

  float* normal_probability(const Tensor& y,float p=0.95,float pitch=0.5){
	  float max=max_T(y);
	  float min=min_T(y);
	  max=max+10.0;
	  min=min-10.0;
	  float p_now_np=0.0;//最新的p值
	  float p_last_np=0.0;//上次的p值
	  float p_add=0.0;//p_add=p_now_np-p_last_np

	  float value_now=max;
	  float value_big=max;
	  float value_little=min;
	  float last_value=max;//上一次的p值
	  float min_cast=value_little;//下界
	  float dx=pitch;
	  int n=(int)((max-min)/dx);
      int if_first=1;
      float dis_length=0.0;
      std::cout<<"开始迭代：=========="<<endl;
	  while(true){
		   if (if_first==1)
		      { for(int i=0;i<n-1;i++){
			        p_add=p_add+normal_probability_density(y,i*dx+min_cast+dx/2)*dx;
		           }//求积分计算分位数点
		           if_first=if_first+1;
		           p_now_np=p_add;
		           p_last_np=p_now_np;
		           last_value=value_now;
		           p_add=0;
		      } //第一次
		   else{
			  if(value_now>last_value)//只计算减少的概率部分
			  { dis_length=value_now-last_value;//两个区间之间的长度
				if(dis_length<1)
				  {n=100;
				   dx=dis_length/n;}
			    else
			       n=(int)(dis_length/dx);
			    min_cast=last_value;//替换当前值
			    for(int i=0;i<n-1;i++){
			    	     p_add=p_add+normal_probability_density(y,i*dx+min_cast+dx/2)*dx;
			    		           }//求积分计算分位数点
			    if_first=if_first+1;
			    p_now_np=p_last_np+p_add;
			    p_last_np=p_now_np;
			    last_value=value_now;
			    p_add=0;
			  }
			  else{
				  if(value_now<last_value)//只计算增加的概率部分
				  { dis_length=last_value-value_now;//两个区间之间的长度
					if(dis_length<1)
					  {n=100;
					   dx=dis_length/n;}
				    else
				       n=(int)(dis_length/dx);
				    min_cast=value_now;//替换当前值
				    for(int i=0;i<n-1;i++){
				    	    p_add=p_add+normal_probability_density(y,i*dx+min_cast+dx/2)*dx;
				    		           }//求积分计算分位数点
				    if_first=if_first+1;
				    p_now_np=p_last_np-p_add;
				    p_last_np=p_now_np;
				    last_value=value_now;
				    p_add=0;
			     }
		      }
		    }

		   std::cout<<"当前概率"<<p_now_np<<endl;
		   if(std::abs(p_now_np-p)<0.005)
			  break;
		   else
			  {if(p_now_np>p)
				 {value_big=value_now;
				  value_now=(value_big+value_little)/2;
				 }
				else
				  {value_little=value_now;
				   value_now=(value_big+value_little)/2;
				  }
			   }
	  }
      float * rezults_N=new float(2);
      rezults_N[0]=p_now_np;
      rezults_N[1]=value_now;
	  return rezults_N;
  }

  void Compute(OpKernelContext* context) override {
	// Check that preserve_index is positive
	//	OP_REQUIRES(context, p_value_down_N <= 1.0,
	//				errors::InvalidArgument("Need p_value_down_N <= 1, got ",
	//										p_value_down_N));
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& input_tensor_shape = context->input(1);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,input_tensor_shape.shape(),&output_tensor));
    auto output_flat = output_tensor->flat<float>();
    float* a=new float[2];
    a=normal_probability(input_tensor,p_value_N,pitch_N);
    output_flat(0)=a[0];
    output_flat(1)=a[1];
    delete a;
  }
 private:
   float p_value_N=0.95;
   float pitch_N=0.5;
};

REGISTER_KERNEL_BUILDER(Name("KdeOut").Device(DEVICE_CPU), KdeOutOp);

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

REGISTER_OP("EfkOut")
      .Input("ekf_op_in: float")
      .Input("ekf_shape: int32")//2个shape长度
      .Output("ekf_op_out: float")
      .Attr("c: float")
      .Attr("Q: float")
      .Attr("T: float")
      .Attr("H: float")
      .Attr("Z: float")
      .Attr("d: float");

using namespace tensorflow;

class EfkOutOp : public OpKernel {
  public:
	  explicit EfkOutOp(OpKernelConstruction* context):OpKernel(context){
			OP_REQUIRES_OK(context,context->GetAttr("c",&c_N));
			OP_REQUIRES_OK(context,context->GetAttr("Q",&Q_N));
			OP_REQUIRES_OK(context,context->GetAttr("T",&T_N));
			OP_REQUIRES_OK(context,context->GetAttr("H",&H_N));
			OP_REQUIRES_OK(context,context->GetAttr("Z",&Z_N));
			OP_REQUIRES_OK(context,context->GetAttr("d",&d_N));
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

  void efk_run(const Tensor& y,Tensor* out_put,float c,float Q,float T,float H,float Z,float d)
  {auto batch_ys=y.flat<float>();
   auto out_put_xs=out_put->flat<float>();
   int lenght=batch_ys.size();
   float a_t_t_1=0.0;
   float p_t_t_1=0.0;
   float F=0.0;

   float a_t=0.0;
   float p_t_1=0.0;
   float rep=0.0;

   for(int i=0;i<lenght;i++)
   {if (i==0)
       {  a_t_t_1=T*batch_ys(0)+c;
		  p_t_t_1=T*Q*T+Q;
          F=Z*p_t_t_1*Z+H;
		  a_t=a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys(0)-Z*a_t_t_1-d);
		  p_t_1=p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1;
          //#预测的y_st
		  rep=Z*a_t+d;
		  out_put_xs(i*3)=batch_ys(0);
		  out_put_xs(i*3+1)=rep;
		  out_put_xs(i*3+2)=batch_ys(0)-rep;
       }
	   else
	   {   a_t_t_1=T*a_t+c;
		   p_t_t_1=T*p_t_1*T+Q;
		   F=Z*p_t_t_1*Z+H;
		   a_t=a_t_t_1+p_t_t_1*Z/F*Z*(batch_ys(i)-Z*a_t_t_1-d);
		   p_t_1=p_t_t_1-p_t_t_1*Z/F*Z*p_t_t_1;
           //#预测的y_st
		   rep=Z*a_t+d;
		   out_put_xs(i*3)=batch_ys(i);
		   out_put_xs(i*3+1)=rep;
		   out_put_xs(i*3+2)=batch_ys(i)-rep;
	   }
     }
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
    efk_run(input_tensor,output_tensor,this->c_N,this->Q_N,this->T_N,this->H_N,this->Z_N,this->d_N);
  }
 private:
   float c_N=0.0001;
   float Q_N=0.05;
   float T_N=0.05;
   float H_N=0.01;
   float Z_N=0.95;
   float d_N=0.0001;
};

REGISTER_KERNEL_BUILDER(Name("EfkOut").Device(DEVICE_CPU), EfkOutOp);

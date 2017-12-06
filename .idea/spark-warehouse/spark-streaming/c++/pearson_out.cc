#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include <cmath>
#include "math.h"
#include <iostream>

using namespace tensorflow;
using namespace std;

REGISTER_OP("PearsonOut")
           .Input("pearson_x_op: float")
           .Input("pearson_y_op: float")
           .Input("pearson_shape: float")
           .Output("pearson_op_out: float");

using namespace tensorflow;

class PearsonOutOp : public OpKernel {
  public:
	  explicit PearsonOutOp(OpKernelConstruction* context):OpKernel(context){
//			OP_REQUIRES_OK(context,context->GetAttr("c",&c_N));
//			OP_REQUIRES_OK(context,context->GetAttr("Q",&Q_N));
//			OP_REQUIRES_OK(context,context->GetAttr("T",&T_N));
//			OP_REQUIRES_OK(context,context->GetAttr("H",&H_N));
//			OP_REQUIRES_OK(context,context->GetAttr("Z",&Z_N));
//			OP_REQUIRES_OK(context,context->GetAttr("d",&d_N));
	  }

  float sum_N(const Tensor& x){
	  auto x_f=x.flat<float>();
	  int lenght=x_f.size();
	  float result=0;
	  for(int i=0;i<lenght;i++)
		 result=result+x_f(i);
	  return result;
  }

  float sum_power(const Tensor& x){
	  auto x_f=x.flat<float>();
	  int lenght=x_f.size();
	  float result=0;
	  for(int i=0;i<lenght;i++)
		 result=result+pow(x_f(i),2);
	  return result;
  }

  float sumofxy_n(const Tensor& x,const Tensor& y){
	auto x_f=x.flat<float>();
	auto y_f=y.flat<float>();
	float sumofab=0.0;
	int length=x_f.size();
	for(int i=0;i<length;i++)
	   {sumofab=sumofab+x_f(i)*y_f(i);}
	return sumofab;
  }

  float Pearson_run(const Tensor& y,const Tensor& x)
  {auto x_f=x.flat<float>();
   auto y_f=y.flat<float>();
   long lenght=x_f.size();
   //求和
   float sum1=sum_N(x);
   float sum2=sum_N(y);
   //#求乘积之和
   float sumofxy=sumofxy_n(x,y);
   //#求平方和
   float sumofx2 = sum_power(x);
   float sumofy2 = sum_power(y);
   float num=sumofxy-(sum1*sum2/lenght);
   //#计算皮尔逊相关系数
   float den=sqrt((sumofx2-pow(sum1,2)/lenght)*(sumofy2-pow(sum2,2)/lenght));
   float result=num/den;
   return result;
   }

  void Compute(OpKernelContext* context) override {
	// Check that preserve_index is positive
	//	OP_REQUIRES(context, p_value_down_N <= 1.0,
	//				errors::InvalidArgument("Need p_value_down_N <= 1, got ",
	//										p_value_down_N));
    // Grab the input tensor
    const Tensor& input_x = context->input(0);
    const Tensor& input_y = context->input(1);
    const Tensor& input_shape=context->input(2);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,input_shape.shape(),&output_tensor));
    auto output_flat = output_tensor->flat<float>();
    output_flat(0)=Pearson_run(input_x,input_y);
  }
//  private:
};

REGISTER_KERNEL_BUILDER(Name("PearsonOut").Device(DEVICE_CPU), PearsonOutOp);



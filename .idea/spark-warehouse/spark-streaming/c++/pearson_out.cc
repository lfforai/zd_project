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
           .Input("model_type: float") //1欧式距离
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

  //1 欧式距离Eucledian Distance
  float Eucledian_run(const Tensor& y,const Tensor& x)
   {auto x_f=x.flat<float>();
    auto y_f=y.flat<float>();
    long lenght=x_f.size();
    //求和
	float result=0;
	for(int i=0;i<lenght;i++)
		 result=result+pow(x_f(i)-y_f(i),2);
    return result;
    }

  //2 Pearson Distance
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

  //3 Manhattan Distance
  float Manhattan_run(const Tensor& y,const Tensor& x)
   {auto x_f=x.flat<float>();
    auto y_f=y.flat<float>();
    long lenght=x_f.size();
    //求和
	float result=0;
	for(int i=0;i<lenght;i++)
		 result=result+abs(x_f(i)-y_f(i));
    return result;
    }

  //4 Cosine Distance 余旋距离
  float Cosine_run(const Tensor& y,const Tensor& x)
   {auto x_f=x.flat<float>();
    auto y_f=y.flat<float>();
    long lenght=x_f.size();
    //求和
	float result=0;

	float temp1=0.0;
	for(int i=0;i<lenght;i++)
		temp1=temp1+pow(x_f(i),2);
	temp1=pow(temp1,0.5);

	float temp2=0.0;
	for(int i=0;i<lenght;i++)
        temp2=temp2+pow(y_f(i),2);
	temp2=pow(temp2,0.5);

	float temp3=0.0;
	for(int i=0;i<lenght;i++)
        temp3=temp3+y_f(i)*x_f(i);
	result=temp3/(temp2*temp1);
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
    const Tensor& modle_type=context->input(3);
    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0,input_shape.shape(),&output_tensor));
    auto output_flat = output_tensor->flat<float>();
    auto modle_type_f=modle_type.flat<float>();
    switch(int(modle_type_f(0)))
        {case 1: {output_flat(0)=Eucledian_run(input_x,input_y);break;}
         case 2: {output_flat(0)=Pearson_run(input_x,input_y);break;}
         case 3: {output_flat(0)=Manhattan_run(input_x,input_y);break;}
         case 4: {output_flat(0)=Cosine_run(input_x,input_y);break;}
         default: break;
        }
  }
//  private:
};

REGISTER_KERNEL_BUILDER(Name("PearsonOut").Device(DEVICE_CPU), PearsonOutOp);



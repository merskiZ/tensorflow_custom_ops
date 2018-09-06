// The file to make a costumized multiply operation which takes in 
// 2 scalers and output the product of them.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("MulCustom")
		.Input("stride_slice: int32")
		.Input("y: int32")
		.Output("stack : int32")
		.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c)
			{c->set_output(0, c->input(0));
				return Status::OK();
			});

class MulCustomOp: public OpKernel
{
public:
  explicit MulCustomOp(OpKernelConstruction* context) : OpKernel(context) {}

	void Compute(OpKernelContext* context) override
	{
		// Grab the input tensor
		const Tensor& stride_slice_tensor = context->input(0);
		auto input_stride_slice = stride_slice_tensor.flat<int32>();

		const Tensor& y_tensor = context->input(1);
		auto input_y = y_tensor.flat<int32>();

		// Create an output tensor
		Tensor* output_tensor = NULL;
		OP_REQUIRES_OK(context, context->allocate_output(0, 
								stride_slice_tensor.shape(), 
								&output_tensor));
		auto output_flat = output_tensor->flat<int32>();

		// Set all but the first element of the output tensor to 0
		const int N = input_stride_slice.size();
		for (int i = 1; i < N; i++)
		{
			output_flat(i) = input_y(i) * input_stride_slice(i);
		}
	}
};

REGISTER_KERNEL_BUILDER(Name("MulCustom").Device(DEVICE_CPU), MulCustomOp);

#include "CommonProcess.h"
#include "US3DLog.h"

CommonProcess::CommonProcess(int isCPUOnly, std::string deviceString)
{
	mIsCPUOnly = isCPUOnly;
	mDeviceString = deviceString;
}

CommonProcess::~CommonProcess()
{
}

// Input - {1,1,a,b,c} Tensor, Output - {1,1,a,b,c} Tensor
int CommonProcess::gaussian3DFilter(at::Tensor& input, at::Tensor& output, int kernelSize, int GPUAcc)        // 1 x 1 x 3 x 3 x 3  Tensor
{
	std::vector<float> kernel3 = { 0.0206, 0.0339, 0.0206, 0.0339, 0.0560, 0.0339, 0.0206, 0.0339, 0.0206,
									0.0339, 0.0560, 0.0339, 0.0560, 0.0923, 0.0560, 0.0339, 0.0560, 0.0339,
									0.0206, 0.0339, 0.0206, 0.0339, 0.0560, 0.0339, 0.0206, 0.0339, 0.0206 };

	//std::vector<float> kernel3 = { 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037,
	//							0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037,
	//							0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037, 0.037 };

	std::vector<float> kernel5 = { 0.0002, 0.0007, 0.0012, 0.0007, 0.0002, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0012, 0.0054, 0.0088, 0.0054, 0.0012, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0002, 0.0007, 0.0012, 0.0007, 0.0002,
									0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0032, 0.0146, 0.0240, 0.0146, 0.0032, 0.0054, 0.0240, 0.0396, 0.0240, 0.0054, 0.0032, 0.0146, 0.0240, 0.0146, 0.0032, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007,
									0.0012, 0.0054, 0.0088, 0.0054, 0.0012, 0.0054, 0.0240, 0.0396, 0.0240, 0.0054, 0.0088, 0.0396, 0.0653, 0.0396, 0.0088, 0.0054, 0.0240, 0.0396, 0.0240, 0.0054, 0.0012, 0.0054, 0.0088, 0.0054, 0.0012,
									0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0032, 0.0146, 0.0240, 0.0146, 0.0032, 0.0054, 0.0240, 0.0396, 0.0240, 0.0054, 0.0032, 0.0146, 0.0240, 0.0146, 0.0032, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007,
									0.0002, 0.0007, 0.0012, 0.0007, 0.0002, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0012, 0.0054, 0.0088, 0.0054, 0.0012, 0.0007, 0.0032, 0.0054, 0.0032, 0.0007, 0.0002, 0.0007, 0.0012, 0.0007, 0.0002 };
	try
	{
		if (GPUAcc == 1)
		{
			input = input.to(mDeviceString);
			if (kernelSize == 3)
			{
				at::Tensor kernel = torch::tensor(kernel3);
				kernel = torch::reshape(kernel, { 1,1, 3, 3, 3 }).to(mDeviceString);
				output = torch::nn::functional::conv3d(input, kernel, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1)).to(at::kCPU);           //5X5 일때는 padding 2, 3X3 일때는 padding 1
				output = output.ceil();
				return RET_OK;
			}
			else if (kernelSize == 5)
			{
				at::Tensor kernel = torch::tensor(kernel5);
				kernel = torch::reshape(kernel, { 1,1, 5, 5, 5 }).to(mDeviceString);
				output = torch::nn::functional::conv3d(input, kernel, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(2)).to(at::kCPU);           //5X5 일때는 padding 2, 3X3 일때는 padding 1
				output = output.ceil();
				return RET_OK;
			}
			else
			{
				return RET_FAILURE;
			}
		}
		else if (GPUAcc == 0)
		{
			if (kernelSize == 3)
			{
				at::Tensor kernel = torch::tensor(kernel3);
				kernel = torch::reshape(kernel, { 1,1, 3, 3, 3 });
				output = torch::nn::functional::conv3d(input, kernel, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1));           //5X5 일때는 padding 2, 3X3 일때는 padding 1
				output = output.ceil();
				return RET_OK;
			}
			else if (kernelSize == 5)
			{
				at::Tensor kernel = torch::tensor(kernel5);
				kernel = torch::reshape(kernel, { 1,1, 5, 5, 5 });
				output = torch::nn::functional::conv3d(input, kernel, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(2));           //5X5 일때는 padding 2, 3X3 일때는 padding 1
				output = output.ceil();
				return RET_OK;
			}
			else
			{
				return RET_FAILURE;
			}
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in gaussian3DFilter");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

// Input - {1,labelnum,a,b,c} Tensor, Output - {1,labelnum,a,b,c} Tensor
int CommonProcess::dilation3D(at::Tensor& input, at::Tensor& output, int labelNum, int volumeDimA, int volumeDimB, int volumeDimC, int* processOnLabel, int* labelPriorityTable, int GPUAcc)
{
	std::vector<float> kernelMorph = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	int dummyLabelNum = labelNum + 1;

	try
	{
		if (GPUAcc == 1)
		{
			input = input.to(mDeviceString);

			float frac = 0.037;     // = 1/27  3x3x3 기준
			at::Tensor kernelD = torch::tensor(kernelMorph);
			kernelD = torch::reshape(kernelD, { 1,1, 3, 3, 3 }).to(mDeviceString);
			at::Tensor DilationData;
			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				if (processOnLabel[labelIndex] == 1 || processOnLabel[labelIndex] == 3)
				{
					DilationData = torch::nn::functional::conv3d(input.index({ 0,labelIndex,Slice(),Slice() ,Slice() }).unsqueeze(0).unsqueeze(0), kernelD, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1)).to(mDeviceString);           //5X5 일때는 padding 2, 3X3 일때는 padding 1
					DilationData = torch::mul(DilationData, frac).to(mDeviceString);
					DilationData = DilationData.ceil().to(mDeviceString);
					output.index_put_({ 0,labelIndex, Slice(),Slice() ,Slice() },
						torch::mul(DilationData.index({ 0,0,Slice(),Slice() ,Slice() }), labelPriorityTable[labelIndex])).to(mDeviceString);
				}
				else
				{
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, input.index({ 0, labelIndex, Slice(),Slice() ,Slice() })).to(mDeviceString);
				}
			}
			at::Tensor argmaxD = output.argmax(1).to(mDeviceString);
			output = torch::zeros_like(output).scatter_(1, argmaxD.unsqueeze(1), 1.0).to(mDeviceString);
			return RET_OK;
		}
		else if (GPUAcc == 0)
		{
			float frac = 0.037;     // = 1/27  3x3x3 기준
			at::Tensor kernelD = torch::tensor(kernelMorph);
			kernelD = torch::reshape(kernelD, { 1,1, 3, 3, 3 });
			at::Tensor DilationData;
			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				if (processOnLabel[labelIndex] == 1 || processOnLabel[labelIndex] == 3)
				{
					DilationData = torch::nn::functional::conv3d(input.index({ 0,labelIndex,Slice(),Slice() ,Slice() }).unsqueeze(0).unsqueeze(0), kernelD, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1));           //5X5 일때는 padding 2, 3X3 일때는 padding 1
					DilationData = torch::mul(DilationData, frac);
					DilationData = DilationData.ceil();
					output.index_put_({ 0,labelIndex, Slice(),Slice() ,Slice() },
						torch::mul(DilationData.index({ 0,0,Slice(),Slice() ,Slice() }), labelPriorityTable[labelIndex]));
				}
				else
				{
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, input.index({ 0, labelIndex, Slice(),Slice() ,Slice() }));
				}
			}
			at::Tensor argmaxD = output.argmax(1);
			output = torch::zeros_like(output).scatter_(1, argmaxD.unsqueeze(1), 1.0);
			return RET_OK;
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in dilation3D");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

// Input - {1,labelnum,a,b,c} Tensor, Output - {1,labelnum,a,b,c} Tensor
int CommonProcess::erosion3D(at::Tensor& input, at::Tensor& output, int labelNum, int volumeDimA, int volumeDimB, int volumeDimC, int* processOnLabel, int GPUAcc)
{
	std::vector<float> kernelMorph = { 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };

	try
	{
		if (GPUAcc == 1)
		{
			input = input.to(mDeviceString);

			float frac = 0.037;     // = 1/27  3x3x3 기준
			at::Tensor kernelD = torch::tensor(kernelMorph);
			kernelD = torch::reshape(kernelD, { 1,1, 3, 3, 3 }).to(mDeviceString);

			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				at::Tensor erosionData;
				if (processOnLabel[labelIndex] == 2 || processOnLabel[labelIndex] == 3)
				{
					erosionData = torch::nn::functional::conv3d(input.index({ 0,labelIndex,Slice(),Slice() ,Slice() }).unsqueeze(0).unsqueeze(0), kernelD, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1)).to(mDeviceString);           //5X5 일때는 padding 2, 3X3 일때는 padding 1
					erosionData = torch::mul(erosionData, frac).to(mDeviceString);
					erosionData = torch::threshold(erosionData, 0.98, 0).to(mDeviceString);
					erosionData = erosionData.ceil().to(mDeviceString);
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, erosionData.index({ 0,0,Slice(),Slice() ,Slice() })).to(mDeviceString);
				}
				else
				{
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, input.index({ 0,labelIndex,Slice(),Slice() ,Slice() })).to(mDeviceString);
				}
			}
			return RET_OK;
		}
		else if (GPUAcc == 0)
		{
			float frac = 0.037;     // = 1/27  3x3x3 기준
			at::Tensor kernelD = torch::tensor(kernelMorph);
			kernelD = torch::reshape(kernelD, { 1,1, 3, 3, 3 });

			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				at::Tensor erosionData;
				if (processOnLabel[labelIndex] == 2 || processOnLabel[labelIndex] == 3)
				{
					erosionData = torch::nn::functional::conv3d(input.index({ 0,labelIndex,Slice(),Slice() ,Slice() }).unsqueeze(0).unsqueeze(0), kernelD, torch::nn::functional::Conv3dFuncOptions().stride(1).padding(1));           //5X5 일때는 padding 2, 3X3 일때는 padding 1
					erosionData = torch::mul(erosionData, frac);
					erosionData = torch::threshold(erosionData, 0.98, 0);
					erosionData = erosionData.ceil();
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, erosionData.index({ 0,0,Slice(),Slice() ,Slice() }));
				}
				else
				{
					output.index_put_({ 0, labelIndex, Slice(),Slice() ,Slice() }, input.index({ 0,labelIndex,Slice(),Slice() ,Slice() }));
				}
			}
			return RET_OK;
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in erosion3D");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

// Input - Rank 1(size = volume size)(Flatten Multi Label Volume), Output - eVec, sampleMean, eValuem Tensor
int CommonProcess::torchPCA(at::Tensor& input, int TargetLabel, at::Tensor& eVec, at::Tensor& sampleMean, at::Tensor& eValue, int volumeDimA, int volumeDimB, int volumeDimC)
{
	try
	{
		at::Tensor input_reshape = input.reshape({ volumeDimA, volumeDimB, volumeDimC });
		at::Tensor svdData = (input_reshape == TargetLabel).nonzero();

		if (svdData.size(0) > 0)
		{
			svdData = svdData.to(torch::kFloat);
			sampleMean = svdData.mean(0);
			svdData = torch::sub(svdData, sampleMean);
			auto outputs = torch::svd(svdData);
			eValue = std::get<1>(outputs);
			eVec = std::get<2>(outputs).transpose(0, 1);
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in torchPCA");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

//  Input - Rank 1(size = volume size)(Flatten Multi Label Volume), Output - {1,labelnum,a,b,c} Tensor
int CommonProcess::toSingleLabels(at::Tensor& input, at::Tensor& output, int labelNum, int* labelIndex, int volumeDimA, int volumeDimB, int volumeDimC)
{
	int VolumeSize = volumeDimA * volumeDimB * volumeDimC;
	try
	{
		for (int i = 1; i < labelNum; i++)
		{
			at::Tensor input_temp(torch::zeros({ VolumeSize }));

			at::Tensor nonz_t = (input == labelIndex[i]).nonzero();
			input_temp.index_put_({ nonz_t }, 1);

			at::Tensor input_temp2 = input_temp.reshape({ volumeDimA, volumeDimB, volumeDimC }).unsqueeze(0).unsqueeze(0);

			output.index_put_({ 0, i, Slice(),Slice() ,Slice() }, input_temp2.index({ 0,0,Slice(),Slice() ,Slice() }));
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in toSingleLabels");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CommonProcess::printTensorInfo(at::Tensor t) {
	torch::ScalarType dtype = t.scalar_type();

	switch (dtype) {
	case torch::kUInt8:
		LOG_T("type:kUInt8");
		break;
	case torch::kInt8:
		LOG_T("type:kInt8");
		break;
	case torch::kInt16:
		LOG_T("type:kInt16");
		break;
	case torch::kInt32:
		LOG_T("type:kInt32");
		break;
	case torch::kInt64:
		LOG_T("type:kInt64");
		break;
	case torch::kFloat:
		LOG_T("type:kFloat");
		break;
	case torch::kFloat16:
		LOG_T("type:kFloat16");
		break;
	default:
		LOG_T("type:etc");
		break;
	}

	for (int i = 0; i < t.dim(); i++) {
		LOG_T("Size[{}]={}", i, t.size(i));
	}
	return 0;
}
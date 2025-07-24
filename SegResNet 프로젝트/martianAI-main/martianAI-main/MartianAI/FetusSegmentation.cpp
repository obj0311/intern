#include "FetusSegmentation.h"
#include "US3DLog.h"

FetusSegmentation::FetusSegmentation(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess)
	:mCommonProcess(commonProcess)
{
	mIsCPUOnly = isCPUOnly;
	mDeviceString = deviceString;
}

FetusSegmentation::~FetusSegmentation()
{
}

int FetusSegmentation::ExcuteFetusSegmentation(torch::jit::script::Module module, at::Tensor& inputTensor, int mode, InferenceInfo* inferenceInfo, unsigned char* outputBuffer)
{
	int resultValue = true;
	if (mode == MARTIAN_SEG_FETUS_1ST)
	{
		resultValue = segResnet(module, inputTensor, inputTensor, outputBuffer, inferenceInfo->SegMean, mode, inferenceInfo->labelNum, inferenceInfo->thresholdTable, inferenceInfo->processOnTable, labelPriorityTable_1st, labelOffsetTable_1st, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize, inferenceInfo->gaussianFilterMode, inferenceInfo->morphologyMode, &inferenceInfo->errorCode, &inferenceInfo->returnValue);
		countVoxels(inputTensor, inferenceInfo->SegMean, inferenceInfo->labelNum, labelOffsetTable_1st, mLabelName1st);
	}
	else if (mode == MARTIAN_SEG_FETUS_2ND)
	{
		resultValue = segResnet(module, inputTensor, inputTensor, outputBuffer, inferenceInfo->SegMean, mode, inferenceInfo->labelNum, inferenceInfo->thresholdTable, inferenceInfo->processOnTable, labelPriorityTable_2nd, labelOffsetTable_2nd, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize, inferenceInfo->gaussianFilterMode, inferenceInfo->morphologyMode, &inferenceInfo->errorCode, &inferenceInfo->returnValue);
		countVoxels(inputTensor, inferenceInfo->SegMean, inferenceInfo->labelNum, labelOffsetTable_2nd, mLabelName2nd);
	}

	inferenceInfo->xSize = FETUSSEG_OUT_VOLUME_SIZE;
	inferenceInfo->ySize = FETUSSEG_OUT_VOLUME_SIZE;
	inferenceInfo->zSize = FETUSSEG_OUT_VOLUME_SIZE * 2;
	return resultValue;
}

// Input - Rank 1(size = volume size) Tensor, Output - {1,1,a,b,c} Tensor
int FetusSegmentation::segResnet(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffer, float* SegMeanOut, int mode, int labelNum, float* labelThreshold, int* processOnLabel, int* labelPriorityTable, int* labelOffsetTable, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue)
{
	int volumesize = volumeDimA * volumeDimB * volumeDimC;
	at::Tensor maxPVolume;

	try
	{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0)
		{
			input = input.to(mDeviceString);
		}

		if (volumeDimA != VOLUME_SIZE)
		{
			input = input.reshape({ 1, 1, volumeDimA, volumeDimB, volumeDimC });
			input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(std::vector<int64_t>({ VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE })));		//Image Resize
			input = input.flatten();
			volumeDimA = VOLUME_SIZE;
			volumeDimB = VOLUME_SIZE;
			volumeDimC = VOLUME_SIZE;
		}

		//libtorch ±â¹Ý Normalization
		at::Tensor nonz_t = input.nonzero();
		at::Tensor mean_t = input.index({ nonz_t }).mean();
		at::Tensor std_t = input.index({ nonz_t }).std();
		input.index_put_({ nonz_t }, (input.index({ nonz_t }) - mean_t) / std_t);

		std::vector<torch::jit::IValue> inputs;

		input = input.reshape({ 1, 1, volumeDimA, volumeDimB, volumeDimC });
		input = input.permute({ 0, 1, 4, 3, 2 });  // Volume Rotation

		inputs.push_back(input);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("segResnet Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor();
		c10::cuda::CUDACachingAllocator::emptyCache();

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("segResnet Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin_A = std::chrono::steady_clock::now();

		output = output.permute({ 0, 1, 4, 3, 2 });  // Volume Rotation

		if (output.size(1) != labelNum)
		{
			LOG_E("Check Label Num : labelNum({}), output({})", labelNum, output.size(1));
			return RET_FAILURE;
		}
		else
		{
			output = torch::sigmoid(output);	// SoftMax, Sigmoid result diff check
			//output = torch::softmax(output, 1);

			output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(std::vector<int64_t>({ FETUSSEG_OUT_VOLUME_SIZE, FETUSSEG_OUT_VOLUME_SIZE, FETUSSEG_OUT_VOLUME_SIZE })));		//Tempcode - severance CNS Test
			c10::cuda::CUDACachingAllocator::emptyCache();

			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				output.index_put_({ 0,labelIndex,Slice(),Slice(),Slice() }, torch::threshold(output.index({ 0,labelIndex,Slice(),Slice(),Slice() }), labelThreshold[labelIndex], 0));
			}

			at::Tensor PVolume;
			PVolume = output.amax(1);
			PVolume = torch::mul(PVolume, 255);
			c10::cuda::CUDACachingAllocator::emptyCache();

			at::Tensor argmax = output.argmax(1);
			c10::cuda::CUDACachingAllocator::emptyCache();

			output = torch::zeros_like(output).scatter_(1, argmax.unsqueeze(1), 1.0);
			c10::cuda::CUDACachingAllocator::emptyCache();

			int GPUAcc;
			if (mIsCPUOnly == 0)
				GPUAcc = 1;
			else
				GPUAcc = 0;

			int tempSize = output.size(2);
			if (morphProcessMode == 1 || morphProcessMode == 3)
			{
				if (mode == MARTIAN_SEG_FETUS_1ST)
				{
					mCommonProcess->dilation3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, labelPriorityTable_1st, GPUAcc);
				}
				else if (mode == MARTIAN_SEG_FETUS_2ND)
				{
					mCommonProcess->dilation3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, labelPriorityTable_2nd, GPUAcc);
				}
			}
			if (morphProcessMode == 2 || morphProcessMode == 3)
			{
				mCommonProcess->erosion3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, GPUAcc);
				//erosion3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, GPUAcc);		//Temp Code - 3D Rendering Test
			}

			// Labeling
			int processIndex = 0;
			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				output.index_put_({ 0,labelIndex,Slice(),Slice(),Slice() },
					torch::mul(output.index({ 0,labelIndex,Slice(),Slice(),Slice() }), labelOffsetTable[labelIndex]));   // Label Offset
			}
			//Merge
			output = torch::sum(output, 1).to(at::kCPU);

			output = output.to(at::kCPU);
			PVolume = PVolume.to(at::kCPU);

			output = output.to(torch::kUInt8);
			PVolume = PVolume.to(torch::kUInt8);

			unsigned char* outputfData1 = output.flatten().data_ptr<unsigned char>();
			unsigned char* outputfData2 = PVolume.flatten().data_ptr<unsigned char>();

			*errorCode = 100;
			*returnValue = 10;

			memcpy(outputBuffer, outputfData1, FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
			memcpy(outputBuffer + FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE, outputfData2, FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE * FETUSSEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
		}

		std::chrono::steady_clock::time_point End_A = std::chrono::steady_clock::now();
		LOG_I("segResnet Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(End_A - begin_A).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in segResnet");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int FetusSegmentation::countVoxels(at::Tensor& input, float* SegMeanOut, int labelNum, int* labelOffsetTable, char labelName[][4])
{
	try
	{
		for (int i = 1; i < labelNum; i++)
		{
			at::Tensor nonz_t = (input == labelOffsetTable[i]).nonzero();
			SegMeanOut[i - 1] = nonz_t.size(0);

			LOG_I("countVoxels {} = {}", labelName[i], SegMeanOut[i-1]);
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in countVoxels");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

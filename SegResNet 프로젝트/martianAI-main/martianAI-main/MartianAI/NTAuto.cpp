#include "NTAuto.h"
#include "US3DLog.h"

NTAuto::NTAuto(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess)
	:mCommonProcess(commonProcess)
{
	mIsCPUOnly = isCPUOnly;
	mDeviceString = deviceString;
}

NTAuto::~NTAuto()
{
}

int NTAuto::ExcuteNTAuto(torch::jit::script::Module module_head, torch::jit::script::Module module_inside, at::Tensor& inputTensor, InferenceInfo* inferenceInfo, unsigned char* outputBuffer)
{
	int resultValue = true;

	resultValue = segResnet_head(module_head, inputTensor, outputBuffer, inferenceInfo->SegMean, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize, inferenceInfo->gaussianFilterMode, inferenceInfo->morphologyMode, &inferenceInfo->errorCode, &inferenceInfo->returnValue);

	if (resultValue != RET_OK)
		return resultValue;

	resultValue = segResnet_inside(module_inside, inputTensor, inputTensor, outputBuffer, inferenceInfo->SegMean, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize, inferenceInfo->gaussianFilterMode, inferenceInfo->morphologyMode, &inferenceInfo->errorCode, &inferenceInfo->returnValue);
	
	if (resultValue != RET_OK)
		return resultValue;

	inferenceInfo->xSize = NTSEG_OUT_VOLUME_SIZE;
	inferenceInfo->ySize = NTSEG_OUT_VOLUME_SIZE;
	inferenceInfo->zSize = NTSEG_OUT_VOLUME_SIZE;
	
	inputTensor = inputTensor.to(torch::kFloat32);
	if (inferenceInfo->SegMean[0] > 0)
	{
		at::Tensor eVec;
		at::Tensor sampleMean;
		at::Tensor eValue;
		resultValue = mCommonProcess->torchPCA(inputTensor, NT_NUM_MID, eVec, sampleMean, eValue, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize);

		if (resultValue == RET_OK)
		{
			eVec = eVec.flatten();		// {3,3}의 경우 먼저 flatten 한번 해줘야 한다.
			float* outputEVec = eVec.flatten().data_ptr<float>();
			float* outputSampleMean = sampleMean.flatten().data_ptr<float>();

			for (int veci = 0; veci < 3; veci++)
			{
				inferenceInfo->pcaVector[veci * 3] = outputEVec[veci * 3 + 2];
				inferenceInfo->pcaVector[veci * 3 + 1] = outputEVec[veci * 3 + 1];
				inferenceInfo->pcaVector[veci * 3 + 2] = outputEVec[veci * 3];
			}
			inferenceInfo->pcaMean[0] = outputSampleMean[2];
			inferenceInfo->pcaMean[1] = outputSampleMean[1];
			inferenceInfo->pcaMean[2] = outputSampleMean[0];
		}
	}
	return resultValue;
}

int NTAuto::segResnet_head(torch::jit::script::Module module, at::Tensor& input, unsigned char* outputBuffer, float* SegMeanOut, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue)
{
	//at::Tensor maxPVolume;
	at::Tensor inputVol = input;
	at::Tensor output;

	try {
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0) {
			inputVol = inputVol.to(mDeviceString);
		}

		if (volumeDimA == NTSEG_IN_VOLUME_SIZE)
		{
			inputVol = inputVol.reshape({ 1, 1, volumeDimA, volumeDimB, volumeDimC });
			inputVol = inputVol.index({ Slice(), Slice(), Slice(16,240), Slice(16,240), Slice(16,240) });

			inputVol = torch::nn::functional::interpolate(inputVol, torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(std::vector<int64_t>({ VOLUME_SIZE, VOLUME_SIZE, VOLUME_SIZE })));        //Image Resize
			//input = input.flatten();
			volumeDimA = VOLUME_SIZE;
			volumeDimB = VOLUME_SIZE;
			volumeDimC = VOLUME_SIZE;
		}
		else
		{
			LOG_E("Input volume size is : {}", volumeDimA);
			return RET_FAILURE;
		}

		//libtorch 기반 Normalization (0~1)
		at::Tensor max_t = (torch::max)(inputVol);
		at::Tensor min_t = (torch::min)(inputVol);
		int minMaxDiff = static_cast<int>((max_t - min_t).item<float>());

		if (minMaxDiff <= 0)
		{
			LOG_E("error -> (max_int - min_int) = 0");
			return RET_FAILURE;
		}

		inputVol = (inputVol - min_t) / (max_t - min_t);

		std::vector<torch::jit::IValue> inputs;

		inputVol = inputVol.reshape({ 1, 1, volumeDimA, volumeDimB, volumeDimC });
		inputVol = inputVol.permute({ 0, 1, 4, 3, 2 });  // Volume Rotation

		inputs.push_back(inputVol);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("segResnet_head Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor(); //volume index : [batch, channel, Z, Y, X]

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("segResnet_head Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin_A = std::chrono::steady_clock::now();

		output = output.permute({ 0, 1, 4, 3, 2 }); //volume index : [batch, channel, X, Y, Z]

		if (output.size(1) != labelNum_NT_head)
		{
			LOG_E("Check Label Num : head labelNum(1), output({})", output.size(1));
			return RET_FAILURE;
		}
		else
		{
			output = torch::sigmoid(output); // head output uses sigmoid activation
			output = torch::where(output >= labelThreshold_NT[0], 1, 0); // head output >= 0.5 threshold

			int GPUAcc;
			if (mIsCPUOnly == 0)
				GPUAcc = 1;
			else
				GPUAcc = 0;

			int tempSize = output.size(2);
			if (morphProcessMode == 1 || morphProcessMode == 3)
			{
				mCommonProcess->dilation3D(output, output, labelNum_NT_head, tempSize, tempSize, tempSize, processOnLabel_NT, labelPriorityTable_NT, GPUAcc);
			}
			if (morphProcessMode == 2 || morphProcessMode == 3)
			{
				mCommonProcess->erosion3D(output, output, labelNum_NT_head, tempSize, tempSize, tempSize, processOnLabel_NT, GPUAcc);
			}

			int processIndex = 0; // head processIndex
			CcTensor = torch::zeros({ tempSize, tempSize, tempSize }).to(torch::kUInt8);
			vector<ComponentInfo> labelInfo;
			
			if (processIndex == NT_IDX_HEAD) {
				*errorCode = NT_ERR_SUCCESS;
				*returnValue = 0;

				output = output.to(at::kCPU);
				at::Tensor InTensor = output.index({ 0, processIndex, Slice(), Slice() ,Slice() }).to(torch::kUInt8).clone();

				// step 1: cc3d
				bool returnCentroid = true;
				cc3d(CcTensor, InTensor, labelInfo, ccLabelMap_NT[processIndex], returnCentroid);

				// Log for connected component labeling result
				LOG_T("{} label's components: {}", labelName_NT[processIndex], labelInfo.size());
				std::ostringstream stream;
				for (int i = 0; i < labelInfo.size(); i++) {
					stream.str(std::string());
					stream << "\t" << i + 1 << "th: " << labelInfo[i].volume << " voxel";
					if (returnCentroid) {
						stream << ", [" << labelInfo[i].centroid[0] << ", " << labelInfo[i].centroid[1] << ", " << labelInfo[i].centroid[2] << "]";
					}
					LOG_T("{}", stream.str());
				}

				// step 2: Check rule base
				// 2-1 check component exist (head)
				if (labelInfo.size() < 1 && RuleLabelExist_NT[processIndex])
				{
					*errorCode = NT_OUTPUT_NOT_EXIST;
					*returnValue = labelMap_NT[processIndex];
					LOG_E("{} label is not exist", labelName_NT[processIndex]);
					return RET_CHECK_ERROR_CODE;
				}

				// Sort descending order of voxel count array
				std::sort(labelInfo.begin(), labelInfo.end(), [](ComponentInfo a, ComponentInfo b) { return a.volume > b.volume; });

				// 2-2 check available component count
				// Removes a component if the number of components exceeds the maximum according to the rule.
				if (labelInfo.size() > RuleMaxComponentCount_NT[processIndex])
				{
					removeExcessComponents(output, CcTensor, processIndex, labelInfo, RuleMaxComponentCount_NT[processIndex], labelInfo.size());
				}

				if ((labelInfo.size() >= 1) && (RuleMinVolumeSize_NT[processIndex] > labelInfo[0].volume))
				{
					*errorCode = NT_OUTPUT_TOO_SMALL;
					*returnValue = labelMap_NT[processIndex];
					LOG_E("{} label is too small", labelName_NT[processIndex]);
					return RET_CHECK_ERROR_CODE;
				}

				if (mIsCPUOnly == 0)
				{
					output = output.to(mDeviceString);
					//maxPVolume = maxPVolume.to(mDeviceString);
				}
			}
			at::Tensor IndexData = output.index({ 0, processIndex, Slice(), Slice(), Slice() }).nonzero().to(torch::kFloat32);

			if (processIndex == NT_IDX_HEAD) {

				float MaximumRadius = 0;
				HeadMaximumRadius(IndexData, &MaximumRadius);
				SegMeanOut[49] = MaximumRadius * (224 / 128.); // From 256 crop to 224 and resize to 128
			}

			at::Tensor DataMean;
			if (IndexData.size(0) >= 1) {
				DataMean = IndexData.mean(0);
				HeadCenter[0] = DataMean[0].item<float>() * (224 / 128.) + 16.; // From 256 crop to 224 and resize to 128
				HeadCenter[1] = DataMean[1].item<float>() * (224 / 128.) + 16.;
				HeadCenter[2] = DataMean[2].item<float>() * (224 / 128.) + 16.;

				SegMeanOut[processIndex * 3 + 0] = HeadCenter[2];
				SegMeanOut[processIndex * 3 + 1] = HeadCenter[1];
				SegMeanOut[processIndex * 3 + 2] = HeadCenter[0];
			}
		}
		std::chrono::steady_clock::time_point End_A = std::chrono::steady_clock::now();
		LOG_I("segResnetNT_head Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(End_A - begin_A).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in segResnetNT_head");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int NTAuto::segResnet_inside(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffer, float* SegMeanOut, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue)
{
	at::Tensor maxPVolume;

	try
	{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0) {
			input = input.to(mDeviceString);
		}
		
		if (volumeDimA == NTSEG_IN_VOLUME_SIZE)
		{	
			// Cropping with head center
			for (int idx = 0; idx < 3; idx++) {
				int center = HeadCenter[idx];

				if (center - 64 < 0) {
					HeadCenter[idx] = 64;
				}
				else if (255 < center + 64) {
					HeadCenter[idx] = 191;
				}
				else {
					HeadCenter[idx] = center;
				}
			}
			int xL = HeadCenter[0] - 64;
			int xR = HeadCenter[0] + 64;
			int yL = HeadCenter[1] - 64;
			int yR = HeadCenter[1] + 64;
			int zL = HeadCenter[2] - 64;
			int zR = HeadCenter[2] + 64;

			input = input.reshape({ volumeDimA, volumeDimB, volumeDimC });
			input = input.index({ Slice(xL, xR), Slice(yL, yR), Slice(zL, zR) });

			volumeDimA = VOLUME_SIZE;
			volumeDimB = VOLUME_SIZE;
			volumeDimC = VOLUME_SIZE;
		}
		else
		{
			LOG_E("Input volume size is : {}", volumeDimA);
			return RET_FAILURE;
		}

		//libtorch 기반 Normalization (0~1)
		at::Tensor max_t = (torch::max)(input);
		at::Tensor min_t = (torch::min)(input);
		int minMaxDiff = static_cast<int>((max_t - min_t).item<float>());

		if (minMaxDiff <= 0)
		{
			LOG_E("error -> (max_int - min_int) = 0");
			return RET_FAILURE;
		}
		input = (input - min_t) / (max_t - min_t);

		std::vector<torch::jit::IValue> inputs;

		input = input.reshape({ 1, 1, volumeDimA, volumeDimB, volumeDimC });
		input = input.permute({ 0, 1, 4, 3, 2 });  // Volume Rotation

		inputs.push_back(input);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("segResnet_inside Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor(); //volume index : [batch, channel, Z, Y, X]

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("segResnet_inside Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin_A = std::chrono::steady_clock::now();

		output = output.permute({ 0, 1, 4, 3, 2 }); //volume index : [batch, channel, X, Y, Z]

		if (output.size(1) != labelNum_NT_inside)
		{
			LOG_E("Check Label Num : Inside labelNum({}), output({})", labelNum_NT_inside, output.size(1));
			return RET_FAILURE;
		}
		else
		{
			output = torch::softmax(output, 1);// inside output (BG, ML, NT, DCP, NB ) : [batch, 1:7, X, Y, Z] -> softmax
			at::Tensor maxPVolume = output.clone();

			for (int labelIndex = 1; labelIndex < labelNum_NT_inside; labelIndex++) // labelIndex = 1 -> remove background
			{
				int postIndex = labelIndex + 1;
				float maxProbability;
				float threshold;

				maxProbability = output.index({ 0, labelIndex, Slice(), Slice(), Slice() }).amax().flatten()[0].item<float>();
				maxProbability = floor(maxProbability * 100) / 100;
				threshold = ((labelIndex != NT_IDX_NT) && (labelIndex != NT_IDX_NB)) ? maxProbability : labelThreshold_NT[postIndex];
				maxPVolume.index_put_({ 0,labelIndex,Slice(),Slice(),Slice() }, torch::threshold(maxPVolume.index({ 0,labelIndex,Slice(),Slice(),Slice() }), threshold, 0));
				LOG_T("{} maxprobability :{}, threshold :{} ", labelName_NT[postIndex], maxProbability, threshold);
			}

			at::Tensor argmax = output.argmax(1);
			output = torch::zeros_like(output).scatter_(1, argmax.unsqueeze(1), 1.0);
			LOG_T("output size: {}, {}, {}, {}, {}", output.size(0), output.size(1), output.size(2), output.size(3), output.size(4));

			int GPUAcc;
			if (mIsCPUOnly == 0)
				GPUAcc = 1;
			else
				GPUAcc = 0;

			int tempSize = output.size(2);
			if (morphProcessMode == 1 || morphProcessMode == 3)
			{
				mCommonProcess->dilation3D(output, output, labelNum_NT_head, tempSize, tempSize, tempSize, processOnLabel_NT, labelPriorityTable_NT, GPUAcc);
			}
			if (morphProcessMode == 2 || morphProcessMode == 3)
			{
				mCommonProcess->erosion3D(output, output, labelNum_NT_head, tempSize, tempSize, tempSize, processOnLabel_NT, GPUAcc);
			}
			// cc3d and Labeling 
			at::Tensor CcTensor = torch::zeros({ volumeDimA, volumeDimB, volumeDimC }).to(torch::kUInt8);
			int processIndex = 0;
			int postIndex = 0;
			for (int labelIndex = 1; labelIndex < labelNum_NT_inside; labelIndex++)
			{
				*errorCode = NT_ERR_SUCCESS;
				*returnValue = 0;

				processIndex = processOrderInside[labelIndex]; // post-process order bg(0), mid(1) -> dcp(3) -> nt(2) -> nb(4)
				postIndex = processIndex + 1;
				vector<ComponentInfo> labelInfo;
				output = output.to(at::kCPU);
				at::Tensor InTensor = output.index({ 0, processIndex, Slice(), Slice() ,Slice() }).to(torch::kUInt8).clone();

				// step 1: cc3d
				// Connected Component Labeling
				bool returnCentroid = true;
				cc3d(CcTensor, InTensor, labelInfo, ccLabelMap_NT[postIndex], returnCentroid);

				// Log for connected component labeling result
				LOG_T("{} label's components: {}", labelName_NT[postIndex], labelInfo.size());
				std::ostringstream stream;
				for (int i = 0; i < labelInfo.size(); i++)
				{
					stream.str(std::string());
					stream << "\t" << i + 1 << "th: " << labelInfo[i].volume << " voxel";
					if (returnCentroid) {
						stream << ", [" << labelInfo[i].centroid[0] << ", " << labelInfo[i].centroid[1] << ", " << labelInfo[i].centroid[2] << "]";
					}
					LOG_T("{}", stream.str());
				}

				// step 2: Check rule base
				// 2-1 check component exist (Mid line)
				if (labelInfo.size() < 1 && RuleLabelExist_NT[postIndex])
				{
					*errorCode = NT_OUTPUT_NOT_EXIST;
					*returnValue = labelMap_NT[postIndex];
					LOG_E("{} label is not exist", labelName_NT[postIndex]);
					return RET_CHECK_ERROR_CODE;
				}

				// Sort descending order of voxel count array
				std::sort(labelInfo.begin(), labelInfo.end(), [](ComponentInfo a, ComponentInfo b) { return a.volume > b.volume; });

				// 2-2 check available component count
				// Removes a component if the number of components exceeds the maximum according to the rule.
				if (labelInfo.size() > RuleMaxComponentCount_NT[postIndex])
				{
					removeExcessComponents(output, CcTensor, processIndex, labelInfo, RuleMaxComponentCount_NT[postIndex], labelInfo.size());
				}

				if (processIndex == NT_IDX_MID && labelInfo.size() >= 1 && RuleMinVolumeSize_NT[postIndex] > labelInfo[0].volume)
				{
					*errorCode = NT_OUTPUT_TOO_SMALL;
					*returnValue = labelMap_NT[postIndex];
					LOG_E("{} label is too small", labelName_NT[postIndex]);
					return RET_CHECK_ERROR_CODE;
				}

				if (mIsCPUOnly == 0)
				{
					output = output.to(mDeviceString);
					maxPVolume = maxPVolume.to(mDeviceString);
				}
				// Update maxPVolume Tensor based on the intersection of 1.0 values in output Tensor.
				maxPVolume = maxPVolume * output;

				at::Tensor IndexData = output.index({ 0,processIndex,Slice(),Slice(),Slice() }).nonzero().to(torch::kFloat32);
				at::Tensor MaxPIndexData = maxPVolume.index({ 0,processIndex,Slice(),Slice(),Slice() }).nonzero().to(torch::kFloat32);
				LOG_T("MaxPIndexData size: {}, IndexData size: {}", MaxPIndexData.size(0), IndexData.size(0));

				at::Tensor DataMean;
				if (MaxPIndexData.size(0) >= 1) {
					DataMean = MaxPIndexData.mean(0);
				}
				else if (IndexData.size(0) >= 1) {
					DataMean = IndexData.mean(0);
				}
				else {
					SegMeanOut[processIndex * 3 + 0] = 0;
					SegMeanOut[processIndex * 3 + 1] = 0;
					SegMeanOut[processIndex * 3 + 2] = 0;
				}

				if (DataMean.size(0) == 3) {
					SegMeanOut[processIndex * 3 + 0] = DataMean[2].item<float>() + (HeadCenter[2] - 64); // From 256^3 to crop and 128^3
					SegMeanOut[processIndex * 3 + 1] = DataMean[1].item<float>() + (HeadCenter[1] - 64);
					SegMeanOut[processIndex * 3 + 2] = DataMean[0].item<float>() + (HeadCenter[0] - 64);
				}

				output.index_put_({ 0, processIndex, Slice(), Slice(), Slice() },
					torch::mul(output.index({ 0,processIndex,Slice(),Slice(),Slice() }), labelOffsetTable_NT[postIndex]));
			}
			//Merge
			output = torch::sum(output.index({ Slice(), Slice(1,5), Slice(), Slice(), Slice() }), 1).to(at::kCPU);  // visualize only inside_output
			LOG_T("output size: {}, {}, {}", output.size(0), output.size(1), output.size(2));

			output = output.to(at::kCPU);
			output = output.to(torch::kUInt8);

			unsigned char* outputfData1 = output.flatten().data_ptr<unsigned char>();
			memcpy(outputBuffer, outputfData1, NTSEG_OUT_VOLUME_SIZE * NTSEG_OUT_VOLUME_SIZE * NTSEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
		}
		std::chrono::steady_clock::time_point End_A = std::chrono::steady_clock::now();
		LOG_I("segResnet Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(End_A - begin_A).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in segResnetNT_Inside");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}


int NTAuto::Measure2DSagmentation(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, float* pcaVector, int width, int height, bool RGB) // inferenceInfo->pcaVector
{
	try {
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();
		if (mIsCPUOnly == 0)
		{
			input = input.to(mDeviceString);
		}

		// image normalization
		at::Tensor max_t = (torch::max)(input);
		at::Tensor min_t = (torch::min)(input);
		int minMaxDiff = static_cast<int>((max_t - min_t).item<float>());

		if((minMaxDiff) <= 0)
		{
			LOG_E("error -> (max_int - min_int) = 0");
			return RET_FAILURE;
		}

		input = (input - min_t) / (max_t - min_t);
		
		if (RGB == true)
		{
			input = input.reshape({ width, height, 3 });
			input = input.permute({ 2,0,1 }).unsqueeze(0);
			input = input.repeat({ 1, 1, 1, 1 });

		}
		else
		{
			input = input.reshape({ width, height, 1 });
			input = input.permute({ 2,0,1 }).unsqueeze(0);
			//input = input.repeat({ 1, 3, 1, 1 });
		}
		
		if (input.size(2) > 256) {
			input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ NT2DSEG_IMAGE_SIZE,NT2DSEG_IMAGE_SIZE })).align_corners(true));
		}

		LOG_T("input size: {}, {}, {}, {}", input.size(0), input.size(1), input.size(2), input.size(3));
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(input);
		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("NTMeasure2DSagmentation Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor().to(at::kCPU); //output index : [batch, channel, Height, Width]
		c10::cuda::CUDACachingAllocator::emptyCache();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("NTMeasure2DSagmentation Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point beginC = std::chrono::steady_clock::now();
		
		at::Tensor NT_wide = output.index({ 0, 0, Slice(),Slice() }).clone();
		NT_wide = torch::sigmoid(NT_wide);
		NT_wide = torch::where(NT_wide >= 0.5, 1, 0);
		
		at::Tensor NT = output.index({ 0, 1, Slice(), Slice() }).clone();
		NT = torch::sigmoid(NT);
		NT = torch::where(NT >= 0.5, 1, 0);
		
		at::Tensor Target_mask = torch::cat({ NT, NT_wide }, 0).flatten();
		
		at::Tensor Yvec = output.index({ 0, 2, Slice(), Slice() }).clone();
		at::Tensor Xvec = output.index({ 0, 3, Slice(), Slice() }).clone();
		
		output = torch::mul(Target_mask, 255);
		output = output.to(torch::kUInt8);

		at::Tensor nonzero_indices = torch::nonzero(NT);
		if (nonzero_indices.size(0) == 0)
		{
			mCommonProcess->printTensorInfo(nonzero_indices);
			return RET_FAILURE;
		}
		at::Tensor mean_indices = torch::mean(nonzero_indices.to(at::kFloat), 0);

		int x = static_cast<int>(mean_indices[1].item<float>()); 
		int y = static_cast<int>(mean_indices[0].item<float>());
		float YvecF = Yvec[y][x].item<float>();
		float XvecF = Xvec[y][x].item<float>();
		pcaVector[0] = XvecF;
		pcaVector[1] = YvecF;
		
		std::chrono::steady_clock::time_point endC = std::chrono::steady_clock::now();
		LOG_I("NTMeasure2DSagmentation Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(endC - beginC).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in NTMeasure2DSagmentation");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;

}

int NTAuto::cc3d(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>& labelInfo, UINT32 mappedLabel, bool returnCentroid) {
	typedef struct { int x; int y; int z; } Point;
	try
	{
		int sizeX, sizeY, sizeZ;
		if (inTensor.dim() == 5) {
			sizeX = inTensor.size(2);
			sizeY = inTensor.size(3);
			sizeZ = inTensor.size(4);
		}
		else if (inTensor.dim() == 3) {
			sizeX = inTensor.size(0);
			sizeY = inTensor.size(1);
			sizeZ = inTensor.size(2);
		}
		else {
			sizeX = sizeY = sizeZ = NTSEG_OUT_VOLUME_SIZE;// VOLUME_SIZE;
		}

		int sizeXY = sizeX * sizeY;
		int sizeXYZ = sizeXY * sizeY;

		// define neighbors point, total 26
		int Neighbors[CCL_CONNECTIVITY_26] = {
			// -z
			-sizeXY - sizeX - 1, -sizeXY - sizeX, -sizeXY - sizeX + 1,
			-sizeXY - 1,         -sizeXY, -sizeXY + 1,
			-sizeXY + sizeX - 1, -sizeXY + sizeX, -sizeXY + sizeX + 1,
			// z
			-sizeX - 1,          -sizeX,          -sizeX + 1,
							 -1, /* [Current Position] */           1,
			 sizeX - 1,           sizeX,           sizeX + 1,
			 // +z
			 sizeXY - sizeX - 1,  sizeXY - sizeX,  sizeXY - sizeX + 1,
			 sizeXY - 1,          sizeXY,  sizeXY + 1,
			 sizeXY + sizeX - 1,  sizeXY + sizeX,  sizeXY + sizeX + 1,
		};
		// ordering: x,y,z
		int CoordMap[CCL_CONNECTIVITY_26][3] = {
			//  x-1         x         x+1
			// z-1
			{-1,-1,-1}, {0,-1,-1}, {1,-1,-1}, // y-1
			{-1, 0,-1}, {0, 0,-1}, {1, 0,-1}, // y
			{-1, 1,-1}, {0, 1,-1}, {1, 1,-1}, // y+1
			// z
			{-1,-1, 0}, {0,-1, 0}, {1,-1, 0},
			{-1, 0, 0},            {1, 0, 0},
			{-1, 1, 0}, {0, 1, 0}, {1, 1, 0},
			// z+1
			{-1,-1, 1}, {0,-1, 1}, {1,-1, 1},
			{-1, 0, 1}, {0, 0, 1}, {1, 0, 1},
			{-1, 1, 1}, {0, 1, 1}, {1, 1, 1}
		};

		if(!inTensor.is_contiguous())
		{
			inTensor = inTensor.contiguous();
		}
		UINT8* inBuffer = inTensor.flatten().data_ptr<UINT8>();

		if (!outTensor.is_contiguous())
		{
			outTensor = outTensor.contiguous();
		}
		UINT8* outBuffer = outTensor.flatten().data_ptr<UINT8>();

		std::queue<Point> queue;
		Point p;
		int newLabel = mappedLabel;
		int OneHotLabel = 1;
		int compCount = 0;
		int curPos = 0;
		int voxCnt = 0;
		int maxVoxel = 0;
		at::Tensor CompCoordTensor;

		if (returnCentroid) {
			//maxVoxel = std::pow((NTSEG_OUT_VOLUME_SIZE / 128.), 3.) * MAX_VOLUME_SIZE; // TODO: Find maximum number for coordinate.  --> 8000(256size)의 경우 오류난 적이 있습니다.(8311) 늘려도 좋을 것 같습니다.
			maxVoxel = std::pow(NTSEG_OUT_VOLUME_SIZE, 3.);
			CompCoordTensor = torch::zeros({ maxVoxel, 3 }); // It stores the coordinates of connected components.
		}

		for (int z = 1; z < sizeZ - 1 && compCount < MAX_COMPONENTS; z++) {
			for (int y = 1; y < sizeY - 1 && compCount < MAX_COMPONENTS; y++) {
				for (int x = 1; x < sizeX - 1 && compCount < MAX_COMPONENTS; x++) {
					curPos = (sizeXY * z) + (sizeX * y) + x;

					if ((inBuffer[curPos] == OneHotLabel) && (outBuffer[curPos] == 0)) { // if unchecked
						// insert point into queue
						queue.push({ x,y,z });

						// loop: while the queue is empty
						while (!queue.empty()) {
							// pop point from queue
							p = queue.front();
							queue.pop();

							// check boundary case
							if (p.x < 0 || p.x > sizeX ||
								p.y < 0 || p.y > sizeY ||
								p.z < 0 || p.z > sizeZ) {
								continue;
							}
							// update output to current label
							curPos = (sizeXY * p.z) + (sizeX * p.y) + p.x;
							outBuffer[curPos] = newLabel;
							if (returnCentroid) {
								CompCoordTensor.index_put_({ voxCnt, Slice() }, torch::tensor({ p.x, p.y, p.z }));
							}
							++voxCnt;

							if (voxCnt >= maxVoxel) {
								LOG_E("Voxel count exceeded the max number of voxel size {}", maxVoxel);
								return RET_FAILURE;
							}

							// loop: visit all neighbors
							for (int i = 0; i < CCL_CONNECTIVITY_26; i++) {
								if (curPos + Neighbors[i] < 0 || curPos + Neighbors[i] > sizeXYZ) {
									continue;
								}
								// check unchecked coordinate
								if ((inBuffer[curPos + Neighbors[i]] == OneHotLabel) && (outBuffer[curPos + Neighbors[i]] == 0)) {
									// update label to output
									outBuffer[curPos + Neighbors[i]] = newLabel;

									// push current position into queue
									queue.push({ p.x + CoordMap[i][0], p.y + CoordMap[i][1], p.z + CoordMap[i][2] });
								}
							}
						}

						// update voxel count and centroid
						ComponentInfo comp;
						comp.volume = voxCnt;

						if (returnCentroid) {
							auto c = CompCoordTensor.index({ Slice(None, voxCnt), Slice() }).mean(0);
							c = c.to(torch::kInt32);
							auto cent = c.data_ptr<INT32>();
							comp.centroid[0] = (UINT32)(cent[0]);
							comp.centroid[1] = (UINT32)(cent[1]);
							comp.centroid[2] = (UINT32)(cent[2]);
							CompCoordTensor.index({ Slice(None, voxCnt), Slice() }).zero_();
						}
						comp.ccLabel = newLabel;
						labelInfo.push_back(comp);

						++compCount;
						++newLabel;
						voxCnt = 0;
					}
				} // for sizeX
			} // for sizeY
		} // for sizeZ
	}
	
	catch (const c10::Error& e) {
		LOG_E("error in cc3d");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}

	return RET_OK;
}

int NTAuto::removeComponent(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, int targetLabel) {
	try {
		at::Tensor outlierCoord = (inTensor == targetLabel);
		outTensor.index_put_({ 0, labelIndex, { outlierCoord } }, 0);
	}
	catch (const c10::Error& e) {
		LOG_E("error in removeComponent");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int NTAuto::removeExcessComponents(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, int startIndex, int endIndex) {
	try {
		vector<int> removedIndex;
		for (int ci = startIndex; ci < endIndex; ci++) {
			//ci = VolumeSize[labelIndex][vi].idx;
			//LOG_D("{} label's {} th component removed. VolumeSize: {}, limit component: {}", labelName_CNS[labelIndex], ci + 1, labelInfo[ci].volume, startIndex);
			removeComponent(outTensor, inTensor, labelIndex, labelInfo[ci].ccLabel);
			removedIndex.push_back(ci);
		}

		// update labelInfo
		sort(removedIndex.begin(), removedIndex.end(), greater<UINT32>()); // to remove element from end side
		for (auto i : removedIndex) {
			labelInfo.erase(labelInfo.begin() + i);
		}

		// Another way, but untested.
		//labelInfo.erase(remove_if(labelInfo.begin(), labelInfo.end(), [threshold](ComponentInfo c) {
		//	return c.volume < threshold; }), labelInfo.end());
	}
	catch (const c10::Error& e) {
		LOG_E("error in removeExcessComponents");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int NTAuto::HeadMaximumRadius(at::Tensor inputData, float* MaximumRadius) {
	try
	{
		at::Tensor inputMean;
		if (inputData.size(0) > 0)
		{
			inputData = inputData.to(torch::kFloat);
			inputMean = inputData.mean(0);
			//LOG_T("inputData size: {}, {}", inputData.size(0), inputData.size(1));
			//LOG_T("inputMean size: {}", inputMean.size(0));
			at::Tensor inputMax = (inputData - inputMean).pow(2).sum(1).sqrt().amax();
			*MaximumRadius = inputMax.flatten().item<float>();
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in HeadMaximumRadius");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}
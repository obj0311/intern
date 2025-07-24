#include "CNSAuto.h"
#include "US3DLog.h"

enum ErrorCode {
	ERR_SUCCESS = 300,
	ERR_UNKNOWN,
	ERR_INVALID_ARGUMENT,
	ERR_LABEL_NOT_EXIST,
	ERR_COMPONENT_EXCEEDED,
	ERR_OUT_OF_RANGE_DISTANCE,
	ERR_OUT_OF_RANGE_DISTANCE_FROM_MID_PLANE,
	ERR_NOT_SYMMETRY,
	ERR_TOO_SMALL
};

int setPlane(IN at::Tensor& vec1, IN at::Tensor& vec2, IN at::Tensor& point, OUT at::Tensor& n, OUT at::Tensor& d);

CNSAuto::CNSAuto(int isCPUOnly, std::string deviceString, CommonProcess* commonProcess)
	:mCommonProcess(commonProcess)
{
	mIsCPUOnly = isCPUOnly;
	mDeviceString = deviceString;
}

CNSAuto::~CNSAuto()
{
}

int CNSAuto::ExcuteCNSAuto(torch::jit::script::Module module, at::Tensor& inputTensor, InferenceInfo* inferenceInfo, unsigned char* outputBuffer)
{
	int resultValue = true;

	resultValue = segResnet(module, inputTensor, inputTensor, outputBuffer, inferenceInfo->SegMean, MARTIAN_CNS_SEG, inferenceInfo->labelNum, inferenceInfo->thresholdTable, inferenceInfo->processOnTable, labelPriorityTable_CNS, labelOffsetTable_CNS, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize, inferenceInfo->gaussianFilterMode, inferenceInfo->morphologyMode, &inferenceInfo->errorCode, &inferenceInfo->returnValue);

	inferenceInfo->xSize = CNSSEG_OUT_VOLUME_SIZE;
	inferenceInfo->ySize = CNSSEG_OUT_VOLUME_SIZE;
	inferenceInfo->zSize = CNSSEG_OUT_VOLUME_SIZE;

	if (resultValue == RET_CHECK_ERROR_CODE)
	{
		return resultValue;
	}

	inputTensor = inputTensor.to(torch::kFloat32);

	if (inferenceInfo->SegMean[18] > 0)
	{
		float pcaMeans[9] = { 0, };

		torchPCAMeans(inputTensor, CNS_NUM_CSP, 1, pcaMeans, CNSSEG_OUT_VOLUME_SIZE, CNSSEG_OUT_VOLUME_SIZE, CNSSEG_OUT_VOLUME_SIZE);

		inferenceInfo->SegMean[24] = pcaMeans[0];
		inferenceInfo->SegMean[25] = pcaMeans[1];
		inferenceInfo->SegMean[26] = pcaMeans[2];
		inferenceInfo->SegMean[27] = pcaMeans[3];
		inferenceInfo->SegMean[28] = pcaMeans[4];
		inferenceInfo->SegMean[29] = pcaMeans[5];
		inferenceInfo->SegMean[30] = pcaMeans[6];
		inferenceInfo->SegMean[31] = pcaMeans[7];
		inferenceInfo->SegMean[32] = pcaMeans[8];
	}

	if (inferenceInfo->SegMean[21] > 0)
	{
		float axisRadius[3] = { 0, };
		torchPCAProjectionAnalysis(inputTensor, CNS_NUM_CB, axisRadius, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize);

		inferenceInfo->SegMean[33] = axisRadius[0];
		inferenceInfo->SegMean[34] = axisRadius[1];
		inferenceInfo->SegMean[35] = axisRadius[2];

		at::Tensor eVec;
		at::Tensor sampleMean;
		at::Tensor eValue;
		resultValue = mCommonProcess->torchPCA(inputTensor, CNS_NUM_MID, eVec, sampleMean, eValue, inferenceInfo->xSize, inferenceInfo->ySize, inferenceInfo->zSize);

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

		// midline -> CSP Vector
		float MIDtoCSP[3];
		MIDtoCSP[0] = inferenceInfo->SegMean[18] - inferenceInfo->SegMean[21];
		MIDtoCSP[1] = inferenceInfo->SegMean[19] - inferenceInfo->SegMean[22];
		MIDtoCSP[2] = inferenceInfo->SegMean[20] - inferenceInfo->SegMean[23];
		float ThtoCSP[3];
		ThtoCSP[0] = inferenceInfo->SegMean[18] - inferenceInfo->SegMean[3];
		ThtoCSP[1] = inferenceInfo->SegMean[19] - inferenceInfo->SegMean[4];
		ThtoCSP[2] = inferenceInfo->SegMean[20] - inferenceInfo->SegMean[5];

		float vecLength1 = sqrt(MIDtoCSP[0] * MIDtoCSP[0] + MIDtoCSP[1] * MIDtoCSP[1] + MIDtoCSP[2] * MIDtoCSP[2]);
		MIDtoCSP[0] = MIDtoCSP[0] / vecLength1;
		MIDtoCSP[1] = MIDtoCSP[1] / vecLength1;
		MIDtoCSP[2] = MIDtoCSP[2] / vecLength1;
		float vecLength2 = sqrt(ThtoCSP[0] * ThtoCSP[0] + ThtoCSP[1] * ThtoCSP[1] + ThtoCSP[2] * ThtoCSP[2]);
		ThtoCSP[0] = ThtoCSP[0] / vecLength2;
		ThtoCSP[1] = ThtoCSP[1] / vecLength2;
		ThtoCSP[2] = ThtoCSP[2] / vecLength2;

		// PCA로 구한 Vector 1, 2 와 midline to csp vector와 Dot product
		float vec1dot = inferenceInfo->pcaVector[0] * MIDtoCSP[0] + inferenceInfo->pcaVector[1] * MIDtoCSP[1] + inferenceInfo->pcaVector[2] * MIDtoCSP[2];
		float vec2dot = inferenceInfo->pcaVector[3] * ThtoCSP[0] + inferenceInfo->pcaVector[4] * ThtoCSP[1] + inferenceInfo->pcaVector[5] * ThtoCSP[2];

		// 두 Vector가 다른 방향을 향하면 반전
		if (vec1dot < 0)
		{
			inferenceInfo->pcaVector[0] = inferenceInfo->pcaVector[0] * -1;
			inferenceInfo->pcaVector[1] = inferenceInfo->pcaVector[1] * -1;
			inferenceInfo->pcaVector[2] = inferenceInfo->pcaVector[2] * -1;
		}
		// 두 Vector가 다른 방향을 향하면 반전
		if (vec2dot < 0)
		{
			inferenceInfo->pcaVector[3] = inferenceInfo->pcaVector[3] * -1;
			inferenceInfo->pcaVector[4] = inferenceInfo->pcaVector[4] * -1;
			inferenceInfo->pcaVector[5] = inferenceInfo->pcaVector[5] * -1;
		}

		// 방향 보정한 Vector Cross Product 하여 vector3 산출
		inferenceInfo->pcaVector[6] = inferenceInfo->pcaVector[1] * inferenceInfo->pcaVector[5] - inferenceInfo->pcaVector[2] * inferenceInfo->pcaVector[4];
		inferenceInfo->pcaVector[7] = inferenceInfo->pcaVector[2] * inferenceInfo->pcaVector[3] - inferenceInfo->pcaVector[0] * inferenceInfo->pcaVector[5];
		inferenceInfo->pcaVector[8] = inferenceInfo->pcaVector[0] * inferenceInfo->pcaVector[4] - inferenceInfo->pcaVector[1] * inferenceInfo->pcaVector[3];

		// improve Th center
		// define mid plane
		auto mid = torch::from_blob(&inferenceInfo->SegMean[21], { 3 });
		auto vec1 = torch::from_blob(&inferenceInfo->pcaVector[0], { 3 });
		auto vec2 = torch::from_blob(&inferenceInfo->pcaVector[3], { 3 });
		setPlane(vec1, vec2, mid, mid_plane_n, mid_plane_d);

		// project Th center to Mid Plane
		auto th = torch::from_blob(&inferenceInfo->SegMean[3], { 3 });
		at::Tensor v = torch::sub(th, mid); // MID -> Th
		auto k = torch::dot(v, mid_plane_n); // distance
		auto th_proj = torch::sub(th, k * mid_plane_n);
		float* th_proj_temp = th_proj.flatten().data_ptr<float>(); // projected point

		inferenceInfo->SegMean[3] = th_proj_temp[0];
		inferenceInfo->SegMean[4] = th_proj_temp[1];
		inferenceInfo->SegMean[5] = th_proj_temp[2];

		// Projection points of the CP and CB to vec2
		LOG_T("CP size: {}, CB size: {}", cnsLabelsInfo[CNS_IDX_CP].size(), cnsLabelsInfo[CNS_IDX_CB].size());
		if (cnsLabelsInfo[CNS_IDX_CP].size() > 0 && cnsLabelsInfo[CNS_IDX_CB].size() > 0)
		{
			// CP
			at::Tensor coordCP = (CcTensor == (int)cnsLabelsInfo[CNS_IDX_CP][0].ccLabel).nonzero().flip(1);
			// dot product of PCA's vec2 and coordinate of volume
			at::Tensor mulCP = torch::mul(vec2, coordCP);
			at::Tensor projCP = mulCP.sum(1);
			// max/min index of projected points
			at::Tensor idxProjMaxCP = projCP.argmax(0);
			at::Tensor idxProjMinCP = projCP.argmin(0);
			// max/min value of projected points
			float valProjMaxCP = projCP.index({ idxProjMaxCP }).item<float>();
			float valProjMinCP = projCP.index({ idxProjMinCP }).item<float>();
			//PLOGD << "valProjMaxCP: " << valProjMaxCP << ", valProjMinCP: " << valProjMinCP;

			// CB
			at::Tensor coordCB = (CcTensor == (int)cnsLabelsInfo[CNS_IDX_CB][0].ccLabel).nonzero().flip(1);
			at::Tensor mulCB = torch::mul(vec2, coordCB);
			at::Tensor projCB = mulCB.sum(1);
			at::Tensor idxProjMaxCB = projCB.argmax(0);
			at::Tensor idxProjMinCB = projCB.argmin(0);
			float valProjMaxCB = projCB.index({ idxProjMaxCB }).item<float>();
			float valProjMinCB = projCB.index({ idxProjMinCB }).item<float>();
			//PLOGD << "valProjMaxCB: " << valProjMaxCB << ", valProjMinCB: " << valProjMinCB;

			// Coordinate of the CP up and CB bottom
			at::Tensor cpUp;
			at::Tensor cbBottom;
			if (valProjMaxCP >= valProjMaxCB) {
				cpUp = coordCP.index({ idxProjMaxCP, Slice() });
				cbBottom = coordCB.index({ idxProjMinCB, Slice() });
			}
			else {
				cpUp = coordCP.index({ idxProjMinCP, Slice() });
				cbBottom = coordCB.index({ idxProjMaxCB, Slice() });
			}

			inferenceInfo->SegMean[36] = (float)cpUp[0].item<int64_t>(); // CP Up
			inferenceInfo->SegMean[37] = (float)cpUp[1].item<int64_t>();
			inferenceInfo->SegMean[38] = (float)cpUp[2].item<int64_t>();
			inferenceInfo->SegMean[39] = (float)cbBottom[0].item<int64_t>(); // CB Bottom
			inferenceInfo->SegMean[40] = (float)cbBottom[1].item<int64_t>();
			inferenceInfo->SegMean[41] = (float)cbBottom[2].item<int64_t>();
			//PLOGD << "CP MAX: " << coordCP.index({ idxProjMaxCP, Slice() });
		}
	}
	else
	{
		inferenceInfo->pcaVector[0] = 0;
		inferenceInfo->pcaVector[1] = 0;
		inferenceInfo->pcaVector[2] = 0;
		inferenceInfo->pcaVector[3] = 0;
		inferenceInfo->pcaVector[4] = 0;
		inferenceInfo->pcaVector[5] = 0;
		inferenceInfo->pcaVector[6] = 0;
		inferenceInfo->pcaVector[7] = 0;
		inferenceInfo->pcaVector[8] = 0;
	}

	LOG_I("Th Mean = {}, {}, {}", inferenceInfo->SegMean[3], inferenceInfo->SegMean[4], inferenceInfo->SegMean[5]);
	LOG_I("CB Mean = {}, {}, {}", inferenceInfo->SegMean[6], inferenceInfo->SegMean[7], inferenceInfo->SegMean[8]);
	LOG_I("CM Mean = {}, {}, {}", inferenceInfo->SegMean[9], inferenceInfo->SegMean[10], inferenceInfo->SegMean[11]);
	LOG_I("CP Mean = {}, {}, {}", inferenceInfo->SegMean[12], inferenceInfo->SegMean[13], inferenceInfo->SegMean[14]);
	LOG_I("PVC Mean = {}, {}, {}", inferenceInfo->SegMean[15], inferenceInfo->SegMean[16], inferenceInfo->SegMean[17]);
	LOG_I("CSP Mean = {}, {}, {}", inferenceInfo->SegMean[18], inferenceInfo->SegMean[19], inferenceInfo->SegMean[20]);
	LOG_I("Midline Mean = {}, {}, {}", inferenceInfo->SegMean[21], inferenceInfo->SegMean[22], inferenceInfo->SegMean[23]);
	LOG_I("eVec1 = {}, {}, {}", inferenceInfo->pcaVector[0], inferenceInfo->pcaVector[1], inferenceInfo->pcaVector[2]);
	LOG_I("eVec2 = {}, {}, {}", inferenceInfo->pcaVector[3], inferenceInfo->pcaVector[4], inferenceInfo->pcaVector[5]);
	LOG_I("eVec3 = {}, {}, {}", inferenceInfo->pcaVector[6], inferenceInfo->pcaVector[7], inferenceInfo->pcaVector[8]);
	LOG_I("CSPK1 = {}, {}, {}", inferenceInfo->SegMean[24], inferenceInfo->SegMean[25], inferenceInfo->SegMean[26]);
	LOG_I("CSPK2 = {}, {}, {}", inferenceInfo->SegMean[27], inferenceInfo->SegMean[28], inferenceInfo->SegMean[29]);
	LOG_I("CSPK3 = {}, {}, {}", inferenceInfo->SegMean[30], inferenceInfo->SegMean[31], inferenceInfo->SegMean[32]);
	LOG_I("CBDistrib = {}, {}, {}", inferenceInfo->SegMean[33], inferenceInfo->SegMean[34], inferenceInfo->SegMean[35]);
	LOG_I("CP Up = {}, {}, {}", inferenceInfo->SegMean[36], inferenceInfo->SegMean[37], inferenceInfo->SegMean[38]);
	LOG_I("CB Bottom = {}, {}, {}", inferenceInfo->SegMean[39], inferenceInfo->SegMean[40], inferenceInfo->SegMean[41]);

	// Store CNS PVolume
	// [
	//inferenceInfo->zSize = CNSSEG_OUT_VOLUME_SIZE * 2;
	// ]
	return resultValue;
}

// Input - Rank 1(size = volume size) Tensor, Output - {1,1,a,b,c} Tensor
int CNSAuto::segResnet(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, unsigned char* outputBuffer, float* SegMeanOut, int mode, int labelNum, float* labelThreshold, int* processOnLabel, int* labelPriorityTable, int* labelOffsetTable, int volumeDimA, int volumeDimB, int volumeDimC, int gauissianFilterMode, int morphProcessMode, int* errorCode, int* returnValue)
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

			output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(std::vector<int64_t>({ CNSSEG_OUT_VOLUME_SIZE, CNSSEG_OUT_VOLUME_SIZE, CNSSEG_OUT_VOLUME_SIZE })));		//Tempcode - severance CNS Test
			maxPVolume = output.clone();

			c10::cuda::CUDACachingAllocator::emptyCache();

			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				float maxProbability = output.index({ 0,labelIndex,Slice(),Slice(),Slice() }).amax().flatten()[0].item<float>();
				maxProbability = floor(maxProbability * 100) / 100;
				float threshold = (labelIndex != CNS_IDX_MID) ? maxProbability : labelThreshold[labelIndex];
				//PLOGD << labelName_CNS[labelIndex] << ">> max probability: " << maxProbability << ", threshold: " << threshold;

				maxPVolume.index_put_({ 0,labelIndex,Slice(),Slice(),Slice() }, torch::threshold(maxPVolume.index({ 0,labelIndex,Slice(),Slice(),Slice() }), threshold, 0));
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
				mCommonProcess->dilation3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, labelPriorityTable_CNS, GPUAcc);
			}
			if (morphProcessMode == 2 || morphProcessMode == 3)
			{
				mCommonProcess->erosion3D(output, output, labelNum, tempSize, tempSize, tempSize, processOnLabel, GPUAcc);
			}

			if (mode == MARTIAN_CNS_SEG) {
				thK2.clear();
				cbK2.clear();
				hasCNSPlanes = false;
				for (vector<ComponentInfo> ci : cnsLabelsInfo) {
					ci.clear();
				}
				CcTensor = torch::zeros({ tempSize, tempSize, tempSize }).to(torch::kUInt8);
			}

			// Labeling
			int processIndex = 0;
			for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
			{
				*errorCode = ERR_SUCCESS;
				*returnValue = 0;

				processIndex = processOrder_CNS[labelIndex];

				//std::chrono::steady_clock::time_point begin_cc3d = std::chrono::steady_clock::now();

				if (processIndex >= CNS_IDX_TH) { // from Th
					vector<ComponentInfo> labelInfo;
					output = output.to(at::kCPU);

					at::Tensor InTensor = output.index({ 0, processIndex, Slice(), Slice() ,Slice() }).to(torch::kUInt8);

					// step 1: cc3d
					// Connected Component Labeling
					// index: 0, 1, 2, 3, 4, 5,  6,  7
					// label: 1, 2, 3, 4, 5, 6,  7,  8 (=index+1)
					// object:BG,Th,CB,CM,CP,PVC,CSP,MID
					bool returnCentroid = true;
					cc3d(CcTensor, InTensor, labelInfo, ccLabelMap_CNS[processIndex], returnCentroid);

					// Log for connected component labeling result
					LOG_T("{} label's components: {}", labelName_CNS[processIndex], labelInfo.size());
					std::ostringstream stream;
					for (int i = 0; i < labelInfo.size(); i++) {
						stream.str(std::string());
						stream << "\t" << i + 1 << "th: " << labelInfo[i].volume << " voxel";
						if (returnCentroid) {
							stream << ", [" << labelInfo[i].centroid[0] << ", " << labelInfo[i].centroid[1] << ", " << labelInfo[i].centroid[2] << "]";
						}
						LOG_T("{}", stream.str());
					}

					//std::chrono::steady_clock::time_point end_cc3d = std::chrono::steady_clock::now();

					// step 2: Remove outlier
					//removeOutlier(output, CcTensor, processIndex, labelInfo, RuleVolumeSize_CNS[processIndex][0]);

					// step 3: Check rule base

					// 3-1 check component exist
					if (labelInfo.size() < 1 && RuleLabelExist_CNS[processIndex]) {
						*errorCode = ERR_LABEL_NOT_EXIST;
						*returnValue = labelMap_CNS[processIndex];
						LOG_E("{} label is not exist", labelName_CNS[processIndex]);
						return RET_CHECK_ERROR_CODE;
					}

					// Sort descending order of voxel count array
					std::sort(labelInfo.begin(), labelInfo.end(), [](ComponentInfo a, ComponentInfo b) { return a.volume > b.volume; });

					// 3-2 check available component count
					// Removes a component if the number of components exceeds the maximum according to the rule.
					if (labelInfo.size() > RuleMaxComponentCount_CNS[processIndex]) { // If the maximum number of components is exceeded
						removeExcessComponents(output, CcTensor, processIndex, labelInfo, RuleMaxComponentCount_CNS[processIndex], labelInfo.size());

						// Since this is not an error case, there is no need to return an error code.
						//*errorCode = ERR_COMPONENT_EXCEEDED;// many components than maximum number by the rule
						//*returnValue = labelMap_CNS[processIndex];
					}

					if (processIndex == CNS_IDX_CSP && labelInfo.size() >= 1 && RuleMinVolumeSize[processIndex] > labelInfo[0].volume) {
						*errorCode = ERR_TOO_SMALL;
						*returnValue = labelMap_CNS[processIndex];
						LOG_E("{} label is too small", labelName_CNS[processIndex]);
						return RET_CHECK_ERROR_CODE;
					}

					if (processIndex == CNS_IDX_TH && labelInfo.size() == 2 && returnCentroid) {
						thK2.push_back({ labelInfo[0].centroid[0], labelInfo[0].centroid[1], labelInfo[0].centroid[2] });
						thK2.push_back({ labelInfo[1].centroid[0], labelInfo[1].centroid[1], labelInfo[1].centroid[2] });
					}

					cnsLabelsInfo[processIndex] = labelInfo;

					// 3-3 Check in range centroid distance
					if (returnCentroid && checkInRange(processIndex, cnsLabelsInfo, errorCode, returnValue) != RET_OK) {
						LOG_E("{} is out of range", labelName_CNS[processIndex]);
						return RET_CHECK_ERROR_CODE;
					}

					if (processIndex == CNS_IDX_MID
						&& thK2.size() == 2
						&& cnsLabelsInfo[CNS_IDX_CSP].size() == 1)
					{
						// At this point we know the centroids of thK2[0], thK2[1] and csp.
						// So we can check and remove MidLine components that are far from MidPlane.
						if (setPlanes(cnsLabelsInfo, thK2) == RET_OK) {
							hasCNSPlanes = true;
							removeComponentFarFromMidPlane(output, CcTensor, cnsLabelsInfo, processIndex, RuleDistanceFromMidPlane_CNS[processIndex]);
						}
					}

					/*
					if (processIndex == CNS_IDX_MID
						&& cbK2.size() == 2
						&& hasCNSPlanes)
					{
						// At this point we know the centroids of cbK2[0], cbK2[1] and planes.
						// So we can check CB component that is symmetry by MidPlane.
						auto cb1 = torch::from_blob(cbK2[0].data(), { 3 });
						auto cb2 = torch::from_blob(cbK2[1].data(), { 3 });
						if (checkSymmetry(plane_n[1], plane_d[1], cb1, cb2, CNS_IDX_CB, errorCode, returnValue) != RET_OK) {
							return RET_CHECK_ERROR_CODE;
						}
					}
					*/

					if (mIsCPUOnly == 0)
					{
						output = output.to(mDeviceString);
						maxPVolume = maxPVolume.to(mDeviceString);
					}
					// output Tensor has changed with removal process.
					// Update maxPVolume Tensor based on the intersection of 1.0 values in output Tensor.
					maxPVolume = maxPVolume * output;
				}

				at::Tensor IndexData = output.index({ 0,processIndex,Slice(),Slice(),Slice() }).nonzero().to(torch::kFloat32);
				at::Tensor MaxPIndexData = maxPVolume.index({ 0,processIndex,Slice(),Slice(),Slice() }).nonzero().to(torch::kFloat32);

				if (processIndex == CNS_IDX_CSP) {
					if (IndexData.size(0) >= 3)
					{
						at::Tensor ClusterCenter;

						torchKMeans(IndexData, ClusterCenter, 3);
						ClusterCenter = ClusterCenter.flatten();
						SegMeanOut[24] = ClusterCenter[2].item<float>();
						SegMeanOut[25] = ClusterCenter[1].item<float>();
						SegMeanOut[26] = ClusterCenter[0].item<float>();
						SegMeanOut[27] = ClusterCenter[5].item<float>();
						SegMeanOut[28] = ClusterCenter[4].item<float>();
						SegMeanOut[29] = ClusterCenter[3].item<float>();
						SegMeanOut[30] = ClusterCenter[8].item<float>();
						SegMeanOut[31] = ClusterCenter[7].item<float>();
						SegMeanOut[32] = ClusterCenter[6].item<float>();
					}
					else
					{
						SegMeanOut[24] = 0;
						SegMeanOut[25] = 0;
						SegMeanOut[26] = 0;
						SegMeanOut[27] = 0;
						SegMeanOut[28] = 0;
						SegMeanOut[29] = 0;
						SegMeanOut[30] = 0;
						SegMeanOut[31] = 0;
						SegMeanOut[32] = 0;
					}
				}

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
					if (processIndex != CNS_IDX_BG)
					{
						*errorCode = ERR_LABEL_NOT_EXIST;
						*returnValue = labelMap_CNS[processIndex];
						return RET_CHECK_ERROR_CODE;
					}
				}

				if (DataMean.size(0) == 3) {
					SegMeanOut[processIndex * 3 + 0] = DataMean[2].item<float>();
					SegMeanOut[processIndex * 3 + 1] = DataMean[1].item<float>();
					SegMeanOut[processIndex * 3 + 2] = DataMean[0].item<float>();
				}

				/*
				if (processIndex == CNS_IDX_CB) {
					at::Tensor K;
					int ret = RET_FAILURE;

					if (IndexData.size(0) >= 2) {
						ret = torchKMeans(IndexData, K, 2);
					}

					if (ret == RET_OK) {
						K = K.flatten();
						cbK2.push_back({ K[2].item<float>(),K[1].item<float>(),K[0].item<float>() });
						cbK2.push_back({ K[5].item<float>(),K[4].item<float>(),K[3].item<float>() });
						//PLOGD << "CB_K2[0]: " << cbK2[0];
						//PLOGD << "CB_K2[1]: " << cbK2[1];
					}
				}
				*/

				output.index_put_({ 0,processIndex,Slice(),Slice(),Slice() },
					torch::mul(output.index({ 0,processIndex,Slice(),Slice(),Slice() }), labelOffsetTable[processIndex]));   // Label Offset	
			}
			//Merge
			output = torch::sum(output, 1).to(at::kCPU);

			output = output.to(at::kCPU);
			PVolume = PVolume.to(at::kCPU);

			output = output.to(torch::kUInt8);
			PVolume = PVolume.to(torch::kUInt8);

			unsigned char* outputfData1 = output.flatten().data_ptr<unsigned char>();
			unsigned char* outputfData2 = PVolume.flatten().data_ptr<unsigned char>();

			memcpy(outputBuffer, outputfData1, CNSSEG_OUT_VOLUME_SIZE* CNSSEG_OUT_VOLUME_SIZE* CNSSEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
			memcpy(outputBuffer + CNSSEG_OUT_VOLUME_SIZE * CNSSEG_OUT_VOLUME_SIZE * CNSSEG_OUT_VOLUME_SIZE, outputfData2, CNSSEG_OUT_VOLUME_SIZE* CNSSEG_OUT_VOLUME_SIZE* CNSSEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
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

int CNSAuto::mmSeg(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, int channel)
{
	int imageSize = width * height;

	try
	{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0)
		{
			input = input.to(mDeviceString);
		}

		////libtorch 기반 Normalization
		at::Tensor nonz_t = input.nonzero();
		at::Tensor mean_t = input.index({ nonz_t }).mean();
		at::Tensor std_t = input.index({ nonz_t }).std();
		input.index_put_({ Slice() }, (input.index({ Slice() }) - mean_t) / std_t);

		if (channel == 3)
		{
			input = input.reshape({ width, height, 3 });
			input = input.permute({ 2,0,1 }).unsqueeze(0);
		}
		else
		{
			input = input.reshape({ width, height, 1 });
			input = input.permute({ 2,0,1 }).unsqueeze(0);
			input = input.repeat({ 1, 3, 1, 1 });
		}

		input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ IN_IMAGE_SIZE, IN_IMAGE_SIZE })).align_corners(true));		//Image Resize

		std::vector<torch::jit::IValue> inputs;
		at::Tensor input_reshape = input.reshape({ 1, 3, IN_IMAGE_SIZE, IN_IMAGE_SIZE });

		inputs.push_back(input_reshape);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("mmSeg Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor().to(at::kCPU);           // 처리한 Data를 CPU mem에 넣기 위해
		c10::cuda::CUDACachingAllocator::emptyCache();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("mmSeg Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point beginC = std::chrono::steady_clock::now();

		output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ width, height })).align_corners(true));		//Image Resize

		output.squeeze_();
		output = output.argmax(0).toType(torch::kByte);
		output = output.reshape({ imageSize });

		std::chrono::steady_clock::time_point endC = std::chrono::steady_clock::now();
		LOG_I("mmSeg Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(endC - beginC).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in mmSeg");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::torchPCAProjectionAnalysis(at::Tensor& input, int TargetLabel, float* axisRadius, int volumeDimA, int volumeDimB, int volumeDimC)
{
	try
	{
		at::Tensor eVecPj;
		at::Tensor sampleMeanPj;
		at::Tensor eValuePj;

		at::Tensor input_reshape = input.reshape({ volumeDimA, volumeDimB, volumeDimC });
		at::Tensor svdData = (input_reshape == TargetLabel).nonzero();

		if (svdData.size(0) > 0)
		{
			svdData = svdData.to(torch::kFloat);
			sampleMeanPj = svdData.mean(0);
			svdData = torch::sub(svdData, sampleMeanPj);
			auto outputs = torch::svd(svdData);
			eValuePj = std::get<1>(outputs);
			eVecPj = std::get<2>(outputs).transpose(0, 1);
			at::Tensor PComponents = torch::mm(eVecPj, svdData.t());
			at::Tensor distribMax = PComponents.amax(1);
			at::Tensor distribMin = PComponents.amin(1);

			float* distribMaxData = distribMax.flatten().data_ptr<float>();
			float* distribMinData = distribMin.flatten().data_ptr<float>();

			for (int i = 0; i < 3; i++)
			{
				axisRadius[i] = (abs(distribMaxData[i]) + abs(distribMinData[i])) / 2;
			}
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in torchPCAProjectionAnalysis");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

// 정해진 PCA 축상의 3점을 획득함 
// axis - 1(1st), 2(2nd), 3(3rd)
int CNSAuto::torchPCAMeans(at::Tensor& input, int TargetLabel, int axis, float* means, int volumeDimA, int volumeDimB, int volumeDimC)
{
	try
	{
		at::Tensor eVecPj;
		at::Tensor sampleMeanPj;
		at::Tensor eValuePj;

		at::Tensor input_reshape = input.reshape({ volumeDimA, volumeDimB, volumeDimC });
		at::Tensor svdData = (input_reshape == TargetLabel).nonzero();

		if (svdData.size(0) > 0 && axis > 0 && axis < 4)
		{
			svdData = svdData.to(torch::kFloat);
			sampleMeanPj = svdData.mean(0);
			svdData = torch::sub(svdData, sampleMeanPj);
			auto outputs = torch::svd(svdData);
			eValuePj = std::get<1>(outputs);
			eVecPj = std::get<2>(outputs).transpose(0, 1);
			at::Tensor PComponents = torch::mm(eVecPj, svdData.t());
			at::Tensor distribMax = PComponents.amax(1);
			at::Tensor distribMin = PComponents.amin(1);

			float* distribMaxData = distribMax.flatten().data_ptr<float>();
			float* distribMinData = distribMin.flatten().data_ptr<float>();

			eVecPj = eVecPj.flatten();		// {3,3}의 경우 먼저 flatten 한번 해줘야 한다.
			float* outputEVec = eVecPj.flatten().data_ptr<float>();
			float* outputSampleMean = sampleMeanPj.flatten().data_ptr<float>();

			means[3] = outputSampleMean[2];
			means[4] = outputSampleMean[1];
			means[5] = outputSampleMean[0];

			means[0] = outputSampleMean[2] + outputEVec[(axis - 1) * 3 + 2] * (distribMaxData[axis - 1] / 2);
			means[1] = outputSampleMean[1] + outputEVec[(axis - 1) * 3 + 1] * (distribMaxData[axis - 1] / 2);
			means[2] = outputSampleMean[0] + outputEVec[(axis - 1) * 3] * (distribMaxData[axis - 1] / 2);

			means[6] = outputSampleMean[2] + outputEVec[(axis - 1) * 3 + 2] * (distribMinData[axis - 1] / 2);
			means[7] = outputSampleMean[1] + outputEVec[(axis - 1) * 3 + 1] * (distribMinData[axis - 1] / 2);
			means[8] = outputSampleMean[0] + outputEVec[(axis - 1) * 3] * (distribMinData[axis - 1] / 2);
		}
		else
		{
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in torchPCAMeans");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::torchKMeans(at::Tensor& input, at::Tensor& CC, int clusterNum)
{
	int nSample = input.size(0);
	int nDim = input.size(1);
	try
	{
		at::Tensor Center = input.index({ Slice(0,clusterNum), Slice() }).clone();
		at::Tensor x_i = input.view({ nSample, 1,3 });
		at::Tensor c_j = Center.view({ 1, clusterNum, 3 });       // Init Cluster
		for (int i = 0; i < 10; i++)
		{
			at::Tensor D_ij = ((x_i - c_j) * (x_i - c_j)).sum(-1);		// Cluster <-> Sample 거리
			at::Tensor cl = D_ij.argmin(1).to(torch::kLong).view(-1);	// Sample별 더 가까운 Cluster 선정 (argmin)
			Center.zero_();
			Center.scatter_add_(0, cl.index({ Slice(), None }).repeat({ 1,3 }), input);		//Cluster에 따라 따로 Add (X,Y,Z각각)
			at::Tensor Ncl = torch::bincount(cl, None, clusterNum).type_as(Center).view({ clusterNum, 1 });		//Cluster에 속한 sample 갯수
			Center = Center / Ncl;		// 중점 계산
			c_j = Center;
		}
		CC = Center;
	}
	catch (const c10::Error& e) {
		LOG_E("error in torchKMeans");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::cc3d(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>& labelInfo, UINT32 mappedLabel, bool returnCentroid) {
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
			sizeX = sizeY = sizeZ = CNSSEG_OUT_VOLUME_SIZE;// VOLUME_SIZE;
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

		UINT8* inBuffer = inTensor.flatten().data_ptr<UINT8>();
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
			maxVoxel = std::pow((CNSSEG_OUT_VOLUME_SIZE / 128.), 3.) * MAX_VOLUME_SIZE; // TODO: Find maximum number for coordinate.  --> 8000(256size)의 경우 오류난 적이 있습니다.(8311) 늘려도 좋을 것 같습니다.
			CompCoordTensor = torch::zeros({ maxVoxel, 3 }); // It stores the coordinates of connected components.
		}

		for (int z = 1; z < sizeZ - 1 && compCount < MAX_COMPONENTS; z++) {
			for (int y = 1; y < sizeY - 1 && compCount < MAX_COMPONENTS; y++) {
				for (int x = 1; x < sizeX - 1 && compCount < MAX_COMPONENTS; x++) {
					curPos = (sizeXY * z) + (sizeX * y) + x;

					if (inBuffer[curPos] == OneHotLabel && outBuffer[curPos] == 0) { // if unchecked
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
								if (inBuffer[curPos + Neighbors[i]] == OneHotLabel && outBuffer[curPos + Neighbors[i]] == 0) {
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

int CNSAuto::removeComponent(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, int targetLabel) {
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

int CNSAuto::removeOutlier(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, UINT32 threshold) {
	try {
		vector<int> removedIndex;

		for (int ci = 0; ci < labelInfo.size(); ci++) {
			if (labelInfo[ci].volume < threshold) {
				//LOG_D("{} label's {} th component removed. VolumeSize: {}, threshold: {}", labelName_CNS[labelIndex], ci + 1, labelInfo[ci].volume, threshold);
				removeComponent(outTensor, inTensor, labelIndex, ccLabelMap_CNS[labelIndex] + ci);
				removedIndex.push_back(ci);
			}
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
		LOG_E("error in removeOutlier");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::removeExcessComponents(at::Tensor& outTensor, at::Tensor& inTensor, int labelIndex, vector<ComponentInfo>& labelInfo, int startIndex, int endIndex) {
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

int CNSAuto::removeComponentFarFromMidPlane(at::Tensor& outTensor, at::Tensor& inTensor, vector<ComponentInfo>* labelsInfo, int targetLabelIndex, int limit) {
	try {
		/*
		float ThMid[3] = {};

		// Calculate middle point between Th1 and Th2
		ThMid[0] = (thK2[0][0] + thK2[1][0]) / 2;
		ThMid[1] = (thK2[0][1] + thK2[1][1]) / 2;
		ThMid[2] = (thK2[0][2] + thK2[1][2]) / 2;

		// To Tensor from blob data
		auto csp = torch::from_blob(labelsInfo[CNS_IDX_CSP][0].centroid, { 3 });
		auto th1 = torch::from_blob(thK2[0].data(), { 3 });
		auto th2 = torch::from_blob(thK2[1].data(), { 3 });
		auto thc = torch::from_blob(ThMid, { 3 });

		at::Tensor vec1 = torch::sub(th1, csp); // Th1 - CSP
		at::Tensor vec2 = torch::sub(th2, csp); // Th2 - CSP
		at::Tensor vec3 = torch::sub(thc, csp); // ThC - CSP

		// plane's equation: ax + by + cz + d = 0
		// where (a,b,c) is normal vector of the plane

		// P1: intersects points CSP, Th1 and Th2
		at::Tensor n1, d1;
		getPlane(vec1, vec2, csp, n1, d1);

		// P2: passing through the point ThC and perpendicular to the P1 plane
		at::Tensor n2, d2;
		getPlane(vec3, n1, csp, n2, d2);
		*/

		// Calculate distance between plane P2 and point
		//PLOGD << labelName_CNS[targetLabelIndex] << " Label's size: " << labelsInfo[targetLabelIndex].size();
		vector<float> distances;
		vector<int> removedIndex;
		for (int ci = 0; ci < labelsInfo[targetLabelIndex].size(); ci++) {
			auto center = torch::from_blob(labelsInfo[targetLabelIndex][ci].centroid, { 3 });
			auto distance = torch::add(torch::dot(center, plane_n[1]), plane_d[1]).abs();
			auto d = distance.data_ptr<float>()[0];
			//PLOGD << "distance["<< ci << "]: " << d;
			if (d > limit) {
				//PLOGD << labelName_CNS[targetLabelIndex] << " label's " << ci + 1 << "th component removed." << " distance: " << d << ", limit distance: " << limit;
				removeComponent(outTensor, inTensor, targetLabelIndex, labelsInfo[targetLabelIndex][ci].ccLabel);
				removedIndex.push_back(ci);
			}
		}

		// update labelInfo
		sort(removedIndex.begin(), removedIndex.end(), greater<UINT32>()); // to remove element from end side
		for (auto i : removedIndex) {
			labelsInfo[targetLabelIndex].erase(labelsInfo[targetLabelIndex].begin() + i);
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in removeComponentFarFromMidPlane");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}

	return RET_OK;
}

int CNSAuto::checkInRange(int labelIndex, vector<ComponentInfo>* labelsInfo, int* errorCode, int* returnValue) {
	try {
		int associatedLabelIndex = RuleDistance_CNS[labelIndex][0];
		if (associatedLabelIndex < 0) // skip because there is no distance rule for this label
			return RET_OK;

		if (labelsInfo[labelIndex].empty()) {
			//PLOGD << labelName_CNS[labelIndex] << "'s labelInfo is empty!";
			*errorCode = ERR_LABEL_NOT_EXIST;
			*returnValue = labelMap_CNS[labelIndex];
			return RET_FAILURE;
		}
		else if (labelsInfo[associatedLabelIndex].empty()) {
			//PLOGD << labelName_CNS[associatedLabelIndex] << "'s labelInfo is empty!";
			*errorCode = ERR_LABEL_NOT_EXIST;
			*returnValue = labelMap_CNS[labelIndex];
			return RET_FAILURE;
		}

		float dx, dy, dz, distance;
		dx = dy = dz = distance = 0.0f;

		switch (labelIndex) {
		case 2: // CB: Check distance of [CB - Th1]
		case 3: // CM: Check distance of [CM - CB]
		case 5: // PVC: Check distance of [PVC - CP]
		case 6: // CSP: Check distance of [CSP - Th1]
			dx = (float)(labelsInfo[labelIndex][0].centroid[0]) - (float)(labelsInfo[associatedLabelIndex][0].centroid[0]);
			dy = (float)(labelsInfo[labelIndex][0].centroid[1]) - (float)(labelsInfo[associatedLabelIndex][0].centroid[1]);
			dz = (float)(labelsInfo[labelIndex][0].centroid[2]) - (float)(labelsInfo[associatedLabelIndex][0].centroid[2]);
			break;
		default:
			*errorCode = ERR_INVALID_ARGUMENT;
			*returnValue = labelMap_CNS[labelIndex];
			return RET_FAILURE;
		}
		distance = sqrt(dx * dx + dy * dy + dz * dz);
		//PLOGD << "distance [" << labelName_CNS[labelIndex] << "] - [" << labelName_CNS[associatedLabelIndex] << "]: " << distance;
		if (distance < RuleDistance_CNS[labelIndex][1]
			|| distance > RuleDistance_CNS[labelIndex][2]) {
			LOG_E("Out of range! Check distance of [{}] - [{}] in {} ~ {}", labelName_CNS[labelIndex], labelName_CNS[associatedLabelIndex], RuleDistance_CNS[labelIndex][1], RuleDistance_CNS[labelIndex][2]);
			*errorCode = ERR_OUT_OF_RANGE_DISTANCE;
			*returnValue = labelMap_CNS[labelIndex];
			return RET_FAILURE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in checkInRange");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::checkDistanceFromMidPlane(IN vector<vector<ComponentInfo>> labelsInfo, IN int targetLabelIndex, IN int limit, int* errorCode, int* returnValue) {
	try {
		// Calculate distance between plane P2 and point
		//PLOGD << labelName_CNS[targetLabelIndex] << " Label's size: " << labelsInfo[targetLabelIndex].size();
		vector<float> distances;
		for (ComponentInfo& comp : labelsInfo[targetLabelIndex]) {
			auto center = torch::from_blob(comp.centroid, { 3 });
			auto distance = torch::add(torch::dot(center, plane_n[1]), plane_d[1]).abs();
			distances.push_back((float)(distance.data_ptr<float>()[0]));
		}

		for (float d : distances) {
			//PLOGD << "distance: " << d;
			if (d > limit) {
				*errorCode = ERR_OUT_OF_RANGE_DISTANCE_FROM_MID_PLANE;
				*returnValue = labelMap_CNS[targetLabelIndex];
				return RET_CHECK_ERROR_CODE;
			}
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in checkDistanceFromMidPlane");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}

	return RET_OK;
}

int CNSAuto::checkSymmetry(at::Tensor n, at::Tensor d, at::Tensor p1, at::Tensor p2, int targetLabelIndex, int* errorCode, int* returnValue) {
	try {
		float ratio = 1.0;
		auto a = torch::add(torch::dot(p1, n), d).data_ptr<float>()[0];
		auto b = torch::add(torch::dot(p2, n), d).data_ptr<float>()[0];

		//LOG_D("a : b = {} : {}",a, b);
		if ((a > 0 && b > 0) || (a < 0 && b < 0)) {
			LOG_E("{} label only exists on one side", labelName_CNS[targetLabelIndex]);
			*errorCode = ERR_NOT_SYMMETRY;
			*returnValue = labelMap_CNS[targetLabelIndex];
			return RET_CHECK_ERROR_CODE;
		}

		a = std::abs(a);
		b = std::abs(b);

		if (a >= b && a != 0) {
			ratio = b / a;
		}
		else if (a < b && b != 0) {
			ratio = a / b;
		}

		//LOG_D("symmetry ratio: {}", ratio);
		if (ratio < SymmetryThreshold_CNS) {
			LOG_E("{} label is not symmetry", labelName_CNS[targetLabelIndex]);
			*errorCode = ERR_NOT_SYMMETRY;
			*returnValue = labelMap_CNS[targetLabelIndex];
			return RET_CHECK_ERROR_CODE;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in checkSymmetry");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int CNSAuto::setPlanes(IN vector<ComponentInfo>* labelsInfo, IN vector<array<float, 3>> thK2) {
	try {
		float ThMid[3] = {};

		// Calculate middle point between Th1 and Th2
		ThMid[0] = (thK2[0][0] + thK2[1][0]) / 2;
		ThMid[1] = (thK2[0][1] + thK2[1][1]) / 2;
		ThMid[2] = (thK2[0][2] + thK2[1][2]) / 2;

		// To Tensor from blob data
		auto csp = torch::from_blob(labelsInfo[CNS_IDX_CSP][0].centroid, { 3 });
		auto th1 = torch::from_blob(thK2[0].data(), { 3 });
		auto th2 = torch::from_blob(thK2[1].data(), { 3 });
		auto thc = torch::from_blob(ThMid, { 3 });

		at::Tensor vec1 = torch::sub(th1, csp); // Th1 - CSP
		at::Tensor vec2 = torch::sub(th2, csp); // Th2 - CSP
		at::Tensor vec3 = torch::sub(thc, csp); // ThC - CSP

		// plane's equation: ax + by + cz + d = 0
		// where (a,b,c) is normal vector of the plane

		// P1: intersects points CSP, Th1 and Th2
		setPlane(vec1, vec2, csp, plane_n[0], plane_d[0]);

		// P2: passing through the point ThC and perpendicular to the P1 plane
		setPlane(vec3, plane_n[0], csp, plane_n[1], plane_d[1]);
	}
	catch (const c10::Error& e) {
		LOG_E("error in setPlanes");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}

	return RET_OK;
}

int setPlane(IN at::Tensor& vec1, IN at::Tensor& vec2, IN at::Tensor& point, OUT at::Tensor& n, OUT at::Tensor& d) {
	try {
		at::Tensor N = torch::linalg_cross(vec1, vec2);
		auto N_norm = torch::linalg_norm(N); // normalization
		n = torch::div(N, N_norm);
		d = -torch::dot(point, n);
	}
	catch (const c10::Error& e) {
		LOG_E("error in setPlane");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}


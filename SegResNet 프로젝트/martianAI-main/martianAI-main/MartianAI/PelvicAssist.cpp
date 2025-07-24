#include <queue>
#include <cmath>
#include <vector>
#include <algorithm>
#include <vector>
#include <limits>
#include "InferenceControl.h"
#include "PelvicAssist.h"
#include "spline.hpp"
#include "US3DLog.h"

using namespace std;
enum ErrorCode {
	PELVICASSIST_SUCCESS = 600,

	PELVICASSIST_ERRORCODE_BASE = 601,

	PELVICASSIST_SEGRESNET_FAILED = 610,
	PELVICASSIST_MISMATCHED_LABEL_COUNT_FAILED,
	PELVICASSIST_LOAD_INFERENCE_FAILED,
	PELVICASSIST_NONE_PUBIS_FAILED,
	PELVICASSIST_NONE_URETHRA_FAILED,
	PELVICASSIST_NONE_ANUS_FAILED,
	PELVICASSIST_NONE_PELVICBRIM_FAILED,

	PELVICASSIST_PREPROCESS_FAILED = 620,
	PELVICASSIST_GETGROUP_FAILED,
	PELVICASSIST_SMALL_PELVICBRIM_FAILED,
	PELVICASSIST_REMOVESMALL_SEGMENTATION_FAILED,

	PELVICASSIST_GETPLANEPOINT_FAILED = 630,
	PELVICASSIST_GETPLANEPOINT_BOTTOM_FAILED,

	PELVICASSIST_GETNORMALVECTOR_FAILED = 640,
	PELVICASSIST_INVALID_CENTROID_ROI_FAILED,
	PELVICASSIST_INVALID_VAGINA_ROI_FAILED,
	PELVICASSIST_INVALID_LENGTH_FAILED
};
int torch_cc3d(torch::Tensor& output, const torch::Tensor& input);
#define CHECK_GPU_MEMORY FALSE

#if defined (CHECK_GPU_MEMORY) && (CHECK_GPU_MEMORY == TRUE)
void checkGPUMemory(string name) {
	size_t free_mem = 0;
	size_t total_mem = 0;

	cudaError_t error = cudaMemGetInfo(&free_mem, &total_mem);

	if (error != cudaSuccess) {
		LOG_E("{} cudaMemGetInfo failed:  {}", name, cudaGetErrorString(error));
		return;
	}

	LOG_I("{} Total Memory: {}", name, static_cast<float>(total_mem));
	LOG_I("{} Free Memory: {}", name, static_cast<float>(free_mem));
	LOG_I("{} Used Memory: {}", name, static_cast<float>(total_mem - free_mem));
}
#else
#define checkGPUMemory(name) ;
#endif

PelvicAssist::PelvicAssist(int isCPUOnly,std::string device_string)
{
	m_isCPUOnly = isCPUOnly;
	m_device_string = device_string;
}

PelvicAssist::~PelvicAssist()
{

}

int PelvicAssist::segResnet_sub(
	torch::jit::script::Module module,
	at::Tensor& input,
	at::Tensor& output,
	int* crop_roi,
	float scale,
	int* resize,
	int labelNum,
	float* labelThreshold,
	int* labelOffsetTable,
	int* errorCode
) {
	std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();
	try
	{
		std::vector<torch::jit::IValue> inputs;

		int roiDimA = crop_roi[1] - crop_roi[0];
		int roiDimB = crop_roi[3] - crop_roi[2];
		int roiDimC = crop_roi[5] - crop_roi[4];
		float length = 0.0;
		int targetRoiDimA = roiDimA;
		int targetRoiDimB = roiDimB;
		int targetRoiDimC = roiDimC;
		if (m_isCPUOnly == 0)
		{
			input = input.to(m_device_string);
		}

		input = input.reshape({ PELVICASSIST_SEG_IN_VOLUME_SIZE , PELVICASSIST_SEG_IN_VOLUME_SIZE , PELVICASSIST_SEG_IN_VOLUME_SIZE });
		input = input.permute({ 2,1,0 }).flatten();

#if !defined(DEV_BYPASS_INFERENCE) || (DEV_BYPASS_INFERENCE!=1)	
		input = input.reshape({ 1, 1, PELVICASSIST_SEG_IN_VOLUME_SIZE,
										PELVICASSIST_SEG_IN_VOLUME_SIZE,
										PELVICASSIST_SEG_IN_VOLUME_SIZE });

		input = input.index({ 0,
							0,
							Slice(crop_roi[0],crop_roi[1]),
							Slice(crop_roi[2],crop_roi[3]),
							Slice(crop_roi[4],crop_roi[5]) }).detach().clone();

		input = input.reshape({ 1, 1, roiDimA, roiDimB, roiDimC });
		if (scale != 1) {
			targetRoiDimA = int(targetRoiDimA * scale);
			targetRoiDimB = int(targetRoiDimB * scale);
			targetRoiDimC = int(targetRoiDimC * scale);
			LOG_T("rescale :{} to ({},{},{})", scale, targetRoiDimA, targetRoiDimB, targetRoiDimC);
			input = torch::nn::functional::interpolate(input,
				torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(
					std::vector<int64_t>({ targetRoiDimA, targetRoiDimB, targetRoiDimC }
					)));
		}

		input = input.flatten();

		//libtorch ±â¹Ý Normalization
		at::Tensor nonz_t = input.nonzero();
		at::Tensor min_t = input.min();
		at::Tensor max_t = input.index({ nonz_t }).max();
		input.index_put_({ nonz_t }, (input.index({ nonz_t }) - min_t) / (max_t - min_t));

		input = input.reshape({ 1, 1, targetRoiDimA, targetRoiDimB, targetRoiDimC });

		inputs.push_back(input);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_T("segResnet Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor();

		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_T("segResnet Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin_A = std::chrono::steady_clock::now();
		if (output.size(1) != labelNum)
		{
			*errorCode = PELVICASSIST_MISMATCHED_LABEL_COUNT_FAILED;
			LOG_E("Check Label Num : labelNum({}), output({}) : ecode={}", labelNum, output.size(1), *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
		output = torch::softmax(output, 1);
		auto options = torch::TensorOptions().dtype(output.dtype()).device(input.device());
		at::Tensor dummyTensor = torch::zeros({ 1, labelNum, PELVICASSIST_SEG_IN_VOLUME_SIZE,
															PELVICASSIST_SEG_IN_VOLUME_SIZE,
															PELVICASSIST_SEG_IN_VOLUME_SIZE }, options);

		if (scale != 1) {
			LOG_T("scale back[{}] to {},{},{}", scale, roiDimA, roiDimB, roiDimC);
			output = torch::nn::functional::interpolate(output,
				torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(
					std::vector<int64_t>({ roiDimA,roiDimB,roiDimC }
					)));//Image Resize
		}
		for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
		{
			at::Tensor temp = output.index({ 0,labelIndex,Slice(),Slice(),Slice() });
			dummyTensor.index_put_({ 0,labelIndex,
				Slice(crop_roi[0],crop_roi[1]),
				Slice(crop_roi[2],crop_roi[3]),
				Slice(crop_roi[4],crop_roi[5])
				},
				torch::threshold(temp, labelThreshold[labelIndex], 0)
			);
		}

		output = dummyTensor;

		at::Tensor argmax = output.argmax(1);
		c10::cuda::CUDACachingAllocator::emptyCache();

		output = torch::zeros_like(output).scatter_(1, argmax.unsqueeze(1), 1.0);
		c10::cuda::CUDACachingAllocator::emptyCache();
		// Labeling
		for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
		{
			output.index_put_({ 0,labelIndex,Slice(),Slice(),Slice() },
				torch::mul(output.index({ 0,labelIndex,Slice(),Slice(),Slice() }), labelOffsetTable[labelIndex]));   // Label Offset
		}
		//Merge
		output = torch::sum(output, 1).to(torch::kUInt8);
	}
	catch (const c10::Error& e) {
			*errorCode = PELVICASSIST_SEGRESNET_FAILED;
			LOG_E("error in segResnet inference : ecode={}", *errorCode);
			LOG_E("Message : {}", e.msg());
			return RET_CHECK_ERROR_CODE;
	}
	return RET_OK;
}

int PelvicAssist::segResnet(
	torch::jit::script::Module module,
	at::Tensor& input,
	at::Tensor& output,
	float* segMeanOut,
	float* pcaVectorOut,
	int labelNum,
	float* labelThreshold,
	int* processOnLabel,
	int volumeDim,
	int gauissianFilterMode,
	int morphProcessMode,
	int* errorCode,
	int* returnValue,
	unsigned char* outputBuffer
)
{
	float z_smooth = 0.2;
	Point Spacing = { 1, 1, 1 };
	CurveItemVector curveItemVector;
	GroupMap Groups;
	int resultValue = RET_OK;
	try
	{
		checkGPUMemory("segResnet start");

		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();
		*errorCode = PELVICASSIST_SUCCESS;
		if (volumeDim != PELVICASSIST_SEG_IN_VOLUME_SIZE)
		{
			input = input.reshape({ 1, 1, volumeDim, volumeDim, volumeDim });
			input = torch::nn::functional::interpolate(input,
				torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(
					std::vector<int64_t>({
										PELVICASSIST_SEG_IN_VOLUME_SIZE,
										PELVICASSIST_SEG_IN_VOLUME_SIZE,
										PELVICASSIST_SEG_IN_VOLUME_SIZE
						})));		//Image Resize
			input = input.flatten();
		}

#if defined(DEV_DEBUG_STANDARD_PLANE_BIN_OUT) && (DEV_DEBUG_STANDARD_PLANE_BIN_OUT==1)
		at::Tensor image = GetImageForStandardPlane(input);

#endif
		std::chrono::steady_clock::time_point begin_A = std::chrono::steady_clock::now();
		
		resultValue = segResnet_sub(
			module,
			input.detach().clone(),
			output,
			crop_roi,
			scale,
			NULL,
			labelNum,
			labelThreshold,
			labelOffsetTable,
			errorCode
		);
		if (resultValue != RET_OK) {
			LOG_E("segResnet_sub failed : ecode={}",*errorCode);
			return resultValue;
		}
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		SaveVolume("./OutData/outputData_3DVolume_inference.bin", output,
			PELVICASSIST_SEG_IN_VOLUME_SIZE, PELVICASSIST_SEG_IN_VOLUME_SIZE, PELVICASSIST_SEG_IN_VOLUME_SIZE);
#endif
		
		if (PELVICASSIST_SEG_IN_VOLUME_SIZE != PELVICASSIST_SEG_OUT_VOLUME_SIZE) {
			output = output.reshape({ 1, 1, PELVICASSIST_SEG_IN_VOLUME_SIZE, 
											PELVICASSIST_SEG_IN_VOLUME_SIZE, 
											PELVICASSIST_SEG_IN_VOLUME_SIZE });
			output = torch::nn::functional::interpolate(output.to(torch::kFloat), 
				torch::nn::functional::InterpolateFuncOptions().mode(torch::kNearest).size(std::vector<int64_t>(
					{
						PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
						PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
						PELVICASSIST_SEG_OUT_VOLUME_SIZE
					})));
			output = output.flatten();
		}
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		SaveVolume("./OutData/outputData_3DVolume_1_inference.bin", output);
#endif
		std::chrono::steady_clock::time_point End_A = std::chrono::steady_clock::now();
		LOG_T("segResnet Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(End_A - begin_A).count() / 1000.0f);
#else
		LOG_T("segResnet inference (bypass). load from binary(./OutData/outputData_3DVolume_1_inference.bin)");
		resultValue = LoadVolume("./OutData/outputData_3DVolume_1_inference.bin", output);
		if (resultValue != RET_OK) {
			*errorCode = PELVICASSIST_LOAD_INFERENCE_FAILED;
			LOG_E("segResnet failed : ecode={}", *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
#endif

		output = output.reshape({ PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
									PELVICASSIST_SEG_OUT_VOLUME_SIZE, 
									PELVICASSIST_SEG_OUT_VOLUME_SIZE });
#if defined (DEV_SKIP_PREPROCESS) && (DEV_SKIP_PREPROCESS==1)
		LOG_T("DEV_SKIP_PREPROCESS=1 skip preprocess");
#else
		resultValue = Preprocess(output, output, Groups , errorCode);
		if (resultValue != RET_OK) {
			LOG_E("error in Preprocess : ecode={}", *errorCode);
			return resultValue;
		}
#endif
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		SaveVolume("./OutData/outputData_PelvicAssist_2_preprocess.bin", output);
#endif
#if defined(SUPPORT_X_ROTATION) && (SUPPORT_X_ROTATION == 1)
		at::Tensor planePoints = torch::zeros({ 4,3 }).to(torch::kFloat32); // top,bottom , left, right
#else
		at::Tensor planePoints = torch::zeros({ 2,3 }).to(torch::kFloat32); // top,bottom
#endif
		at::Tensor outputData = output.detach().reshape({PELVICASSIST_SEG_OUT_VOLUME_SIZE,PELVICASSIST_SEG_OUT_VOLUME_SIZE,PELVICASSIST_SEG_OUT_VOLUME_SIZE});
		outputData = outputData.permute({ 2,1,0 }).flatten().to(at::kCPU).to(torch::kUInt8);
		unsigned char* outputfData = outputData.flatten().data_ptr<unsigned char>();
		memcpy(outputBuffer,
			outputfData,
			pow(PELVICASSIST_SEG_OUT_VOLUME_SIZE, 3) * sizeof(unsigned char));
		resultValue = getPlanePoints(output, planePoints, errorCode);
		if (resultValue != RET_OK) {
			LOG_E("error in getPlanePoints : ecode={}",*errorCode);
			return resultValue;
		}
		at::Tensor topbottom = planePoints.index({ Slice(0,2),Slice()}).detach().clone();
		float length = linalg_norm(topbottom[1] - topbottom[0],0,false).item<float>();
		LOG_T("Top:({},{},{}),Bottom:({},{},{}),length:{}",
			topbottom[0][0].item<float>(), topbottom[0][1].item<float>(), topbottom[0][2].item<float>(),
			topbottom[1][0].item<float>(), topbottom[1][1].item<float>(), topbottom[1][2].item<float>(),
			length
		);
		at::Tensor centroid = topbottom.to(torch::kFloat32).mean(0).to(torch::kFloat32);
		LOG_T("centroid:({},{},{})", centroid[0].item<float>(), 
									 centroid[1].item<float>(), 
									 centroid[2].item<float>());
		
#if defined(DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		for (int i = 0; i < planePoints.size(0); i++) {
			debugPoints.push_back({ planePoints[i][0].item<float>(),planePoints[i][1].item<float>(),planePoints[i][2].item<float>()});
		}
		debugPoints.push_back({ centroid[0].item<float>(),centroid[1].item<float>(),centroid[2].item<float>() });
#endif
		if (centroid[0].item<float>() < centroid_roi[0][0] || centroid[0].item<float>() > centroid_roi[0][1] ||
			centroid[1].item<float>() < centroid_roi[1][0] || centroid[1].item<float>() > centroid_roi[1][1] ||
			centroid[2].item<float>() < centroid_roi[2][0] || centroid[2].item<float>() > centroid_roi[2][1] )
		{
			*errorCode = PELVICASSIST_INVALID_CENTROID_ROI_FAILED;
			LOG_E("error in invalid centroid roi : ecode={}", *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
		if (length < lengthMin) {
			*errorCode = PELVICASSIST_INVALID_LENGTH_FAILED;
			LOG_E("error in length({}) < lengthMin({}) : ecode={}",length,lengthMin, *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
		// centroid
		segMeanOut[0] = centroid[0].item<float>();
		segMeanOut[1] = centroid[1].item<float>();
		segMeanOut[2] = centroid[2].item<float>();
		// top
		segMeanOut[3] = topbottom[0][0].item<float>();
		segMeanOut[4] = topbottom[0][1].item<float>();
		segMeanOut[5] = topbottom[0][2].item<float>();

		// bottom 
		segMeanOut[6] = topbottom[1][0].item<float>();
		segMeanOut[7] = topbottom[1][1].item<float>();
		segMeanOut[8] = topbottom[1][2].item<float>();

		resultValue = GetNormalVector(topbottom, &pcaVectorOut[0]);
		if (resultValue != RET_OK) {
			*errorCode = PELVICASSIST_GETNORMALVECTOR_FAILED;
			LOG_E("error in GetNormalVector : ecode={}", *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
#if defined (SUPPORT_X_ROTATION) && (SUPPORT_X_ROTATION == 1)
		at::Tensor leftright = planePoints.index({Slice(2,4),Slice()}).detach().clone();
		LOG_T("Left({},{},{}),Right({},{},{})",
			leftright[0][0].item<float>(), leftright[0][1].item<float>(), leftright[0][2].item<float>(),
			leftright[1][0].item<float>(), leftright[1][1].item<float>(), leftright[1][2].item<float>()
		);
		// left
		segMeanOut[9]  = leftright[0][0].item<float>();
		segMeanOut[10] = leftright[0][1].item<float>();
		segMeanOut[11] = leftright[0][2].item<float>();

		// right
		segMeanOut[12] = leftright[1][0].item<float>();
		segMeanOut[13] = leftright[1][1].item<float>();
		segMeanOut[14] = leftright[1][2].item<float>();

		leftright[0][0] = 0; // force x=0 to get vector(y,z)
		leftright[1][0] = 0; // force x=0 to get vector(y,z)
		//resultValue = GetNormalVector(leftright, &pcaVectorOut[3*3]);
#endif
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		at::Tensor temp = output.detach().clone();
		int markSize = 1;
		
		for (int i = 0; i < debugPoints.size(); i++) {
			LOG_T("debugPoint[{}] ({},{},{})",i, int(debugPoints[i].x), int(debugPoints[i].y), int(debugPoints[i].z));
			temp.index_put_({
				Slice(int(debugPoints[i].x) - markSize,int(debugPoints[i].x) + markSize + 1),
				Slice(int(debugPoints[i].y) - markSize,int(debugPoints[i].y) + markSize + 1),
				Slice(int(debugPoints[i].z) - markSize,int(debugPoints[i].z) + markSize + 1)}, DEV_DEBUG_MARK_VALUE);
		}
		SaveVolume("./OutData/outputData_PelvicAssist_3_Points.bin", temp);
#endif
#if defined(DEV_DEBUG_STANDARD_PLANE_BIN_OUT) && (DEV_DEBUG_STANDARD_PLANE_BIN_OUT == 1)
		GetStandardPlane(image.to("cpu"), centroid, pcaVectorOut);
#endif
	}
	catch (const c10::Error& e) {
		*errorCode = PELVICASSIST_ERRORCODE_BASE;
		LOG_E("error in segResnet inference : ecode={}",*errorCode);
		LOG_E("Message : {}", e.msg());
		return RET_CHECK_ERROR_CODE;
	}
	

	return RET_OK;
}

int PelvicAssist::GetNormalVector(at::Tensor planePoints , float* pcaVectorOut) {
	try {
		int returnValue = RET_OK;
		float smooth = 0.9999;
		int shake = 5;
		int shaketoken = 5; // Odd number 5, 7, 9 ...
		int tokens = 2 * shaketoken;//  # should n* shaketoken
		float sx, sy, sz;
		vector<double> x, y, z;
		vector<double> t = Linspace(0, 1, planePoints.size(0));
		for (int i = 0; i < t.size(); i++) {
			x.push_back(planePoints[i][0].item<float>());
			y.push_back(planePoints[i][1].item<float>());
			z.push_back(planePoints[i][2].item<float>());
		}
		cubic_spline spline_x(t, x);
		cubic_spline spline_y(t, y);
		cubic_spline spline_z(t, z);

		PointVector pV = PointVector();
		pV.reserve(tokens);
		Point p;
		vector<double> t2 = Linspace(0, 1, tokens);
		for (int i = 0.0; i < t2.size(); i += 1) {
			p.x = (float)spline_x(t2[i]);
			p.y = (float)spline_y(t2[i]);
			p.z = (float)spline_z(t2[i]);
			pV.push_back(p);
		}

		for (int i = 1; i < shaketoken - 1; i++) {
			pV[i * int(tokens / shaketoken)].z += shake * pow(-1, i);  //# shake z value
		}
				
		at::Tensor spBase = toDevice(PointVector2Tensor(pV));
		at::Tensor sampleMean = toDevice(spBase.mean(0));
		spBase = torch::sub(spBase, sampleMean);
		
		auto outputs = torch::svd(spBase);
		at::Tensor v = std::get<2>(outputs).transpose(0, 1);
		
		for (int i = 0; i < 3; i++) {
			torch::Tensor magnitude = torch::sqrt(torch::sum(torch::pow(v[i], 2)));
			pcaVectorOut[i*3 + 0] = (v[i][0] / magnitude).item<float>();
			pcaVectorOut[i*3 + 1] = (v[i][1] / magnitude).item<float>();
			pcaVectorOut[i*3 + 2] = (v[i][2] / magnitude).item<float>();
			LOG_T("Normal{}:({},{},{})", i, pcaVectorOut[i * 3 + 0], pcaVectorOut[i * 3 + 1], pcaVectorOut[i * 3 + 2]);
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error : ");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

#if defined(DEV_BYPASS_INFERENCE) && (DEV_BYPASS_INFERENCE == 1)
int PelvicAssist::LoadVolume(const std::string& filepath, at::Tensor& volume) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try {
		unsigned char* buffer = new unsigned char[PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE];
		memset(buffer, 0x00, sizeof(unsigned char) * PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE);

		std::ifstream fin(filepath, std::ios::binary);
		fin.read(reinterpret_cast<char*>(buffer), PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
		fin.close();
		std::vector<unsigned char> fdata(buffer, buffer + PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
		volume = torch::tensor(fdata).to(torch::kUInt8);
	}
	catch (const c10::Error& e) {
		LOG_E("error : {}",filepath);
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}
#endif
#if defined(DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT == 1)
void PelvicAssist::SaveVolume(const std::string& filepath,at::Tensor volume , int dimA , int dimB , int dimC) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try {
		printTensorInfo(volume);
		volume = volume.reshape({ dimA,dimB,dimC});
		volume = volume.permute({ 2,1,0 });
		std::ofstream FILE1(filepath, std::ios::out | std::ofstream::binary);
		FILE1.write(reinterpret_cast<const char*>(volume.to(at::kCPU).to(torch::kUInt8).flatten().data_ptr<unsigned char>()),
			dimA * dimB * dimC * sizeof(unsigned char));
		FILE1.close();
	}
	catch (const c10::Error& e) {
		LOG_E("error : {}",filepath);
		LOG_E("Message : {}", e.msg());
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {} {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f, filepath);
}
#endif

int PelvicAssist::Tensor2PointVector(at::Tensor& t, PointVector& v) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	v.reserve(t.size(0));
	at::Tensor temp = t.to(torch::kFloat32).contiguous();
	float* pfloatT = temp.data_ptr<float>();
	auto const ptr = reinterpret_cast<Point*>(pfloatT);
	PointVector vtemp = PointVector(ptr, ptr + t.size(0));
	v = vtemp;
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}

vector<double> PelvicAssist::Linspace(double startnum, double endnum, int num)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	std::vector<double> linspaced;
	if (num == 0) { return linspaced; }
	if (num == 1) {
		linspaced.push_back(startnum);
		return linspaced;
	}
	linspaced.reserve(num);
	float delta = (endnum - startnum) / (num - 1);
	for (int i = 0; i < num - 1; ++i){
		linspaced.push_back(startnum + delta * i);
	}
	linspaced.push_back(endnum);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return linspaced;
}

at::Tensor PelvicAssist::LinspaceT(double startnum, double endnum, int num)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	at::Tensor linspaced = torch::zeros({num});
	linspaced = toDevice(linspaced);
	if (num == 0) { return linspaced; }
	if (num == 1) {
		linspaced.index_put_({ 0 }, startnum);
		return linspaced;
	}
	float delta = (endnum - startnum) / (num - 1);
	for (int i = 0; i < num - 1; ++i) {
		linspaced.index_put_({ i }, startnum + delta * i);
	}
	linspaced.index_put_({num-1}, endnum);
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return linspaced;
}

int PelvicAssist::Preprocess(
	at::Tensor& input,
	at::Tensor& output,
	GroupMap& Groups,
	int* errorCode
) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try {
		int resultValue = RET_OK;
		// Step1. get Groups Start
		resultValue = GetGroup(Groups,input,labelNum,labelOffsetTable);
		if (resultValue != RET_OK) {
			*errorCode = PELVICASSIST_GETGROUP_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		if (Groups.count(PELVICBRIM_KEY) != 1 || Groups.at(PELVICBRIM_KEY).size() < 1) {
			LOG_E("PELVICBRIM is required!!");
			*errorCode = PELVICASSIST_NONE_PELVICBRIM_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		if (Groups.at(PELVICBRIM_KEY)[0].size(0) < PELVICBRIM_VOXEL_MIN) {
			LOG_E("PELVICBRIM is too small({}<{})!!", Groups.at(PELVICBRIM_KEY)[0].size(0), PELVICBRIM_VOXEL_MIN);
			*errorCode = PELVICASSIST_SMALL_PELVICBRIM_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		// Groups End
		// Step2. 
		output = torch::zeros_like(output);
		for (int i = 1; i < labelNum; i++) { // 0 is BG
			resultValue = RemoveSmall(Groups, output, labelOffsetTable[i] , labelMaxCountTable[i] , labelSizeTable[i]);
			if (resultValue != RET_OK) {
				LOG_E("RemoveSmall {}", labelOffsetTable[i]);
				*errorCode = PELVICASSIST_REMOVESMALL_SEGMENTATION_FAILED;
				return RET_CHECK_ERROR_CODE;
			}
		}
		// DEBUG
#if !defined(NDEBUG)
		LOG_T("Validate Groups");
		GroupMap tempGroup;
		GetGroup(tempGroup, output, labelNum, labelOffsetTable);
#endif
		//
	}
	catch (const c10::Error& e) {
		*errorCode = PELVICASSIST_PREPROCESS_FAILED;
		LOG_E("error : ecode={}",*errorCode);
		LOG_E("Message : {}", e.msg());
		return RET_CHECK_ERROR_CODE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("PELVICASSIST Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}

float PelvicAssist::distance2d(float x1, float y1, float x2, float y2) {
	return sqrt(std::pow((x2 - x1),2) + std::pow((y2 - y1),2));
}
float PelvicAssist::distance3d(float x1, float y1, float z1, float x2, float y2, float z2) {
	return sqrt(std::pow((x2 - x1), 2) + std::pow((y2 - y1), 2) + std::pow((z2 - z1), 2));
}

int PelvicAssist::RemoveSmall(GroupMap& Groups, at::Tensor& output,int labelKey,int maxCount, int minSize) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try
	{
		if (Groups.count(labelKey) == 1)
		{
			TensorVector lt = Groups.at(labelKey);
			TensorVector newlt;
			newlt.reserve(min(maxCount, int(lt.size())));
			std::sort(lt.begin(),lt.end(),less_than_size());
			at::Tensor points;
			int ltsize = lt.size();
			float sizeThreshold = 0;
			int pointSize = 0;
			for (int i = 0; i < maxCount && i < lt.size(); i++ ) {
				points = lt[i];
				pointSize = points.size(0);
				if (pointSize < minSize) {
					LOG_T("RemoveSmall[{}][{}]:skip points vocels:{} < min:{}", labelKey, i, pointSize, minSize);
					continue;
				}
				LOG_T("RemoveSmall[{}][{}]:push points vocels:{} min:{}}", labelKey, i , pointSize,minSize);
				newlt.push_back(points);
				points = points.to(torch::kInt64).t();
				output.index_put_({ points[0],points[1], points[2] }, labelKey);
			}
			output = output.to(torch::kUInt8);
			Groups.erase(labelKey);
			Groups.insert(make_pair(labelKey, newlt));
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error : ");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

	return RET_OK;
}

int PelvicAssist::GetGroup(
	GroupMap& Groups,
	at::Tensor input,
	int labelNum,
	int* labelOffsetTable
)
{
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try
	{
		at::Tensor InTensor;
		for (int labelIndex = 0; labelIndex < labelNum; labelIndex++)
		{
			if (labelOffsetTable[labelIndex] > 0)
			{
				TensorVector group;
				int returnValue = RET_OK;
				returnValue = cc3d(group, (input == labelOffsetTable[labelIndex]));
				if (returnValue != RET_OK) {
					LOG_E("Message : cc3d");
				}
				LOG_T("{}[{}] label's components: {}", labelName[labelIndex],labelOffsetTable[labelIndex], group.size());
				std::sort(group.begin(), group.end(), less_than_size());
				std::ostringstream stream;
				for (int i = 0; i < group.size() && i < MAX_COMPONENTS; i++) {
					stream.str(std::string());
					stream << "\t" << i + 1 << "th: " << group[i].size(0) << " voxel";
					LOG_T("{}", stream.str());
				}
				Groups.insert(make_pair(labelOffsetTable[labelIndex], group));
			}
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error : ");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}


int PelvicAssist::cc3d(TensorVector& outlist, at::Tensor& inTensor) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	
	typedef struct { int x; int y; int z; } Point;
	try
	{
		if (m_isCPUOnly == 0
#if defined (DEV_FORCE_CPU_CC3D) && (DEV_FORCE_CPU_CC3D==1)
			&& false
#endif
			) {
			int returnValue = RET_FAILURE;
			at::Tensor ccTensor;
			at::Tensor uniqIdTensor;
			int labelid = 0;
			returnValue = torch_cc3d(ccTensor, inTensor.to(torch::kUInt8));
			if (returnValue != RET_OK) {
				LOG_E("error in torch_cc3d");
				return RET_FAILURE;
			}
			uniqIdTensor = std::get<0>(torch::_unique(ccTensor.flatten(), false, false));
			for (int i = 0; i < uniqIdTensor.size(0); i++) {
				labelid = uniqIdTensor.index({ i }).item<int>();
				if (labelid != 0) {
					outlist.push_back((ccTensor == labelid).nonzero());
				}
			}
		}
		else {
			int sizeX, sizeY, sizeZ;
#if defined (DEV_FORCE_CPU_CC3D) && (DEV_FORCE_CPU_CC3D==1)
			inTensor = inTensor.to(at::kCPU);
#endif
			inTensor = inTensor.to(torch::kBool);
			if (inTensor.dim() == 3) {
				sizeX = inTensor.size(0);
				sizeY = inTensor.size(1);
				sizeZ = inTensor.size(2);
			}
			else {
				LOG_E("error in cc3d : invalid Dimension");
				return RET_FAILURE;
			}
			at::Tensor outTensor = torch::zeros({ sizeX, sizeY, sizeZ }).to(torch::kUInt8);

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

			bool* inBuffer = inTensor.flatten().data_ptr<bool>();
			UINT8* outBuffer = outTensor.flatten().data_ptr<UINT8>();
			std::queue<Point> queue;
			Point p;
			int mappedLabel = 1;
			int newLabel = mappedLabel;
			int compCount = 0;
			int curPos = 0;
			int voxCnt = 0;

			for (int z = 1; z < sizeZ - 1 && compCount < MAX_COMPONENTS; z++) {
				for (int y = 1; y < sizeY - 1 && compCount < MAX_COMPONENTS; y++) {
					for (int x = 1; x < sizeX - 1 && compCount < MAX_COMPONENTS; x++) {
						curPos = (sizeXY * z) + (sizeX * y) + x;

						if (inBuffer[curPos] == true && outBuffer[curPos] == 0) { // if unchecked
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
								++voxCnt;

								// loop: visit all neighbors
								for (int i = 0; i < CCL_CONNECTIVITY_26; i++) {
									if (curPos + Neighbors[i] < 0 || curPos + Neighbors[i] > sizeXYZ) {
										continue;
									}
									// check unchecked coordinate
									if (inBuffer[curPos + Neighbors[i]] == true && outBuffer[curPos + Neighbors[i]] == 0) {
										// update label to output
										outBuffer[curPos + Neighbors[i]] = newLabel;

										// push current position into queue
										queue.push({ p.x + CoordMap[i][0], p.y + CoordMap[i][1], p.z + CoordMap[i][2] });
									}
								}
							}
							// update voxel count and centroid
							++compCount;
							++newLabel;
							voxCnt = 0;
						}
					} // for sizeX
				} // for sizeY
			} // for sizeZ
			outlist.reserve(compCount);
			for (int ci = 0; ci < compCount; ci++) {
				outlist.push_back((outTensor == (int)mappedLabel + ci).nonzero());
			}
#if defined (DEV_FORCE_CPU_CC3D) && (DEV_FORCE_CPU_CC3D==1)
			inTensor = toDevice(inTensor);
#endif
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error :");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}	
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}

at::Tensor PelvicAssist::toDevice(const at::Tensor& t) {
	if (m_isCPUOnly == 0 && t.device() == at::kCPU) // GPU
	{
		return t.to(m_device_string);
	}
	return t;
}

at::Tensor PelvicAssist::PointVector2Tensor(PointVector& pointVector) {
	return toDevice(torch::from_blob(pointVector.data(), { (long long)pointVector.size(),3 }, torch::TensorOptions().dtype(torch::kFloat32)));
}

at::Tensor PelvicAssist::rotate_y(float angle, at::Tensor& point) {
	at::Tensor rad = torch::deg2rad(torch::tensor(angle));
	at::Tensor matrix = torch::zeros({3, 3}).to(torch::kFloat32);
	matrix[0][0] = torch::cos(rad);
	matrix[0][1] = 0;
	matrix[0][2] = torch::sin(rad);

	matrix[1][0] = 0;
	matrix[1][1] = 1;
	matrix[1][2] = 0;

	matrix[2][0] = -torch::sin(rad);
	matrix[2][1] = 0;
	matrix[2][2] = torch::cos(rad);

	matrix = toDevice(matrix);
	point = toDevice(point);
	at::Tensor rotated_point = torch::matmul(matrix.to(torch::kFloat32), point.to(torch::kFloat32));
	LOG_T("rotate_y angle:{} point:({},{},{})->({},{},{})", angle, 
		point[0].item<float>(), point[1].item<float>(), point[2].item<float>(),
		rotated_point[0].item<float>(), rotated_point[1].item<float>(), rotated_point[2].item<float>());
	return rotated_point.to(torch::kFloat32);
}

float PelvicAssist::getAngle(at::Tensor point){
	at::Tensor phi;
	if (point.size(0) == 3) {
		phi = torch::arctan2(torch::sqrt(torch::pow(point[0],2) + torch::pow(point[1],2)), point[2]);
	} else if (point.size(0) == 2) {
		phi = torch::arctan2(point[0], point[1]);
	}
	return torch::rad2deg(phi).item<float>();
}

torch::Tensor PelvicAssist::linalg_norm(torch::Tensor tensor, int dim = 0, bool keepdim = false) {
	return torch::sqrt(torch::sum(torch::pow(tensor, 2), { dim }, keepdim));
}

int PelvicAssist::getPathPoint(at::Tensor points1, at::Tensor points2, float angleRangeMin, float angleRangeMax , at::Tensor& out,int mode) { // 0 short, 1 longest
	at::Tensor _point;
	float _distance = 0;
	bool found = FALSE;
	if (mode == 0)
		_distance = std::numeric_limits<float>::max();

	points1 = toDevice(points1);
	points2 = toDevice(points2);
	LOG_T("getPathPoint:points1.size:{}", points1.size(0));
	LOG_T("getPathPoint:points2.size:{}", points2.size(0));
	for (int i = 0; i < points1.size(0); i+=2) {
		for (int j = 0; j < points2.size(0); j+=2) {
			float distance = linalg_norm(points2[j] - points1[i]).item<float>();
			if ((mode == 0 && distance < _distance) || (mode == 1 && distance > _distance)) {
				at::Tensor p1 = points1[i].detach().clone();
				at::Tensor p2 = points2[j].detach().clone();

				float angle = getAngle(p2 - p1);
				if (angle < angleRangeMin || angle > angleRangeMax)
					continue;
				_distance = distance;
				out[0] = p1;
				out[1] = p2;
				found = TRUE;
			}
		}
	}
	if (found == FALSE)
		return RET_FAILURE;
	return RET_OK;
}

int PelvicAssist::getPlanePoints(at::Tensor& object,at::Tensor& result,int* errorCode, int debugout) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int returnValue = RET_OK;

	try
	{
		int topType = 0;
		if (debugout) LOG_T("debugout mode :{}", debugout);
		at::Tensor out = object.detach().clone();

		int cropAngle = 20;
		int topAngle = 90;
		int bottomAngle = 90;
		int topPathPointMode = 0;

		at::Tensor pubis = torch::where(object == PUBIS_KEY, PUBIS_KEY, 0);
		at::Tensor urethra = torch::where(object == URETHRA_KEY, URETHRA_KEY, 0);
		at::Tensor vagina = torch::where(object == VAGINA_KEY, VAGINA_KEY, 0);
		at::Tensor anus = torch::where(object == ANUS_KEY, ANUS_KEY, 0);

		if (pubis.nonzero().size(0) < 1) {
			*errorCode = PELVICASSIST_NONE_PUBIS_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		if (urethra.nonzero().size(0) < 1) {
			*errorCode = PELVICASSIST_NONE_URETHRA_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		if (anus.nonzero().size(0) < 1) {
			*errorCode = PELVICASSIST_NONE_ANUS_FAILED;
			return RET_CHECK_ERROR_CODE;
		}
		if (vagina.nonzero().size(0) > 0) {
			at::Tensor vaginaTemp = vagina.nonzero().to(torch::kFloat16).mean(0);
			LOG_T("vaginaCenterXYZ {},{},{}", vaginaTemp[0].item<int>(), vaginaTemp[1].item<int>() , vaginaTemp[2].item<int>());
			if (vaginaTemp[0].item<float>() < centroid_roi[0][0] || vaginaTemp[0].item<float>() > centroid_roi[0][1] ||
				vaginaTemp[1].item<float>() < centroid_roi[1][0] || vaginaTemp[1].item<float>() > centroid_roi[1][1] ||
				vaginaTemp[2].item<float>() < centroid_roi[2][0] || vaginaTemp[2].item<float>() > centroid_roi[2][1])
			{
				*errorCode = PELVICASSIST_INVALID_VAGINA_ROI_FAILED;
				LOG_E("error in invalid vagina roi : ecode={}", *errorCode);
				return RET_CHECK_ERROR_CODE;
			}
				
		}

		at::Tensor born = torch::where(torch::bitwise_or(object == PELVICBRIM_KEY, object == PUBIS_KEY), PELVICBRIM_KEY, 0);
		at::Tensor points = born.nonzero().to(torch::kFloat32);
		at::Tensor center = points.mean(0).to(torch::kFloat32);
		points = points.t();
		float minV = points[0].min().item<float>();
		float maxV = points[0].max().item<float>();
		LOG_T("min:{} , max:{}", minV, maxV);
		LOG_T("center:({},{},{})", center[0].item<float>(), center[1].item<float>(), center[2].item<float>());

		at::Tensor minTensor = torch::tensor({ minV - center[0].item<float>(), float(0.0), float(0.0) });
		at::Tensor maxTensor = torch::tensor({ maxV - center[0].item<float>(), float(0.0), float(0.0) });

		at::Tensor rotatedMinLeft = rotate_y(-cropAngle, minTensor) + center; ;
		at::Tensor rotatedMinRight = rotate_y(+cropAngle, minTensor) + center;
		at::Tensor rotatedMaxLeft = rotate_y(+cropAngle, maxTensor) + center;
		at::Tensor rotatedMaxRight = rotate_y(-cropAngle, maxTensor) + center;
		int cropZMin = int(min(rotatedMinLeft[2].item<INT16>(), rotatedMaxLeft[2].item<INT16>())) - 1;
		int cropZMax = int(max(rotatedMinRight[2].item<INT16>(), rotatedMaxRight[2].item<INT16>())) + 1;
		LOG_T("cropZMin:{} , cropZMax:{}", cropZMin, cropZMax);

		born.index_put_({ Slice(),Slice(),Slice(None,cropZMin) }, 0);
		born.index_put_({ Slice(),Slice(),Slice(cropZMax,None) }, 0);
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		if (debugout)
			SaveVolume("./OutData/outputData_3DVolume_cropped.bin", born);
#endif
		
		int maskOffset = 2;
		at::Tensor pubisMean = pubis.nonzero().to(torch::kFloat32).mean(0).t();
		result[0] = pubisMean;
		LOG_T("top from pubis {},{},{}", result[0][0].item<int>(), result[0][1].item<int>(), result[0][2].item<int>());
		
		LOG_T("Bottom");
		int deltaX = anus.nonzero().t()[0].min().item<INT16>();
		at::Tensor bottom = born.index({ Slice(deltaX + 1,None) ,Slice() ,Slice() });
		at::Tensor bottomMask = torch::zeros_like(bottom);
		bottomMask.index_put_({ Slice(maskOffset,None), Slice() , Slice() },
			bottom.index({ Slice(None, -maskOffset), Slice(), Slice() }));
		bottom = torch::bitwise_and(bottom != 0, bottomMask == 0);

		at::Tensor points2 = bottom.nonzero().t().to(torch::kFloat32);
		points2[0] += (float(deltaX) + float(1.0));
		points2 = points2.t();
		at::Tensor bottomPoint = torch::zeros({ 2,3 }).to(torch::kFloat32);
		for (int retry = 1; retry < 15; retry++) {
			returnValue = getPathPoint(center.unsqueeze(0),
				points2,
				bottomAngle - 5 * retry,
				bottomAngle + 5 * retry,
				bottomPoint,
				0);
			if (returnValue == RET_OK) {
				break;
			}
		}
		if (returnValue != RET_OK) {
			LOG_T("Bottom from Anus");
			for (int retry = 1; retry < 15; retry++) {
				returnValue = getPathPoint(center.unsqueeze(0),
					anus.nonzero().to(torch::kFloat),
					bottomAngle - 5 * retry,
					bottomAngle + 5 * retry,
					bottomPoint,
					0);
				if (returnValue == RET_OK) {
					break;
				}
			}
		}
		if (returnValue != RET_OK) {
			*errorCode = PELVICASSIST_GETPLANEPOINT_BOTTOM_FAILED;
			LOG_E("getPathPoint.bottom : ecode={}", *errorCode);
			return RET_CHECK_ERROR_CODE;
		}
		LOG_T("bottom(born) ({},{},{})", bottomPoint[1][0].item<int>(), bottomPoint[1][1].item<int>(), bottomPoint[1][2].item<int>());
		at::Tensor bottomX = bottomPoint[1][0];
		result[1] = anus.nonzero().to(torch::kFloat16).mean(0).t();
		result[1][1] = bottomPoint[1][1];
		LOG_T("bottom(anus mean) ({},{},{})", result[1][0].item<int>(), result[1][1].item<int>(), result[1][2].item<int>());


		int steps = torch::sqrt(torch::pow(result[1][0] - result[0][0], 2) + torch::pow(result[1][1] - result[0][1], 2)).item<int>() + 1;
		at::Tensor t = torch::linspace(0, 1, steps);
		at::Tensor linePoints = toDevice(result[0] + t.unsqueeze(1) * (result[1]-result[0]));
		bool foundUrethra = FALSE;

		at::Tensor urethraNearMean;
		int dyDefault = 0;
		for (int dy = dyDefault; dy < 5; dy++) {
			float urethraDistanceMin = -1;
			float oldDistance = 0;
			for (int i = 0; i < linePoints.size(0) / 2; i++) {
				int sY = linePoints[i][1].item<int>();
				int eY = linePoints[i][1].item<int>() + dy + 1;
				if (sY < 0)
					sY = 0;
				if (eY > urethra.size(0) - 1)
					eY = urethra.size(0) - 1;
				at::Tensor urethraTemp = urethra.index({ Slice(),Slice(sY,eY),Slice() }).nonzero();
				int urethraSemi2dCount = urethraTemp.size(0) / (dy + 1);
				LOG_T("i{} check urethra dy:{} [{},{}] count:{}",i, dy, sY, eY, urethraSemi2dCount);
				if (urethraSemi2dCount > 5) {
					foundUrethra = TRUE;
					at::Tensor p = urethraTemp.to(torch::kFloat16).mean(0).t();
					p[1] += sY;
					float d = torch::sqrt(torch::pow(torch::sum(linePoints[i] - p), 2)).item<float>();
					LOG_T("urethra found y:{}~{} urethra({},{},{}),line({},{},{}) d:{}", sY, eY, 
						p[0].item<float>(), p[1].item<float>(), p[2].item<float>(),
						linePoints[i][0].item<float>(), linePoints[i][1].item<float>(), linePoints[i][2].item<float>(),
						d);
					if (urethraDistanceMin == -1 || d < urethraDistanceMin) {
						urethraDistanceMin = d;
						urethraNearMean = linePoints[i];
						urethraNearMean[2] = p[2];
					}
					if (oldDistance != 0 && d > oldDistance) {
						break;
					}
					oldDistance = d;
				}
			}
			if (foundUrethra) {
				at::Tensor newResult = urethraNearMean;
				LOG_T("top ({},{},{}) -> ({},{},{})",
					result[0][0].item<float>(), result[0][1].item<float>(), result[0][2].item<float>(), 
					newResult[0].item<float>(), newResult[1].item<float>(), newResult[2].item<float>());
				result[0] = newResult;
				break;
			}				
		}
		
		if (foundUrethra == FALSE ){
			result[0][2] = urethra.nonzero().t()[2].to(torch::kFloat16).mean();
			LOG_I("foundUrethra:False top -> ({},{},{})", result[0][0].item<float>(), result[0][1].item<float>(), result[0][2].item<float>());
		}
		
		t = (pubisMean[0] - result[0][0]) / (result[1][0] - result[0][0]);
		result[0][0] = pubisMean[0];
		result[0][1] = pubisMean[1];
		result[0][2] = result[0][2] + t * (result[1][2] - result[0][2]);
		LOG_I("extended top -> ({},{},{})", result[0][0].item<float>(), result[0][1].item<float>(), result[0][2].item<float>());

		t = (bottomX - pubisMean[0]) / (result[1][0] - pubisMean[0]);
		result[1][2] = result[0][2] + t * (result[1][2] - result[0][2]);
		result[1][0] = bottomPoint[1][0];
		result[1][1] = bottomPoint[1][1];
		LOG_I("extended bottom -> ({},{},{})", result[1][0].item<float>(), result[1][1].item<float>(), result[1][2].item<float>());


#if defined(DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		debugPoints.push_back({ result[0][0].item<float>(),result[0][1].item<float>(),result[0][2].item<float>()});
#endif
		
#if defined(DEBUG_INFERENCE_STEP_BIN_OUT) && (DEBUG_INFERENCE_STEP_BIN_OUT ==1)
		debugPoints.push_back({result[1][0].item<float>(),result[1][1].item<float>(),result[1][2].item<float>());
#endif

#if defined (SUPPORT_X_ROTATION) && ( SUPPORT_X_ROTATION )
#if 0
		at::Tensor leftright;
		returnValue = getLeftRightPoint(object == PELVICBRIM_KEY,
								((result[0][2] + result[1][2]) / 2).item<int>(), 
								result[0],  // top
								result[1],  // bottom
								leftright); // leftright
		if (returnValue == RET_OK) {
		    int deltaX = (((result[0][0] + result[1][0]) / 2) - (leftright[0][0] + leftright[1][0]) / 2).item<int>();
			int deltaY = (((result[0][1] + result[1][1]) / 2) - (leftright[0][1] + leftright[1][1]) / 2).item<int>();
			int deltaZ = (((result[0][2] + result[1][2]) / 2) - (leftright[0][2] + leftright[1][2]) / 2).item<int>();
			result[2] = leftright[0];
			result[2][0] += deltaX;
			result[2][1] += deltaY;
			result[2][2] += deltaZ;
			
			result[3] = leftright[1];
			result[3][0] += deltaX;
			result[3][1] += deltaY;
			result[3][2] += deltaZ;
		}
#endif
#endif

	}
	catch (const c10::Error& e) {
		*errorCode = PELVICASSIST_GETPLANEPOINT_FAILED;
		LOG_E("error : ecode={}",*errorCode);
		LOG_E("Message : {}", e.msg());
		return RET_CHECK_ERROR_CODE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}

#if defined (SUPPORT_X_ROTATION) && ( SUPPORT_X_ROTATION )
int PelvicAssist::distance_point_to_line(at::Tensor points, at::Tensor& out, at::Tensor A,at::Tensor B) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int returnValue = RET_OK;
	vector<at::Tensor> result;
	try {
		LOG_T("distance_point_to_line:points:{}", points.size(0));
		A = toDevice(A);
		B = toDevice(B);
		at::Tensor direction = B - A;
		at::Tensor direction_length_squared = torch::dot(direction, direction);
		
		float distance = 0;
		float _distance_max = 0;
		at::Tensor P, t ,Q;
		for (int i = 0; i < points.size(0); i++) {
			P = points[i].to(torch::kFloat32);
			t = torch::dot(P - A, direction) / direction_length_squared;
			Q = A + t * direction;
			distance = linalg_norm(P - Q).item<float>();
			if (distance > _distance_max) {
				_distance_max = distance;
				result.clear();
				result.push_back(P.to(torch::kInt16));
			}
			else if (distance == _distance_max) {
				result.push_back(P.to(torch::kInt16));
			}
		}
		LOG_T("max distance is distance:{} ,count:{}", _distance_max, result.size());
	}
	catch (const c10::Error& e) {
		LOG_E("error :");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	if (result.size() > 0) {
		out = result[0].to(torch::kInt16);
		return RET_OK;
	}
	return RET_FAILURE;
}

int PelvicAssist::getLeftRightPoint(at::Tensor object, int z, at::Tensor top, at::Tensor bottom, at::Tensor& leftrightPoint, int maskOffset) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	int returnValue = RET_OK;
	try {
		bool debugout = FALSE;
		at::Tensor leftTemp;
		at::Tensor rightTemp;

		at::Tensor left = torch::zeros_like(object);
		at::Tensor leftMask = torch::zeros_like(left);
		int ymin = (int)((top[1].item<float>() + bottom[1].item<float>()) / 2 - 3);
		int ymax = (int)((top[1].item<float>() + bottom[1].item<float>()) / 2 + 3);
		left.index_put_({ Slice(top[0].item<int>(),bottom[0].item<int>()),Slice(ymin,ymax) ,Slice(None, z) }, 
			object.index({ Slice(top[0].item<int>(),bottom[0].item<int>()),Slice(ymin, ymax), Slice(None, z)}));
		leftMask.index_put_({ Slice(),Slice(),Slice(None,z - maskOffset)},
			left.index({Slice(),Slice(),Slice(maskOffset,z)}));
		left = torch::bitwise_and(left != 0, leftMask == 0);
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		if (debugout)
			SaveVolume("./out/utilPointLH.left.bin", left);
#endif
		returnValue = distance_point_to_line(left.nonzero(), leftTemp, top, bottom);
		if (returnValue != RET_OK) {
			LOG_W("leftPoint is empty");
			return RET_FAILURE;
		}
		
		at::Tensor right = torch::zeros_like(object);
		at::Tensor rightMask = torch::zeros_like(right);
		right.index_put_({Slice(top[0].item<int>(),bottom[0].item<int>()),Slice(ymin ,ymax), Slice(z ,None)},
			object.index({Slice(top[0].item<int>(),bottom[0].item<int>()),Slice(ymin, ymax), Slice(z, None)}));
		rightMask.index_put_({ Slice(),Slice(),Slice(z + maskOffset , None)}, 
			right.index({ Slice(),Slice(),Slice(z, - maskOffset)}));
		right = torch::bitwise_and(right != 0, rightMask == 0);
#if defined (DEV_INFERENCE_STEP_BIN_OUT) && (DEV_INFERENCE_STEP_BIN_OUT==1)
		if (debugout) {
			SaveVolume("./out/utilPointLH.right.bin", right);
		}
#endif
		returnValue = distance_point_to_line(right.nonzero(), rightTemp, top, bottom);
		if (returnValue != RET_OK) {
			LOG_W("rightPoint is empty");
			return returnValue;
		}
		leftrightPoint = torch::zeros({ 2,3 });
		leftrightPoint[0] = leftTemp;
		leftrightPoint[1] = rightTemp;
	}
	catch (const c10::Error& e) {
		LOG_E("error :");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return RET_OK;
}
#endif

int PelvicAssist::printTensorInfo(at::Tensor t) {
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

#if defined(DEV_DEBUG_STANDARD_PLANE_BIN_OUT) && (DEV_DEBUG_STANDARD_PLANE_BIN_OUT == 1)
at::Tensor PelvicAssist::GetImageForStandardPlane(at::Tensor& input) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	at::Tensor temp;
	if (PELVICASSIST_SEG_IN_VOLUME_SIZE != PELVICASSIST_SEG_OUT_VOLUME_SIZE) {
		temp = input.detach().clone();
		if (m_isCPUOnly == 0)
		{
			temp = temp.to(m_device_string);
		}
		temp = temp.reshape({ 1, 1, PELVICASSIST_SEG_IN_VOLUME_SIZE,
									PELVICASSIST_SEG_IN_VOLUME_SIZE, 
									PELVICASSIST_SEG_IN_VOLUME_SIZE });
		temp = torch::nn::functional::interpolate(temp,
			torch::nn::functional::InterpolateFuncOptions().mode(torch::kTrilinear).size(
				std::vector<int64_t>({
									PELVICASSIST_SEG_OUT_VOLUME_SIZE,
									PELVICASSIST_SEG_OUT_VOLUME_SIZE,
									PELVICASSIST_SEG_OUT_VOLUME_SIZE
					})));		//Image Resize
		temp = temp.flatten();
		temp = temp.reshape({ PELVICASSIST_SEG_OUT_VOLUME_SIZE,
						PELVICASSIST_SEG_OUT_VOLUME_SIZE,
						PELVICASSIST_SEG_OUT_VOLUME_SIZE });
	}
	else {
		temp = input.reshape({ PELVICASSIST_SEG_OUT_VOLUME_SIZE,
								PELVICASSIST_SEG_OUT_VOLUME_SIZE,
								PELVICASSIST_SEG_OUT_VOLUME_SIZE });
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
	return temp;
}

void PelvicAssist::GetStandardPlane(at::Tensor image, at::Tensor centroid, float* pcaVectorOut) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try {
		at::Tensor sliceImage = torch::zeros({ PELVICASSIST_SEG_OUT_VOLUME_SIZE ,PELVICASSIST_SEG_OUT_VOLUME_SIZE });
		at::Tensor normal = at::tensor({ pcaVectorOut[3 * 2 + 0],pcaVectorOut[3 * 2 +1],pcaVectorOut[3 * 2 + 2] });
		at::Tensor d = (-centroid.dot(normal));
		at::Tensor linspace = LinspaceT(0, PELVICASSIST_SEG_OUT_VOLUME_SIZE, PELVICASSIST_SEG_OUT_VOLUME_SIZE).to(torch::kInt16);
		vector<at::Tensor> XXZZ = torch::meshgrid({ linspace, linspace }, "ij");
		at::Tensor XX = XXZZ[0].to("cpu");
		at::Tensor ZZ = XXZZ[1].to("cpu");
		at::Tensor YY = (-normal[0].item<float>() * XX - normal[2].item<float>() * ZZ - d) * 1.0 / normal[1].item<float>();
		for (int z = 0; z < PELVICASSIST_SEG_OUT_VOLUME_SIZE; z++) {
			for (int x = 0; x < PELVICASSIST_SEG_OUT_VOLUME_SIZE; x++) {
				int y = YY[z][x].item<int>();
				if (y >= 0 && y < PELVICASSIST_SEG_OUT_VOLUME_SIZE) {
					sliceImage.index_put_({ z,x }, image.index({ x,y,z }));
				}
			}
		}
		Save2dVolume("./OutData/DEBUG_SliceImage.bin", sliceImage);
	}
	catch (const c10::Error& e) {
		LOG_E("error :");
		LOG_E("Message : {}", e.msg());
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);
}

void PelvicAssist::Save2dVolume(const std::string& filepath, at::Tensor& volume) {
	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
	try {
		std::ofstream FILE1(filepath, std::ios::out | std::ofstream::binary);
		FILE1.write(reinterpret_cast<const char*>(volume.to(at::kCPU).to(torch::kUInt8).flatten().data_ptr<unsigned char>()),
			PELVICASSIST_SEG_OUT_VOLUME_SIZE * PELVICASSIST_SEG_OUT_VOLUME_SIZE * sizeof(unsigned char));
		FILE1.close();
	}
	catch (const c10::Error& e) {
		LOG_E("error : {}", filepath);
		LOG_E("Message : {}", e.msg());
	}
	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
	LOG_T("time = {} {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f, filepath);
}
#endif

int PelvicAssist::Measure2DSagmentation(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, double rotate)
{
	try{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();
		if (m_isCPUOnly == 0)
		{
			input = input.to(m_device_string);
		}
		input.index_put_({ Slice() }, ((input.index({ Slice() }) / 255)));
		input = input.reshape({ height,width, 4 });
		if (rotate != 0)
		{
			int RotateNum = (int)rotate / 90;
			input = input.rot90(4 - (int64_t)RotateNum, { 0, 1 });
		}
		input = input.permute({ 2,0,1 }).unsqueeze(0);
		input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ PELVIC2DSEG_IMAGE_SIZE, PELVIC2DSEG_IMAGE_SIZE })).align_corners(true));
		at::Tensor inputRenderImage = input.slice(1, 0, 3);
		inputRenderImage = (inputRenderImage - inputRenderImage.mean()) / inputRenderImage.std();
		at::Tensor inputLevatorAniMask = input.slice(1, 3, 4);
		if (inputLevatorAniMask.max().item<float>() > 0)
		{
			inputLevatorAniMask = inputLevatorAniMask / inputLevatorAniMask.max();
		}
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(inputRenderImage);
		inputs.push_back(inputLevatorAniMask);
		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("Measure2DSagmentation Preprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor().to(at::kCPU);
		c10::cuda::CUDACachingAllocator::emptyCache();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("Measure2DSagmentation Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point beginC = std::chrono::steady_clock::now();
		output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ height,width })).align_corners(true));
		output = output.flatten();
		output = torch::sigmoid(output) * 255;
		std::chrono::steady_clock::time_point endC = std::chrono::steady_clock::now();
		LOG_I("Measure2DSagmentation Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(endC - beginC).count() / 1000.0f);
	}
	catch (const c10::Error& e) {
		LOG_E("error in Measure2DSagmentation");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}


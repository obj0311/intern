#include "Beautify.h"
#include "US3DLog.h"

Beautify::Beautify(int isCPUOnly, std::string deviceString)
{
	mIsCPUOnly = isCPUOnly;
	mDeviceString = deviceString;
}

Beautify::~Beautify()
{

}

// Input - Rank 1(size = image size) Tensor, Output - {1,3,width,height} Tensor
int Beautify::StyleTransfer(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, double rotate)
{
	int imageSize = width * height * 3;

	try
	{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0)
		{
			input = input.to(mDeviceString);
		}

		//libtorch 기반 Normalization
		input.index_put_({ Slice() }, ((input.index({ Slice() }) / 255) - 0.5) / 0.5);
		input = input.reshape({ height,width, 3 });

		if (rotate != 0)
		{
			int RotateNum = (int)rotate / 90;
			input = input.rot90(4 - (int64_t)RotateNum, { 0, 1 });				//Image Rotate(Torch rot90은 반시계 방향, 5D Viewer가 시계방향 이라 "4-RotateNum" 으로..
		}

		//input = input.permute({ 2,0,1 }).unsqueeze(0).flip(1);				// 임시코드 - 참고용 - BGR to RGB, (1024,1024,3) to (1, 3, 1024, 1024)
		input = input.permute({ 2,0,1 }).unsqueeze(0);							// RGB, (1024,1024,3) to (1, 3, 1024, 1024)

		input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ IN_IMAGE_SIZE, IN_IMAGE_SIZE })).align_corners(true));		//Image Resize

		std::vector<torch::jit::IValue> inputs;
		at::Tensor input_reshape = input.reshape({ 1, 3, IN_IMAGE_SIZE, IN_IMAGE_SIZE });

		inputs.push_back(input_reshape);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("styleTransfer Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		output = module.forward(inputs).toTensor().to(at::kCPU);           // 처리한 Data를 CPU mem에 넣기 위해
		c10::cuda::CUDACachingAllocator::emptyCache();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("styleTransfer Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		std::chrono::steady_clock::time_point beginC = std::chrono::steady_clock::now();
		output.index_put_({ Slice() }, ((output.index({ Slice() }) * 0.5) + 0.5));

		output = torch::nn::functional::interpolate(output, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ height,width })).align_corners(true));		//Image Resize

		output = torch::clip(output, 0.0, 1.0);
		output = torch::mul(output, 255.0);
		//output = output.squeeze(0).permute({ 1,2,0 }).flip(2);      // 임시코드 - 참고용 - RGB to BGR, (1,3,1024,1024) to (1024,1024,3)
		output = output.squeeze(0).permute({ 1,2,0 });      // RGB, (1,3,1024,1024) to (1024,1024,3)

		output = output.reshape({ imageSize });

		std::chrono::steady_clock::time_point endC = std::chrono::steady_clock::now();
		LOG_I("styleTransfer Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(endC - beginC).count() / 1000.0f);

	}
	catch (const c10::Error& e) {
		LOG_E("error in styleTransfer");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}

int Beautify::FaceClassification(torch::jit::script::Module module, at::Tensor& input, at::Tensor& output, int width, int height, int* isClearFace)
{
	int imageSize = width * height * 3;

	try
	{
		std::chrono::steady_clock::time_point begin_pre = std::chrono::steady_clock::now();

		if (mIsCPUOnly == 0)
		{
			input = input.to(mDeviceString);
		}

		//libtorch 기반 Normalization
		input.index_put_({ Slice() }, ((input.index({ Slice() }) / 255) - 0.5) / 0.5);
		input = input.reshape({ height, width, 3 });
	
		input = input.permute({ 2,0,1 }).unsqueeze(0);

		input = torch::nn::functional::interpolate(input, torch::nn::functional::InterpolateFuncOptions().mode(torch::kBilinear).size(std::vector<int64_t>({ FACECLASSIFICATION_IMAGE_SIZE, FACECLASSIFICATION_IMAGE_SIZE })).align_corners(true));		//Image Resize

		std::vector<torch::jit::IValue> inputs;
		at::Tensor input_reshape = input.reshape({ 1, 3, FACECLASSIFICATION_IMAGE_SIZE, FACECLASSIFICATION_IMAGE_SIZE });

		inputs.push_back(input_reshape);

		std::chrono::steady_clock::time_point end_pre = std::chrono::steady_clock::now();
		LOG_I("FaceClassification Postprocess time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end_pre - begin_pre).count() / 1000.0f);

		std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
		auto faceClassOut = module.forward(inputs).toTensor().to(at::kCPU);           // 처리한 Data를 CPU mem에 넣기 위해
		c10::cuda::CUDACachingAllocator::emptyCache();
		std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
		LOG_I("FaceClassification Forward time = {}", std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() / 1000.0f);

		faceClassOut = torch::softmax(faceClassOut, 1);
		float* faceProbTemp = faceClassOut.flatten().data_ptr<float>();

		float clearFaceProb = faceProbTemp[0];
		float wrongFaceProb = faceProbTemp[1];

		LOG_I("ClearFace Prob = {}, WrongFace Prob = {}", clearFaceProb, wrongFaceProb);

		if (0.0f < clearFaceProb)
		{
			*isClearFace = 1;
		}
		else
		{
			*isClearFace = 0;
		}
	}
	catch (const c10::Error& e) {
		LOG_E("error in FaceClassification");
		LOG_E("Message : {}", e.msg());
		return RET_FAILURE;
	}
	return RET_OK;
}
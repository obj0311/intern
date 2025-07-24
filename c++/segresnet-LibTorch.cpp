// segresnet-LibTorch.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#define _CRT_SECURE_NO_WARNINGS

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <filesystem>
#include <fstream>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNiftiImageIOFactory.h"
#include "itkPNGImageIOFactory.h"
#include "itkImageRegionConstIterator.h"


inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

inline torch::Tensor preprocess_itk(const std::string& img_path, torch::Device device, const std::string& device_str) {

    auto pre_s = std::chrono::high_resolution_clock::now();
    using ImageType = itk::Image<float, 3>;
    auto reader = itk::ImageFileReader<ImageType>::New();
    reader->SetFileName(img_path);
    reader->Update();

    ImageType::Pointer img = reader->GetOutput();
    auto region = img->GetLargestPossibleRegion();
    auto size = region.GetSize();

    int depth = size[2], height = size[1], width = size[0];

    // ITK -> flat vector -> Tensor
    // 1차원 벡터로 buffer에 저장
    std::vector<float> buffer(width * height * depth);
    itk::ImageRegionConstIterator<ImageType> it(img, region);
    size_t idx = 0;
    for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) buffer[idx] = it.Get();

    // 비제로 값만 모아서 mean/std 계산
    //평균 0, 표준편차1 이 되도록 정규화
    std::vector<float> nonzero;
    for (float v : buffer) if (v != 0.0f) nonzero.push_back(v);
    float mean = 0, stddev = 1;
    if (!nonzero.empty()) {
        // 계산 (std::accumulate, 변동성 계산), 오버플로우 방지해 직접 구현
        double sum = 0; for (float v : nonzero) sum += v;
        mean = static_cast<float>(sum / nonzero.size());
        double var = 0; for (float v : nonzero) var += (v - mean) * (v - mean);
        stddev = static_cast<float>(sqrt(var / nonzero.size()));
        if (stddev < 1e-8f) stddev = 1;
    }
    // 정규화 적용
    for (float& v : buffer) { if (v != 0.0f) v = (v - mean) / stddev; }

    // 기존: buffer[z*height*width + y*width + x]
    // 변경: buffer[(z * height + y) * width + x] 를 for루프에서 (x, y, z)읽어서 옮김

    std::vector<float> reordered(128 * 128 * 128, 0.0f);
    for (size_t z = 0; z < size[2]; ++z)
        for (size_t y = 0; y < size[1]; ++y)
            for (size_t x = 0; x < size[0]; ++x) {
                // ITK/Libtorch축 다르면 아래 index 바꿔야 함
                size_t orig_idx = z * size[1] * size[0] + y * size[0] + x;
                size_t target_idx = x * size[1] * size[2] + y * size[2] + z; // (x,y,z) → (z,y,x) 등
                reordered[target_idx] = buffer[orig_idx];
            }



    // LibTorch expects NCDHW: (Batch, Channel, Depth, Height, Width)
    // 모델에 input으로 넣기 전에 shape 변환
    auto tensor = torch::from_blob(reordered.data(), { 1, 1, depth, height, width }, torch::kFloat32).clone(); //여기
    tensor = tensor.to(device);
    
    auto pre_e = std::chrono::high_resolution_clock::now(); 
    auto pre = pre_e - pre_s;
    std::cout << "LibTorch_" << device_str << "_preprocess Time : " << std::chrono::duration<double>(pre).count() << " seconds\n";


    return tensor;
}

void save_nifti(const at::Tensor& output_ori, const std::string& out_path, const std::string& device_str) {

    // shape (1,1,128,128,128) 로 출력됨
    auto post_s = std::chrono::high_resolution_clock::now();
    auto output = output_ori.cpu();
    size_t num_classes = output.size(1);
    size_t depth = output.size(2), height = output.size(3), width = output.size(4);
    size_t voxel_count = depth * height * width;
    float* output_data = output.data_ptr<float>();


    const float threshold[6] = { 1.0, 0.8, 0.3, 0.1, 0.5, 0.5 };
    std::vector<std::vector<float>> probability_map(num_classes, std::vector<float>(voxel_count, 0.f));
    for (size_t c = 0; c < num_classes; ++c) {
        for (size_t v = 0; v < voxel_count; ++v) {
            float prob = sigmoid(output_data[c * voxel_count + v]);
            probability_map[c][v] = (prob > threshold[c]) ? prob : 0.f;
        }
    }
    std::vector<uint8_t> volume_array(voxel_count, 0);
    for (size_t v = 0; v < voxel_count; ++v) {
        float max_prob = 0.f;
        int max_label = 0;
        for (size_t c = 0; c < num_classes; ++c) {
            if (probability_map[c][v] > max_prob) {
                max_prob = probability_map[c][v];
                max_label = c + 1;
            }
        }
        volume_array[v] = (max_prob > 0.f) ? max_label : 0;
    }
    // ----------------------------------
    // 각 voxel 별로 가장 높은 확률값을 가지는 클래스를 선택
    // 클래스가 6개면 (0.1,0.1,0.1,0.5,0.1,0.1) 이렇게 확률을 가진다고 하면 저 0.5가 선택되는것


    std::vector<uint8_t> reorder_post(voxel_count, 0);
    for (size_t z = 0; z < depth; ++z)
        for (size_t y = 0; y < height; ++y)
            for (size_t x = 0; x < width; ++x) {
                size_t src_idx = z * height * width + y * width + x;         // 기존 인덱싱 (z, y, x)
                size_t tgt_idx = x * height * depth + y * depth + z;         // 저장시 (x, y, z) → 변경 필요시 조정
                reorder_post[tgt_idx] = volume_array[src_idx];
            }


    using LabelImageType = itk::Image<uint8_t, 3>;
    LabelImageType::Pointer label_image = LabelImageType::New();
    LabelImageType::RegionType regionnew;
    LabelImageType::IndexType start = { 0, 0, 0 };
    LabelImageType::SizeType sizenew = { width, height, depth };
    regionnew.SetSize(sizenew); regionnew.SetIndex(start);
    label_image->SetRegions(regionnew); label_image->Allocate(); label_image->FillBuffer(0);

    for (size_t z = 0; z < depth; ++z) {
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                LabelImageType::IndexType idx3d = { static_cast<long>(x), static_cast<long>(y), static_cast<long>(z) };
                size_t flat_idx = z * height * width + y * width + x;
                label_image->SetPixel(idx3d, reorder_post[flat_idx]); //여기 x,y,z 로 변환한 reorder_post 데이터를 setPixel로 복사 label_image 생성
            }
        }
    }
    auto writer = itk::ImageFileWriter<LabelImageType>::New();
    writer->SetFileName(out_path); 
    writer->SetInput(label_image);
    writer->Update();

    auto post_e = std::chrono::high_resolution_clock::now();
    auto post = post_e - post_s;
    std::cout << "LibTorch_" << device_str << "_postprocess Time : " << std::chrono::duration<double>(post).count() << " seconds\n";

}


void inference(const std::string& model_path, const std::string& img_path, const std::string& out_path, torch::Device device, const std::string& device_str) {
    itk::NiftiImageIOFactory::RegisterOneFactory();

    torch::jit::script::Module model = torch::jit::load(model_path, device);
    model.eval();
    model.to(device);

    torch::Tensor input_vol = preprocess_itk(img_path, device, device_str);
    auto s = std::chrono::high_resolution_clock::now();

    // Inference
    at::Tensor output = model.forward({ input_vol }).toTensor(); // Expect shape: (1, NUM_LABELS, D, H, W)

    auto e = std::chrono::high_resolution_clock::now();
    auto inf = e - s;
    std::cout << "LibTorch_" << device_str << "_Inference Time : " << std::chrono::duration<double>(inf).count() << " seconds\n";

    save_nifti(output, out_path, device_str);

}

int main(int argc, char* argv[]) {
    try {
    itk::NiftiImageIOFactory::RegisterOneFactory();
    itk::PNGImageIOFactory::RegisterOneFactory();
    // Command-line args: --model model.pt --input image.nii.gz --output output.nii.gz [--device cpu/gpu]
    std::string model_path, input_path, output_path, device_str ;
    torch::Device device = torch::kCPU;
    for (int i = 1; i < argc; ) {
        std::string arg = argv[i];
        if (arg == "--model") { model_path = argv[i + 1]; i += 2; }
        else if (arg == "--input") { input_path = argv[i + 1]; i += 2; }
        else if (arg == "--output") { output_path = argv[i + 1]; i += 2; }
        else if (arg == "--device") { device_str = argv[i + 1]; i += 2; }
        else { ++i; }
    }
    if (model_path.empty()) { std::cout << "pt 파일 경로를 입력하세요: "; std::getline(std::cin, model_path); }
    if (input_path.empty()) { std::cout << "입력 이미지 경로를 입력하세요: "; std::getline(std::cin, input_path); }
    if (output_path.empty()) { std::cout << "출력 이미지 이름을 설정하세요: "; std::getline(std::cin, output_path); }
    if (device_str.empty()) { std::cout << "디바이스명을 입력하세요(CPU, GPU 등): "; std::getline(std::cin, device_str); }

    if (device_str == "CPU") {
        device = torch::kCPU;
    }
    else {
        device = torch::kCUDA;
    }

    if (!std::filesystem::exists(model_path)) { std::cerr << "pt 파일이 존재하지 않습니다: " << model_path << std::endl; return 1; }
    if (!std::filesystem::exists(input_path)) { std::cerr << "입력 이미지가 존재하지 않습니다: " << input_path << std::endl; return 1; }

    for (int k = 0; k < 3;) {
        inference(model_path, input_path, output_path, device, device_str);
        k++;
    }

    }
    catch (const std::exception& ex) {
        std::cerr << "예외 발생: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

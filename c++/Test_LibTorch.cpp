#define _CRT_SECURE_NO_WARNINGS

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <windows.h>
#include <psapi.h> // Windows 메모리 측정을 위한 헤더
#include <filesystem> // C++17 이상
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


torch::Tensor preprocess(const char* img_path, int size = 512, torch::Device device = torch::kCUDA) {

    int width, height, channels;
    unsigned char* img = stbi_load(img_path, &width, &height, &channels, 3);

    std::vector<float> img_data(width* height* channels);
    for (int i = 0; i < width * height * channels; ++i) {
        img_data[i] = img[i] / 255.0f;
    }

    
    // Mat을 Tensor로 변환 (NCHW)
    torch::Tensor tensor = torch::from_blob(img_data.data(), { height, width, channels }, torch::kFloat32).permute({ 2,0,1 }).unsqueeze(0);
    tensor = (tensor - 0.5f) / 0.5f;


    tensor = tensor.to(device);


    return tensor;
}

// 후처리 함수 (Python의 postprocess와 동일)
void postprocess(const torch::Tensor& output_tensor, const char* save_path) {
    // 1. 텐서 -> NumPy 변환 및 차원 조정

    torch::Tensor output_np = output_tensor.squeeze(0); // (1,3,512,512) -> (3,512,512)
    output_np = output_np.permute({ 1, 2, 0 }); // CHW -> HWC (512,512,3)



    // 2. [-1,1] -> [0,255] 범위 변환
    output_np = (output_np * 0.5 + 0.5) * 255; // [-1,1] -> [0,255]
    output_np = torch::clamp(output_np, 0, 255);

    // 3. Tensor -> cv::Mat 변환
    output_np = output_np.to(torch::kCPU);
    output_np = output_np.toType(torch::kU8);

    output_np = output_np.reshape({ 512 * 512 * 3 });

    stbi_write_png(save_path, 512, 512, 3, output_np.data_ptr(), 512 * 3);

    //std::cout << "Image saved to " << save_path << std::endl;
}
void inference(torch::Device device, std::string file_path, const char* input_img_path) {
    torch::jit::script::Module model;
    model = torch::jit::load(file_path, device);
    model.to(device);

    auto pre_s = std::chrono::high_resolution_clock::now();
    torch::Tensor input_img = preprocess(input_img_path, 512, device);
    auto pre_e = std::chrono::high_resolution_clock::now();

    auto pre = pre_e - pre_s;

    torch::NoGradGuard no_grad;
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_img);

    auto s = std::chrono::high_resolution_clock::now();
    at::Tensor output = model.forward(inputs).toTensor();
    auto e = std::chrono::high_resolution_clock::now();

    auto inf = e - s;

    std::string device_name = (device == torch::kCUDA) ? "gpu" : "cpu";
    std::string filename = "output_libtorch_" + device_name + ".png";


    auto post_s = std::chrono::high_resolution_clock::now();
    postprocess(output, filename.c_str());
    auto post_e = std::chrono::high_resolution_clock::now();

    auto post = post_e - post_s;

    std::cout << "LibTorch_" << device_name << "_preprocess Time : " << std::chrono::duration<double>(pre).count() << " seconds\n";
    std::cout << "LibTorch_" << device_name << "_Inference Time : " << std::chrono::duration<double>(inf).count() << " seconds\n";
    std::cout << "LibTorch_" << device_name << "_postprocess Time : " << std::chrono::duration<double>(post).count() << " seconds\n";

}


int main(int argc, char* argv[]) {
    //std::string file_path = "D:/obj_data/portraitVue-Restoration-code/(lr = 1e-4, batch = 4)/all_24000_newcomposite.pt";
    //const char* input_img_path = "D:/obj_data/portraitVue-Restoration-code/input.jpg";
    std::string input_loc, model_loc, device;
    torch::Device device_name = torch::kCPU;

    for (int i = 1; i < argc; ) {
        std::string arg = argv[i];
        if (arg == "--model") {
            model_loc = argv[i + 1];
            i += 2;
        }
        else if (arg == "--input") {
            input_loc = argv[i + 1];
            i += 2;
        }
        else if (arg == "--device") {
            device = argv[i + 1];
            i += 2;
        }
        else {
            std::cerr << "Unknown option: " << arg << std::endl;
            return 1;
        }
    }


    if (input_loc.empty()) {
        std::cout << "입력 이미지 경로를 입력하세요: ";
        std::getline(std::cin, input_loc);
    }
    if (model_loc.empty()) {
        std::cout << "pt파일 모델 경로를 입력하세요: ";
        std::getline(std::cin, model_loc);
    }
    if (device.empty()) {
        std::cout << "디바이스명을 입력하세요(CPU, GPU 등): ";
        std::getline(std::cin, device);
    }
    //std::getline으로 직접 입력할때는 "" 따옴표를 제외하고 경로를 입력해야한다.


    if (!std::filesystem::exists(model_loc)) {
        std::cerr << "xml 파일이 존재하지 않습니다: " << model_loc << std::endl;
        return 1;
    }
    if (!std::filesystem::exists(input_loc)) {
        std::cerr << "입력 이미지가 존재하지 않습니다: " << input_loc << std::endl;
        return 1;
    }

    if (device == "CPU") {
        device_name = torch::kCPU;
    }
    else {
        device_name = torch::kCUDA;
    }
    for (int k = 0; k < 3;) {
        inference(device_name, model_loc, input_loc.c_str());
        k++;
    }

    std::cout << "LibTorch Model Size: " << std::filesystem::file_size(model_loc) / (1024 * 1024) << " MB\n";

    return 0;
}
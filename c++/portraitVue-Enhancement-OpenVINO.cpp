#define _CRT_SECURE_NO_WARNINGS

#include <openvino/openvino.hpp>
#include <iostream>
#include <random>
#include <algorithm>
#include <vector>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

int main(int argc, char* argv[])
{

        std::string bin_loc, xml_loc, device;
        std::string input_loc, output_loc;
        //const char* input_loc = nullptr;
        //const char* output_loc = nullptr;
        for (int i = 1; i < argc; ) {
            std::string arg = argv[i];
            if (arg == "--bin") {
                bin_loc = argv[i + 1];
                i += 2;
            }
            else if (arg == "--xml") {
                xml_loc = argv[i + 1];
                i += 2;
            }
            else if (arg == "--input") {
                input_loc = argv[i + 1];
                i += 2;
            }
            else if (arg == "--output") {
                output_loc = argv[i + 1];
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



        if (bin_loc.empty()) {
            std::cout << "bin 파일 경로를 입력하세요: ";
            std::getline(std::cin, bin_loc);
        }
        if (xml_loc.empty()) {
            std::cout << "xml 파일 경로를 입력하세요: ";
            std::getline(std::cin, xml_loc);
        }
        if (input_loc.empty()) {
            std::cout << "입력 이미지 경로를 입력하세요: ";
            std::getline(std::cin, input_loc);
        }
        if (output_loc.empty()) {
            std::cout << "출력 이미지 경로를 입력하세요: ";
            std::getline(std::cin, output_loc);
        }
        if (device.empty()) {
            std::cout << "OpenVINO 디바이스명을 입력하세요(CPU, GPU 등): ";
            std::getline(std::cin, device);
        }
        //std::getline으로 직접 입력할때는 "" 따옴표를 제외하고 경로를 입력해야한다.

        if (!std::filesystem::exists(bin_loc)) {
            std::cerr << "bin 파일이 존재하지 않습니다: " << bin_loc << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(xml_loc)) {
            std::cerr << "xml 파일이 존재하지 않습니다: " << xml_loc << std::endl;
            return 1;
        }
        if (!std::filesystem::exists(input_loc)) {
            std::cerr << "입력 이미지가 존재하지 않습니다: " << input_loc << std::endl;
            return 1;
        }

        // 1. Create OpenVINO Runtime Core
        ov::Core core;

        // 2. Compile Model(load xml / bin)
        auto model = core.read_model(xml_loc, bin_loc);
       

        auto compiled_model = core.compile_model(model, device);

        


        //size_t ov_mem_before = get_current_memory_usage();

        // 3. Create Inference Request
        auto infer_request = compiled_model.create_infer_request();





        // 4. Set Inputs
        int width, height, channels;

        auto pre_s = std::chrono::high_resolution_clock::now();

        unsigned char* img = stbi_load(input_loc.c_str(), &width, &height, &channels, 3);
  
        std::vector<float> img_data(width * height * channels);
        for (int i = 0; i < width * height * channels; ++i) {
            img_data[i] = img[i] / 255.0f;
        }

        //torch::Tensor img_tensor = torch::from_blob(img_data.data(), {height, width, channels}, torch::kFloat32).permute({2,0,1}).unsqueeze(0);

        std::vector<float> img_data_nchw(width * height * channels);
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    int src_idx = (h * width + w) * channels + c; // HWC → CHW
                    int dst_idx = c * (height * width) + h * width + w;
                    img_data_nchw[dst_idx] = img_data[src_idx];
                }
            }
        }

        for (auto& val : img_data_nchw) {
            val = (val - 0.5f) / 0.5f;
        }

        ov::Shape img_shape = { 1, 3, 512, 512 };
        ov::Tensor ov_img_tensor(ov::element::f32, img_shape, img_data_nchw.data());

        auto pre_e = std::chrono::high_resolution_clock::now();
        auto pre = pre_e - pre_s;

        // 입력 텐서 설정 

        infer_request.set_input_tensor(0, ov_img_tensor);
 


        // 5. Start Inference
        auto s = std::chrono::high_resolution_clock::now();
        infer_request.infer();
 

        auto e = std::chrono::high_resolution_clock::now();
        auto inf = e - s;


        // 6. Process Inference Results
        auto output_tensor = infer_request.get_output_tensor(0);

        float* output_data = output_tensor.data<float>();



        // 7. Results to Image 
        // (1,3,512,512) → (3,512,512) → (512,512,3) → [0,255] 변환

        auto post_s = std::chrono::high_resolution_clock::now();

        std::vector<unsigned char> interleaved(512 * 512 * 3);

        size_t plane_size = 512 * 512;
        for (int y = 0; y < 512; ++y) {
            for (int x = 0; x < 512; ++x) {
                size_t idx = y * 512 + x;
                float r = (output_data[0 * plane_size + idx] * 0.5f + 0.5f) * 255.0f;
                float g = (output_data[1 * plane_size + idx] * 0.5f + 0.5f) * 255.0f;
                float b = (output_data[2 * plane_size + idx] * 0.5f + 0.5f) * 255.0f;
                // interleaved: [R,G,B, R,G,B, ...]
                interleaved[3 * idx + 0] = static_cast<unsigned char>(std::clamp(r, 0.0f, 255.0f));
                interleaved[3 * idx + 1] = static_cast<unsigned char>(std::clamp(g, 0.0f, 255.0f));
                interleaved[3 * idx + 2] = static_cast<unsigned char>(std::clamp(b, 0.0f, 255.0f));
            }
        }


        auto post_e = std::chrono::high_resolution_clock::now();
        auto post = post_e - post_s;

        std::cout << "OpenVINO PreProcess Time: " << std::chrono::duration<double>(pre).count() << " seconds\n";

        std::cout << "OpenVINO Inference Time: " << std::chrono::duration<double>(inf).count() << " seconds\n";

        std::cout << "OpenVINO PostProcess Time: " << std::chrono::duration<double>(post).count() << " seconds\n";

        std::cout << "OpenVINO Model Size: " << (std::filesystem::file_size(bin_loc) + std::filesystem::file_size(xml_loc)) / (1024 * 1024) << " MB\n";

        stbi_write_png(output_loc.c_str(), 512, 512, 3, interleaved.data(), 512 * 3);
    
        return 0;

}

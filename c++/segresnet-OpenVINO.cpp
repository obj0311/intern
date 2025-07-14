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
#include "itkImage.h"


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

        



        // 3. Create Inference Request
        auto infer_request = compiled_model.create_infer_request();





        // 4. Set Inputs
 
        using ImageType = itk::Image<float, 3>;
        auto reader = itk::ImageFileReader<ImageType>::New();
        reader->SetFileName("input_image.nii.gz");
        reader->Update();
        ImageType::Pointer image = reader->GetOutput();

        // 예시: ITK에서 NumPy 스타일 배열로 변환 (pseudocode)
        auto region = image->GetLargestPossibleRegion();
        auto size = region.GetSize();
        std::vector<float> buffer(size[0] * size[1] * size[2]);
        itk::ImageRegionConstIterator<ImageType> it(image, region);
        size_t idx = 0;
        for (it.GoToBegin(); !it.IsAtEnd(); ++it, ++idx) {
            buffer[idx] = static_cast<float>(it.Get());
        }

        auto pre_s = std::chrono::high_resolution_clock::now();

  
        ov::Shape img_shape = { 1, 1, 128, 128, 128 };
        ov::Tensor ov_img_tensor(ov::element::f32, image_shape, buffer.data());

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

        auto post_s = std::chrono::high_resolution_clock::now();

        // 예시: argmax 계산 (multi-class segmentation)
        size_t voxel_count = output_shape[2] * output_shape[3] * output_shape[4];
        std::vector<uint8_t> label_map(voxel_count);
        for (size_t i = 0; i < voxel_count; ++i) {
            float max_val = output_data[i];
            int max_idx = 0;
            for (int c = 1; c < num_classes; ++c) {
                float val = output_data[c * voxel_count + i];
                if (val > max_val) {
                    max_val = val;
                    max_idx = c;
                }
            }
            label_map[i] = static_cast<uint8_t>(max_idx);
        }

        using LabelImageType = itk::Image<uint8_t, 3>;
        // label_map을 ITK 이미지로 변환 후 저장
        auto label_image = LabelImageType::New();
        // ... (버퍼를 ITK 이미지로 복사)
        auto writer = itk::ImageFileWriter<LabelImageType>::New();
        writer->SetFileName(output_loc);
        writer->SetInput(label_image);
        writer->Update();

        auto post_e = std::chrono::high_resolution_clock::now();
        auto post = post_e - post_s;

        std::cout << "OpenVINO PreProcess Time: " << std::chrono::duration<double>(pre).count() << " seconds\n";

        std::cout << "OpenVINO Inference Time: " << std::chrono::duration<double>(inf).count() << " seconds\n";

        std::cout << "OpenVINO PostProcess Time: " << std::chrono::duration<double>(post).count() << " seconds\n";

        std::cout << "OpenVINO Model Size: " << (std::filesystem::file_size(bin_loc) + std::filesystem::file_size(xml_loc)) / (1024 * 1024) << " MB\n";

        stbi_write_png(output_loc.c_str(), 512, 512, 3, interleaved.data(), 512 * 3);
    
        return 0;

}

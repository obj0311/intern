#include <torch/script.h>
#include <iostream>
#include <memory>
using namespace std;

int main() {                                     // --- (1)

    torch::jit::script::Module module;                                      // --- (3)
    try {
        module = torch::jit::load("D:/obj_data/vscode/traced_script_model.pt");                                 // --- (4)
    }
    catch (const c10::Error& e) {
        cerr << "error loading the module \n";                                // --- (5)
        return -1;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch::ones({ 1, 3, 224, 224 }));                       // --- (1)

    at::Tensor output = module.forward(inputs).toTensor();                 // --- (2)
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

    cout << "ok \n";                                                        // --- (6)
}
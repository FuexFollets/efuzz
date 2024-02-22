#include <fstream>
#include <ios>
#include <iostream>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>

#include <efuzz/encode.hpp>
#include <efuzz/train_encoder.hpp>

int main(int argc, char** argv) {
    std::vector<std::string> args(std::next(argv), std::next(argv, argc));

    for (const auto& arg: args) {
        std::cout << "Arg: " << arg << '\n';
    }

    const std::string dataset_path = args.at(0);
    const bool use_output_log = (args.size() > 1);
    const std::string output_log_path = use_output_log ? args.at(1) : "encoder_training.log";

    efuzz::Encoder<std::string, std::integral_constant<int, 10>> encoder;

    const std::size_t input_size = encoder.get_nn_input_size();
    const std::size_t output_size = encoder.get_nn_output_size();
    const std::size_t layer_count = 10; // arbitrary
    const float slope = static_cast<float>(input_size) / static_cast<float>(output_size);

    std::vector<std::size_t> layer_sizes;

    layer_sizes.push_back(input_size);

    for (std::size_t i = 0; i < layer_count; ++i) {
        layer_sizes.push_back(static_cast<std::size_t>(input_size - (slope * i)));
    }

    layer_sizes.push_back(output_size);

    encoder.set_encoding_nn_layer_sizes(layer_sizes);

    efuzz::EncoderTrainer<std::string, std::integral_constant<int, 10>> encoder_trainer(encoder);

    std::cout << "Training encoder with dataset: " << dataset_path << '\n';
    std::vector<std::string> dataset;
    std::ifstream dataset_file(dataset_path);
    std::string line;

    const std::size_t max_lines = 10;

    std::size_t line_count {};
    while (std::getline(dataset_file, line) && line_count++ < max_lines) {
        dataset.push_back(line);
    }

    encoder_trainer.set_dataset(std::make_shared<std::vector<std::string>>(dataset));

    const std::size_t random_iterations = 5; // arbitrary
    std::cout << "Training encoder with dataset size: " << dataset.size() << '\n';
    std::cout << "Training encoder using random training for " << random_iterations
              << " iterations per training iteration\n";

    const std::size_t training_iterations = 10000; // arbitrary

    float current_min_cost {};

    std::optional<std::ofstream> output_log;

    if (use_output_log) {
        output_log.emplace(output_log_path, std::ios_base::app);
    }

    for (std::size_t iteration {}; iteration < training_iterations; ++iteration) {
        // const auto res = encoder_trainer.train_random(random_iterations);
        const auto res = encoder_trainer.train_all();

        std::cout << "Iteration: " << iteration << '\n';
        std::cout << "res.original_cost: " << res.original_cost << '\n';
        std::cout << "res.modified_cost: " << res.modified_cost << '\n';

        current_min_cost = std::min(res.original_cost, res.modified_cost);

        const bool was_modified = encoder_trainer.apply_training_result(res);

        if (use_output_log && output_log.has_value()) {
            *output_log << "Iteration: " << iteration << '\n';
            *output_log << "res.original_cost: " << res.original_cost << '\n';
            *output_log << "res.modified_cost: " << res.modified_cost << '\n';
            *output_log << "was_modified: " << was_modified << '\n';
            *output_log << "--------------------------------\n\n";
            *output_log << std::flush;
        }

        std::cout << "was_modified: " << was_modified << '\n';

        std::cout << "--------------------------------\n\n";
    }
}

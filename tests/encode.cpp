#include <cstddef>
#include <string>

#include <efuzz/encode.hpp>
#include <type_traits>

int main() {
    efuzz::Encoder<std::string, std::integral_constant<int, 10>> encoder;

    const std::size_t input_size = encoder.get_nn_input_size();
    const std::size_t output_size = encoder.get_nn_output_size();

    std::vector<std::size_t> layer_sizes = {input_size, 10, 10, output_size};

    encoder.set_encoding_nn_layer_sizes(layer_sizes);

    auto res = encoder.encode("airplane");

    std::cout << "Encoded: " << res << '\n';
}

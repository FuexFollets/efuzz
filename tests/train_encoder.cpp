#include <iostream>
#include <string>
#include <type_traits>

#include <efuzz/encode.hpp>
#include <efuzz/train_encoder.hpp>

int main() {
    efuzz::Encoder<std::string, std::integral_constant<int, 10>> encoder;

    efuzz::TrainEncoder<std::string, std::integral_constant<int, 10>> encoder_trainer(encoder);
}

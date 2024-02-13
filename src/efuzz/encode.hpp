#ifndef EFUZZ_ENCODE_HPP
#define EFUZZ_ENCODE_HPP

#include <cstddef>
#include <type_traits>

#include <cereal/cereal.hpp>
#include <Eigen/Core>

#include <efuzz/cereal_eigen.hpp>
#include <efuzz/neural_network/neural_network.hpp>

namespace efuzz {
    template <typename StringT>
    requires requires { typename StringT::value_type; }

    class Encoder {
        public:

        using char_type = typename StringT::value_type;

        Encoder() = default;
        Encoder(const Encoder&) = default;
        Encoder(Encoder&&) = default;
        Encoder& operator=(const Encoder&) = default;
        Encoder& operator=(Encoder&&) = default;

        private:

        using _char_encoder_size = std::integral_constant<std::size_t, sizeof(char_type) * 8>;

        NeuralNetwork _word_vector_encoder_nn;
        Eigen::VectorXf _encoding_result;
    };
} // namespace efuzz

#endif // EFUZZ_ENCODE_HPP

#include <Eigen/Core>

#include <efuzz/encode.hpp>

namespace efuzz {
    template <typename StringT>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT>::encode(const StringT& word) -> encoding_result_type {
        reset_encoding_result();

        for (const auto& letter: word) {
            encode_letter(letter);
        }

        return _encoding_result;
    }

    template <typename StringT>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT>::encode_letter(const char_type& letter) -> this_type& {
        Eigen::Vector<float, char_encoder_size::value> letter_binary_encoding;

        char_type mask = 1;

        for (std::size_t i = 0; i < char_encoder_size::value; ++i) {
            letter_binary_encoding(i) = (letter & mask) ? 1.0F : 0.0F;
            mask <<= 1;
        }

        const Eigen::Vector<float, neural_network_input_size::value> input {
            letter_binary_encoding.cat(_encoding_result)};

        _encoding_result = _word_vector_encoder_nn.compute(input);

        return *this;
    }
} // namespace efuzz

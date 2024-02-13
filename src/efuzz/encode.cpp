#include <Eigen/Core>

#include <efuzz/encode.hpp>

namespace efuzz {
    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::encode(const StringT& word)
        -> encoding_result_type {
        reset_encoding_result();

        for (const auto& letter: word) {
            encode_letter(letter);
        }

        return _encoding_result;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::encode_letter(const char_type& letter)
        -> this_type& {
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

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::reset_encoding_result() -> this_type& {
        _encoding_result.setZero();

        return *this;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::get_encoding_result() const
        -> encoding_result_type& {
        return _encoding_result;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::set_word_vector_encoder_nn(
        const NeuralNetwork& neural_network) -> this_type& {
        _word_vector_encoder_nn = neural_network;

        return *this;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::get_word_vector_encoder_nn() const
        -> NeuralNetwork {
        return _word_vector_encoder_nn;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::set_encoding_nn_layer_sizes(
        const std::vector<std::size_t>& layer_sizes, bool random) -> this_type& {
        assert(layer_sizes.front() == get_nn_input_size());
        assert(layer_sizes.back() == get_nn_output_size());

        _word_vector_encoder_nn = NeuralNetwork(layer_sizes, random);

        return *this;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::get_nn_input_size() const -> std::size_t {
        if constexpr (encoding_result_size_is_dynamic::value) {
            return char_encoder_size::value + encoding_result_size;
        }

        assert(_encoding_result_size.has_value());

        return char_encoder_size::value + _encoding_result_size;
    }

    template <typename StringT, int encoding_result_size>
    requires requires { typename StringT::value_type; }

    auto Encoder<StringT, encoding_result_size>::get_nn_output_size() const -> std::size_t {
        if constexpr (encoding_result_size_is_dynamic::value) {
            return encoding_result_size;
        }

        assert(_encoding_result_size.has_value());

        return _encoding_result_size.value();
    }
} // namespace efuzz

#ifndef EFUZZ_ENCODE_HPP
#define EFUZZ_ENCODE_HPP

#include <cmath>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <vector>

#include <cereal/cereal.hpp>
#include <Eigen/Core>

#include <efuzz/cereal_eigen.hpp>
#include <efuzz/neural_network/neural_network.hpp>

namespace efuzz {
    template <typename StringT_, int encoding_result_size = -1>
    requires requires { typename StringT_::value_type; }

    class Encoder {
        public:

        using StringT = StringT_;
        using char_type = typename StringT::value_type;
        using this_type = Encoder<StringT, encoding_result_size>;
        using char_encoder_size = std::integral_constant<std::size_t, sizeof(char_type) * 8>;
        using encoding_result_size_is_dynamic =
            std::bool_constant<std::greater()(encoding_result_size, 0)>;
        using encoding_result_type =
            std::conditional_t<encoding_result_size_is_dynamic::value,
                               Eigen::Vector<float, encoding_result_size>, Eigen::VectorXf>;
        using neural_network_input_type = std::conditional_t<
            encoding_result_size_is_dynamic::value,
            Eigen::Vector<float, char_encoder_size::value + encoding_result_size>, Eigen::VectorXf>;
        using neural_network_output_type = encoding_result_type;

        Encoder() = default;
        Encoder(const Encoder&) = default;
        Encoder(Encoder&&) = default;
        this_type& operator=(const this_type&) = default;
        this_type& operator=(this_type&&) = default;

        template <typename Archive>
        void serialize(Archive& archive) {
            archive(_word_vector_encoder_nn);
        }

        encoding_result_type encode(const StringT& word);
        this_type& encode_letter(const char_type& letter);
        this_type& reset_encoding_result();
        [[nodiscard]] encoding_result_type get_encoding_result() const;

        this_type& set_word_vector_encoder_nn(const NeuralNetwork& neural_network);
        [[nodiscard]] NeuralNetwork get_word_vector_encoder_nn() const;
        this_type& set_encoding_nn_layer_sizes(const std::vector<std::size_t>& layer_sizes,
                                               bool random = true);

        [[nodiscard]] std::size_t get_nn_input_size() const;
        [[nodiscard]] std::size_t get_nn_output_size() const;
        [[nodiscard]] constexpr float output_norm_max() const
            requires encoding_result_size_is_dynamic::value;
        [[nodiscard]] float output_norm_max() const
            requires(!encoding_result_size_is_dynamic::value);

        private:

        using neural_network_input_size = std::conditional_t<
            encoding_result_size_is_dynamic::value,
            std::integral_constant<std::size_t, char_encoder_size::value + encoding_result_size>,
            std::nullptr_t>;

        // using neural_network_output_size = std::integral_constant<std::size_t, >;

        using neural_network_output_size = std::conditional_t<
            encoding_result_size_is_dynamic::value,
            std::integral_constant<std::size_t, static_cast<std::size_t>(encoding_result_size)>,
            std::nullptr_t>;

        NeuralNetwork _word_vector_encoder_nn; // Recurrent Neural Network (RNN)
        encoding_result_type _encoding_result;
        std::optional<std::size_t> _encoding_result_size;
    };

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::encode(const StringT& word)
        -> encoding_result_type {
        reset_encoding_result();

        for (const auto& letter: word) {
            encode_letter(letter);
        }

        return _encoding_result;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::encode_letter(const char_type& letter)
        -> this_type& {
        Eigen::Vector<float, char_encoder_size::value> letter_binary_encoding;

        char_type mask = 1;

        for (std::size_t i = 0; i < char_encoder_size::value; ++i) {
            letter_binary_encoding(i) = (letter & mask) ? 1.0F : 0.0F;
            mask <<= 1;
        }

        neural_network_input_type input;

        input << letter_binary_encoding, _encoding_result;

        _encoding_result = _word_vector_encoder_nn.compute(input);

        return *this;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::reset_encoding_result() -> this_type& {
        _encoding_result.setZero();

        return *this;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::get_encoding_result() const
        -> encoding_result_type {
        return _encoding_result;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::set_word_vector_encoder_nn(
        const NeuralNetwork& neural_network) -> this_type& {
        _word_vector_encoder_nn = neural_network;

        return *this;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::get_word_vector_encoder_nn() const
        -> NeuralNetwork {
        return _word_vector_encoder_nn;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::set_encoding_nn_layer_sizes(
        const std::vector<std::size_t>& layer_sizes, bool random) -> this_type& {
        assert(layer_sizes.front() == get_nn_input_size());
        assert(layer_sizes.back() == get_nn_output_size());

        _word_vector_encoder_nn = NeuralNetwork(layer_sizes, random);

        return *this;
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::get_nn_input_size() const -> std::size_t {
        if constexpr (encoding_result_size_is_dynamic::value) {
            return char_encoder_size::value + encoding_result_size;
        }

        assert(_encoding_result_size.has_value());

        return char_encoder_size::value + _encoding_result_size.value();
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::get_nn_output_size() const -> std::size_t {
        if constexpr (encoding_result_size_is_dynamic::value) {
            return encoding_result_size;
        }

        assert(_encoding_result_size.has_value());

        return _encoding_result_size.value();
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    constexpr auto Encoder<StringT_, encoding_result_size>::output_norm_max() const -> float
        requires encoding_result_size_is_dynamic::value {
        return std::sqrt(static_cast<float>(encoding_result_size));
    }

    template <typename StringT_, int encoding_result_size>
    requires requires { typename StringT_::value_type; }

    auto Encoder<StringT_, encoding_result_size>::output_norm_max() const -> float
        requires(!encoding_result_size_is_dynamic::value) {
        assert(_encoding_result_size.has_value());

        return std::sqrt(static_cast<float>(_encoding_result_size.value()));
    }
} // namespace efuzz

#endif // EFUZZ_ENCODE_HPP

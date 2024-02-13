#ifndef EFUZZ_ENCODE_HPP
#define EFUZZ_ENCODE_HPP

#include <cstddef>
#include <functional>
#include <type_traits>

#include <cereal/cereal.hpp>
#include <Eigen/Core>

#include <efuzz/cereal_eigen.hpp>
#include <efuzz/neural_network/neural_network.hpp>

namespace efuzz {
    template <typename StringT, int encoding_result_size = -1>
    requires requires { typename StringT::value_type; }

    class Encoder {
        public:

        using char_type = typename StringT::value_type;
        using this_type = Encoder<StringT, encoding_result_size>;
        using char_encoder_size = std::integral_constant<std::size_t, sizeof(char_type) * 8>;
        using encoding_result_size_is_dynamic =
            std::bool_constant<std::greater()(encoding_result_size, 0)>;
        using encoding_result_type =
            std::conditional_t<encoding_result_size_is_dynamic::value,
                               Eigen::Vector<float, encoding_result_size>, Eigen::VectorXf>;

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
        [[nodiscard]] encoding_result_type& get_encoding_result() const;

        this_type& set_word_vector_encoder_nn(const NeuralNetwork& neural_network);
        [[nodiscard]] NeuralNetwork get_word_vector_encoder_nn() const;

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
    };

    template class Encoder<std::string>;
    template class Encoder<std::wstring>;
    template class Encoder<std::u8string>;
    template class Encoder<std::u16string>;
    template class Encoder<std::u32string>;
} // namespace efuzz

#endif // EFUZZ_ENCODE_HPP

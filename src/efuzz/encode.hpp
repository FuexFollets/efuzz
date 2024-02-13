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
        using this_type = Encoder<StringT>;
        using char_encoder_size = std::integral_constant<std::size_t, sizeof(char_type) * 8>;
        using encoding_result_type = Eigen::Vector<float, char_encoder_size::value>;

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
        [[nodiscard]] this_type& get_encoding_result(encoding_result_type& encoding_result) const;

        this_type& set_word_vector_encoder_nn(const NeuralNetwork& neural_network);
        [[nodiscard]] NeuralNetwork get_word_vector_encoder_nn() const;

        private:

        using neural_network_input_size =
            std::integral_constant<std::size_t,
                                   char_encoder_size::value + encoding_result_type::value>;

        using neural_network_output_size =
            std::integral_constant<std::size_t, encoding_result_type::value>;

        NeuralNetwork _word_vector_encoder_nn; // Recurrent Neural Network (RNN)
        encoding_result_type _encoding_result;
    };
} // namespace efuzz

#endif // EFUZZ_ENCODE_HPP

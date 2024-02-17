#ifndef EFUZZ_TRAIN_ENCODER_HPP
#define EFUZZ_TRAIN_ENCODER_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <type_traits>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <rapidfuzz/fuzz.hpp>

#include <efuzz/encode.hpp>
#include <efuzz/neural_network/neural_network.hpp>

template <template <typename...> class Template, typename T>
struct is_instantiation_of : std::false_type {};

template <template <typename...> class Template, typename... Args>
struct is_instantiation_of<Template, Template<Args...>> : std::true_type {};

namespace efuzz {
    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value class TrainEncoder {
        public:

        using this_type = TrainEncoder<EncoderT>;
        using StringT = typename EncoderT::StringT;
        using DatasetT = std::shared_ptr<std::vector<StringT>>;

        TrainEncoder() = default;
        TrainEncoder(const TrainEncoder&) = default;
        TrainEncoder(TrainEncoder&&) = default;
        TrainEncoder& operator=(const TrainEncoder&) = default;
        TrainEncoder& operator=(TrainEncoder&&) = default;

        explicit TrainEncoder(EncoderT encoder);
        explicit TrainEncoder(EncoderT encoder, DatasetT dataset);

        template <typename Archive>
        void serialize(Archive& archive) {
            archive(_encoder, _dataset);
        }

        void set_dataset(DatasetT dataset);
        void add_to_dataset(const StringT& string);
        void add_to_dataset(const std::vector<StringT>& strings);
        EncoderT get_encoder() const;
        DatasetT get_dataset() const;

        [[nodiscard]] float cost(const StringT& string_1, const StringT& string_2) const;
        NeuralNetwork::NeuralNetworkDiff train_random(std::size_t iterations);
        NeuralNetwork::NeuralNetworkDiff train_all(std::size_t iterations);

        this_type& modify_encoder(const NeuralNetwork::NeuralNetworkDiff& diff);

        private:

        EncoderT _encoder;
        std::optional<DatasetT> _dataset;
    };

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    TrainEncoder<EncoderT>::TrainEncoder(EncoderT encoder) : _encoder(encoder) {
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    TrainEncoder<EncoderT>::TrainEncoder(EncoderT encoder, DatasetT dataset) :
        _encoder(encoder), _dataset(dataset) {
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    void TrainEncoder<EncoderT>::set_dataset(DatasetT dataset) {
        _dataset = dataset;
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    void TrainEncoder<EncoderT>::add_to_dataset(
        const typename TrainEncoder<EncoderT>::StringT& string) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->push_back(string);
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    void TrainEncoder<EncoderT>::add_to_dataset(const std::vector<StringT>& strings) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->insert(_dataset->end(), strings.begin(), strings.end());
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    EncoderT TrainEncoder<EncoderT>::get_encoder() const {
        return _encoder;
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    typename TrainEncoder<EncoderT>::DatasetT TrainEncoder<EncoderT>::get_dataset() const {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }

        return _dataset;
    }

    template <typename EncoderT>
    requires is_instantiation_of<Encoder, EncoderT>::value
    float TrainEncoder<EncoderT>::cost(const StringT& string_1, const StringT& string_2) const {
        const auto encoded_1 = _encoder.encode(string_1);
        const auto encoded_2 = _encoder.encode(string_2);
        const float max_normalized_difference = _encoder.output_norm_max();
        const float encoded_normalized_difference =
            (encoded_1 - encoded_2).norm() / max_normalized_difference;

        constexpr float max_rapidfuzz_difference = 100.0F;
        const float rapidfuzz_difference =
            rapidfuzz::fuzz::ratio(string_1, string_2) / max_rapidfuzz_difference;

        return std::abs(encoded_normalized_difference - rapidfuzz_difference);
    }
} // namespace efuzz

#endif // EFUZZ_TRAIN_ENCODER_HPP

#ifndef EFUZZ_TRAIN_ENCODER_HPP
#define EFUZZ_TRAIN_ENCODER_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <rapidfuzz/fuzz.hpp>

#include <efuzz/encode.hpp>
#include <efuzz/neural_network/neural_network.hpp>

namespace efuzz {
    template <typename EncoderT>
    class TrainEncoder {
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

        NeuralNetwork::NeuralNetworkDiff train_random(std::size_t iterations);
        NeuralNetwork::NeuralNetworkDiff train_all(std::size_t iterations);

        this_type& modify_encoder(const NeuralNetwork::NeuralNetworkDiff& diff);

        private:

        EncoderT _encoder;
        std::optional<DatasetT> _dataset;
    };

    template <typename EncoderT>
    TrainEncoder<EncoderT>::TrainEncoder(EncoderT encoder) : _encoder(encoder) {
    }

    template <typename EncoderT>
    TrainEncoder<EncoderT>::TrainEncoder(EncoderT encoder, DatasetT dataset) :
        _encoder(encoder), _dataset(dataset) {
    }

    template <typename EncoderT>
    void TrainEncoder<EncoderT>::set_dataset(DatasetT dataset) {
        _dataset = dataset;
    }

    template <typename EncoderT>
    void TrainEncoder<EncoderT>::add_to_dataset(
        const typename TrainEncoder<EncoderT>::StringT& string) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->push_back(string);
    }

    template <typename EncoderT>
    void TrainEncoder<EncoderT>::add_to_dataset(const std::vector<StringT>& strings) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->insert(_dataset->end(), strings.begin(), strings.end());
    }

    template <typename EncoderT>
    EncoderT TrainEncoder<EncoderT>::get_encoder() const {
        return _encoder;
    }

    template <typename EncoderT>
    typename TrainEncoder<EncoderT>::DatasetT TrainEncoder<EncoderT>::get_dataset() const {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }

        return _dataset;
    }
} // namespace efuzz

#endif // EFUZZ_TRAIN_ENCODER_HPP

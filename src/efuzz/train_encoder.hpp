#ifndef EFUZZ_TRAIN_ENCODER_HPP
#define EFUZZ_TRAIN_ENCODER_HPP

#include <cstddef>
#include <memory>
#include <optional>
#include <random>
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
    template <typename StringT_>
    class TrainEncoder {
        public:

        using this_type = TrainEncoder<StringT_>;
        using StringT = StringT_;
        using DatasetT = std::shared_ptr<std::vector<StringT>>;

        TrainEncoder() = default;
        TrainEncoder(const TrainEncoder&) = default;
        TrainEncoder(TrainEncoder&&) = default;
        TrainEncoder& operator=(const TrainEncoder&) = default;
        TrainEncoder& operator=(TrainEncoder&&) = default;

        explicit TrainEncoder(Encoder<StringT> encoder);
        explicit TrainEncoder(Encoder<StringT> encoder, DatasetT dataset);

        template <typename Archive>
        void serialize(Archive& archive) {
            archive(_encoder, _dataset);
        }

        void set_dataset(DatasetT dataset);
        void add_to_dataset(const StringT& string);
        void add_to_dataset(const std::vector<StringT>& strings);
        Encoder<StringT> get_encoder() const;
        DatasetT get_dataset() const;

        [[nodiscard]] float cost(const StringT& string_1, const StringT& string_2) const;

        struct TrainingResult {
            std::optional<NeuralNetwork::NeuralNetworkDiff> diff;
            float original_cost {};
            float modified_cost {};

            template <typename Archive>
            void serialize(Archive& archive) {
                archive(diff, original_cost, modified_cost);
            }
        };

        TrainingResult train(const StringT& string_1, const StringT& string_2);
        TrainingResult train(const std::vector<std::pair<StringT, StringT>>& string_pairs);
        TrainingResult train_random(std::size_t iterations);
        TrainingResult train_all(std::size_t iterations);

        this_type& modify_encoder(const NeuralNetwork::NeuralNetworkDiff& diff);

        private:

        Encoder<StringT> _encoder;
        std::optional<DatasetT> _dataset;
    };

    template <typename StringT_>
    TrainEncoder<StringT_>::TrainEncoder(Encoder<StringT> encoder) : _encoder(encoder) {
    }

    template <typename StringT_>
    TrainEncoder<StringT_>::TrainEncoder(Encoder<StringT> encoder, DatasetT dataset) :
        _encoder(encoder), _dataset(dataset) {
    }

    template <typename StringT_>
    void TrainEncoder<StringT_>::set_dataset(DatasetT dataset) {
        _dataset = dataset;
    }

    template <typename StringT_>
    void TrainEncoder<StringT_>::add_to_dataset(
        const typename TrainEncoder<StringT_>::StringT& string) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->push_back(string);
    }

    template <typename StringT_>
    void TrainEncoder<StringT_>::add_to_dataset(const std::vector<StringT>& strings) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->insert(_dataset->end(), strings.begin(), strings.end());
    }

    template <typename StringT_>
    Encoder<StringT_> TrainEncoder<StringT_>::get_encoder() const {
        return _encoder;
    }

    template <typename StringT_>
    typename TrainEncoder<StringT_>::DatasetT TrainEncoder<StringT_>::get_dataset() const {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }

        return _dataset;
    }

    template <typename StringT_>
    float TrainEncoder<StringT_>::cost(const StringT& string_1, const StringT& string_2) const {
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

    template <typename StringT_>
    TrainEncoder<StringT_>::TrainingResult TrainEncoder<StringT_>::train(const StringT& string_1,
                                                                         const StringT& string_2) {
        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();
        const float original_cost = cost(string_1, string_2);

        const NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();

        _encoder.get_word_vector_encoder_nn().modify(diff);

        const float modified_cost = cost(string_1, string_2);

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        if (modified_cost < original_cost) {
            return TrainingResult {
                .diff = diff, .original_cost = original_cost, .modified_cost = modified_cost};
        }

        return TrainingResult {.original_cost = original_cost, .modified_cost = modified_cost};
    }

    template <typename StringT_>
    TrainEncoder<StringT_>::TrainingResult TrainEncoder<StringT_>::train(
        const std::vector<std::pair<StringT, StringT>>& string_pairs) {
        if (string_pairs.empty()) {
            throw std::runtime_error("Empty string pairs provided");
        }

        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();

        float average_cost_of_unmodified_encoder {};

        for (const auto& [string_1, string_2]: string_pairs) {
            const float cost = TrainEncoder<StringT_>::cost(string_1, string_2);

            average_cost_of_unmodified_encoder += cost;
        }

        average_cost_of_unmodified_encoder /= string_pairs.size();

        NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();

        _encoder.get_word_vector_encoder_nn().modify(diff);

        float average_cost_of_modified_encoder {};

        for (const auto& [string_1, string_2]: string_pairs) {
            const float cost = TrainEncoder<StringT_>::cost(string_1, string_2);

            average_cost_of_modified_encoder += cost;
        }

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        if (average_cost_of_modified_encoder < average_cost_of_unmodified_encoder) {
            return TrainingResult {.diff = diff,
                                   .original_cost = average_cost_of_unmodified_encoder,
                                   .modified_cost = average_cost_of_modified_encoder};
        }

        return TrainingResult {.original_cost = average_cost_of_unmodified_encoder,
                               .modified_cost = average_cost_of_modified_encoder};
    }

    template <typename StringT_>
    TrainEncoder<StringT_>::TrainingResult
        TrainEncoder<StringT_>::train_random(std::size_t iterations) {
        if (!_dataset) {
            throw std::runtime_error("No dataset provided");
        }

        if (_dataset->empty()) {
            throw std::runtime_error("Empty dataset provided");
        }

        const std::size_t dataset_size = _dataset->size();

        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();
        std::random_device random_device;
        std::mt19937 random(random_device());

        float average_cost_of_unmodified_encoder {};

        for (std::size_t iteration {}; iteration < iterations; ++iteration) {
            const NeuralNetwork::NeuralNetworkDiff random_diff =
                _encoder.get_word_vector_encoder_nn().random_diff();
            const StringT& string_1 = _dataset->at(random() % dataset_size);
            const StringT& string_2 = _dataset->at(random() % dataset_size);

            const float cost = cost(string_1, string_2);

            average_cost_of_unmodified_encoder += cost;
        }

        average_cost_of_unmodified_encoder /= iterations;

        NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();
        _encoder.get_word_vector_encoder_nn().modify(diff);

        float average_cost_of_modified_encoder {};

        for (std::size_t iteration {}; iteration < iterations; ++iteration) {
            const StringT& string_1 = _dataset->at(random() % dataset_size);
            const StringT& string_2 = _dataset->at(random() % dataset_size);

            const float cost = cost(string_1, string_2);

            average_cost_of_modified_encoder += cost;
        }

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        if (average_cost_of_unmodified_encoder < average_cost_of_modified_encoder) {
            return TrainingResult {.diff = diff,
                                   .original_cost = average_cost_of_unmodified_encoder,
                                   .modified_cost = average_cost_of_modified_encoder};
        }
    }
} // namespace efuzz

#endif // EFUZZ_TRAIN_ENCODER_HPP

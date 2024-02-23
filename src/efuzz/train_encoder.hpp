#ifndef EFUZZ_TRAIN_ENCODER_HPP
#define EFUZZ_TRAIN_ENCODER_HPP

#include <cstddef>
#include <random>

#include <cereal/cereal.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <rapidfuzz/fuzz.hpp>

#include <efuzz/encode.hpp>
#include <efuzz/neural_network/neural_network.hpp>

namespace efuzz {
    template <StdString StringT_,
              IntegralConstant encoding_result_size_ = std::integral_constant<int, -1>>
    class EncoderTrainer {
        public:

        struct CostLogDatapoint {
            std::size_t iteration {};
            float original_cost {};
            float modified_cost {};

            template <typename Archive>
            void serialize(Archive& archive) {
                archive(iteration, original_cost, modified_cost);
            }
        };

        using this_type = EncoderTrainer<StringT_, encoding_result_size_>;
        using StringT = StringT_;
        using DatasetT = std::shared_ptr<std::vector<StringT>>;
        using EncoderT = Encoder<StringT, encoding_result_size_>;

        EncoderTrainer() = default;
        EncoderTrainer(const EncoderTrainer&) = default;
        EncoderTrainer(EncoderTrainer&&) = default;
        EncoderTrainer& operator=(const EncoderTrainer&) = default;
        EncoderTrainer& operator=(EncoderTrainer&&) = default;

        explicit EncoderTrainer(EncoderT encoder);
        explicit EncoderTrainer(EncoderT encoder, DatasetT dataset);

        template <typename Archive>
        void serialize(Archive& archive) {
            archive(_encoder, _dataset, _training_iterations, _cost_log);
        }

        void set_dataset(DatasetT dataset);
        void add_to_dataset(const StringT& string, bool reset_training_iterations = false);
        void add_to_dataset(const std::vector<StringT>& strings,
                            bool reset_training_iterations = false);
        EncoderT get_encoder() const;
        DatasetT get_dataset() const;
        [[nodiscard]] std::size_t get_training_iterations() const;
        [[nodiscard]] std::vector<CostLogDatapoint> get_cost_log() const;
        this_type& clear_cost_log();

        [[nodiscard]] float cost(const StringT& string_1, const StringT& string_2);

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
        TrainingResult train_all();

        this_type& modify_encoder(const NeuralNetwork::NeuralNetworkDiff& diff);
        bool apply_training_result(const TrainingResult& training_result);

        private:

        EncoderT _encoder;
        std::optional<DatasetT> _dataset;
        std::size_t _training_iterations {};
        std::vector<CostLogDatapoint> _cost_log;
    };

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    EncoderTrainer<StringT_, encoding_result_size_>::EncoderTrainer(EncoderT encoder) :
        _encoder(encoder) {
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    EncoderTrainer<StringT_, encoding_result_size_>::EncoderTrainer(EncoderT encoder,
                                                                    DatasetT dataset) :
        _encoder(encoder),
        _dataset(dataset) {
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    void EncoderTrainer<StringT_, encoding_result_size_>::set_dataset(DatasetT dataset) {
        _dataset = dataset;
        _training_iterations = 0;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    void EncoderTrainer<StringT_, encoding_result_size_>::add_to_dataset(
        const StringT& string, bool reset_training_iterations) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }
        _dataset->push_back(string);

        if (reset_training_iterations) {
            _training_iterations = 0;
        }
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    void EncoderTrainer<StringT_, encoding_result_size_>::add_to_dataset(
        const std::vector<StringT>& strings, bool reset_training_iterations) {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }

        _dataset->insert(_dataset->end(), strings.begin(), strings.end());

        if (reset_training_iterations) {
            _training_iterations = 0;
        }
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    Encoder<StringT_, encoding_result_size_>
        EncoderTrainer<StringT_, encoding_result_size_>::get_encoder() const {
        return _encoder;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    typename EncoderTrainer<StringT_, encoding_result_size_>::DatasetT
        EncoderTrainer<StringT_, encoding_result_size_>::get_dataset() const {
        if (!_dataset) {
            _dataset = std::make_shared<std::vector<StringT>>();
        }

        return _dataset;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    std::size_t EncoderTrainer<StringT_, encoding_result_size_>::get_training_iterations() const {
        return _training_iterations;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    std::vector<typename EncoderTrainer<StringT_, encoding_result_size_>::CostLogDatapoint>
        EncoderTrainer<StringT_, encoding_result_size_>::get_cost_log() const {
        return _cost_log;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    auto EncoderTrainer<StringT_, encoding_result_size_>::clear_cost_log() -> this_type& {
        _cost_log.clear();

        return *this;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    float EncoderTrainer<StringT_, encoding_result_size_>::cost(const StringT& string_1,
                                                                const StringT& string_2) {
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

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    typename EncoderTrainer<StringT_, encoding_result_size_>::TrainingResult
        EncoderTrainer<StringT_, encoding_result_size_>::train(const StringT& string_1,
                                                               const StringT& string_2) {
        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();
        const float original_cost = cost(string_1, string_2);

        const NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();

        modify_encoder(diff);

        const float modified_cost = cost(string_1, string_2);

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        if (modified_cost < original_cost) {
            return TrainingResult {
                .diff = diff, .original_cost = original_cost, .modified_cost = modified_cost};
        }

        return TrainingResult {.original_cost = original_cost, .modified_cost = modified_cost};
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    typename EncoderTrainer<StringT_, encoding_result_size_>::TrainingResult
        EncoderTrainer<StringT_, encoding_result_size_>::train(
            const std::vector<std::pair<StringT, StringT>>& string_pairs) { // Non-wrapped
        if (string_pairs.empty()) {
            throw std::runtime_error("Empty string pairs provided");
        }
        _training_iterations++;

        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();

        float average_cost_of_unmodified_encoder {};

        for (const auto& [string_1, string_2]: string_pairs) {
            const float cost =
                EncoderTrainer<StringT_, encoding_result_size_>::cost(string_1, string_2);

            average_cost_of_unmodified_encoder += cost;
        }

        average_cost_of_unmodified_encoder /= string_pairs.size();

        NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();

        modify_encoder(diff);

        float average_cost_of_modified_encoder {};

        for (const auto& [string_1, string_2]: string_pairs) {
            const float cost =
                EncoderTrainer<StringT_, encoding_result_size_>::cost(string_1, string_2);

            average_cost_of_modified_encoder += cost;
        }

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        average_cost_of_modified_encoder /= string_pairs.size();

        if (average_cost_of_modified_encoder < average_cost_of_unmodified_encoder) {
            return TrainingResult {.diff = diff,
                                   .original_cost = average_cost_of_unmodified_encoder,
                                   .modified_cost = average_cost_of_modified_encoder};
        }

        return TrainingResult {.original_cost = average_cost_of_unmodified_encoder,
                               .modified_cost = average_cost_of_modified_encoder};
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    typename EncoderTrainer<StringT_, encoding_result_size_>::TrainingResult
        EncoderTrainer<StringT_, encoding_result_size_>::train_random(std::size_t iterations) {
        if (!_dataset) {
            throw std::runtime_error("No dataset provided");
        }

        const std::size_t dataset_size = _dataset.value()->size();

        if (dataset_size < 2) {
            throw std::runtime_error("Dataset too small");
        }

        std::vector<std::pair<StringT, StringT>> string_pairs;

        std::random_device random_device;
        std::mt19937 random_engine(random_device());
        std::uniform_int_distribution<std::size_t> distribution(0, dataset_size - 1);

        for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
            const std::size_t index_1 = distribution(random_engine);
            const std::size_t index_2 = distribution(random_engine);

            if (index_1 == index_2) {
                continue;
            }

            string_pairs.emplace_back((*_dataset.value()) [index_1], (*_dataset.value()) [index_2]);
        }

        return train(string_pairs);
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    typename EncoderTrainer<StringT_, encoding_result_size_>::TrainingResult
        EncoderTrainer<StringT_, encoding_result_size_>::train_all() { // Non-wrapped
        if (!_dataset) {
            throw std::runtime_error("No dataset provided");
        }

        std::size_t dataset_size = _dataset.value()->size();

        if (dataset_size < 2) {
            throw std::runtime_error("Dataset too small");
        }

        if (_encoder.get_word_vector_encoder_nn().layer_sizes.empty()) {
            throw std::runtime_error(
                "No neural network layer sizes set. Try encoder.set_encoding_nn_layer_sizes() or "
                "encoder.set_word_vector_encoder_nn()");
        }

        _training_iterations++;

        // Dont create string pairs, that takes too much memory

        const NeuralNetwork original_encoder_nn = _encoder.get_word_vector_encoder_nn();

        float average_cost_of_unmodified_encoder {};

        for (std::size_t indexer_1 = 0; indexer_1 < dataset_size; ++indexer_1) {
            for (std::size_t indexer_2 = 0; indexer_2 < dataset_size; ++indexer_2) {
                if (indexer_1 == indexer_2) {
                    continue;
                }

                const float cost = EncoderTrainer<StringT_, encoding_result_size_>::cost(
                    (*_dataset.value()) [indexer_1], (*_dataset.value()) [indexer_2]);

                average_cost_of_unmodified_encoder += cost;
            }
        }

        const std::size_t comparisons = dataset_size * (dataset_size - 1);

        average_cost_of_unmodified_encoder /= comparisons;

        NeuralNetwork::NeuralNetworkDiff diff = original_encoder_nn.random_diff();

        modify_encoder(diff);

        float average_cost_of_modified_encoder {};

        for (std::size_t indexer_1 = 0; indexer_1 < dataset_size; ++indexer_1) {
            for (std::size_t indexer_2 = 0; indexer_2 < dataset_size; ++indexer_2) {
                if (indexer_1 == indexer_2) {
                    continue;
                }

                const float cost = EncoderTrainer<StringT_, encoding_result_size_>::cost(
                    (*_dataset.value()) [indexer_1], (*_dataset.value()) [indexer_2]);

                average_cost_of_modified_encoder += cost;
            }
        }

        _encoder.set_word_vector_encoder_nn(original_encoder_nn);

        average_cost_of_modified_encoder /= comparisons;

        if (average_cost_of_modified_encoder < average_cost_of_unmodified_encoder) {
            return TrainingResult {.diff = diff,
                                   .original_cost = average_cost_of_unmodified_encoder,
                                   .modified_cost = average_cost_of_modified_encoder};
        }

        return TrainingResult {.original_cost = average_cost_of_unmodified_encoder,
                               .modified_cost = average_cost_of_modified_encoder};
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    auto EncoderTrainer<StringT_, encoding_result_size_>::modify_encoder(
        const NeuralNetwork::NeuralNetworkDiff& diff) -> this_type& {
        _encoder.modify_word_vector_encoder_nn(diff);

        return *this;
    }

    template <StdString StringT_, IntegralConstant encoding_result_size_>
    bool EncoderTrainer<StringT_, encoding_result_size_>::apply_training_result(
        const TrainingResult& training_result) {
        if (training_result.diff && training_result.modified_cost < training_result.original_cost) {
            _encoder.modify_word_vector_encoder_nn(training_result.diff.value());

            return true;
        }

        return false;
    }
} // namespace efuzz

#endif // EFUZZ_TRAIN_ENCODER_HPP

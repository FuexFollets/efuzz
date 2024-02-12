#ifndef LEXOCRAFT_NEURAL_NETWORK_HPP
#define LEXOCRAFT_NEURAL_NETWORK_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <Eigen/Core>

#include <efuzz/cereal_eigen.hpp>

namespace lc {
    using vbuffer_t = std::vector<std::uint8_t>;

    class NeuralNetwork {
        public:

        class NeuralNetworkDiff {
            public:

            constexpr static std::size_t FIELD_COUNT {3};
            std::vector<Eigen::MatrixXf> weight_diffs;
            std::vector<Eigen::VectorXf> bias_diffs;
            std::vector<std::size_t> layer_sizes;

            NeuralNetworkDiff() = default;
            NeuralNetworkDiff(NeuralNetworkDiff&& other) noexcept = default;
            NeuralNetworkDiff(const NeuralNetworkDiff& other) noexcept = default;
            explicit NeuralNetworkDiff(const std::vector<std::size_t>& layer_sizes);

            NeuralNetworkDiff& operator=(NeuralNetworkDiff&& other) noexcept = default;
            NeuralNetworkDiff& operator=(const NeuralNetworkDiff& other) noexcept = default;

            NeuralNetworkDiff& operator+=(const NeuralNetworkDiff& other) noexcept;
            NeuralNetworkDiff& operator-=(const NeuralNetworkDiff& other) noexcept;
            NeuralNetworkDiff& operator*=(float scalar) noexcept;
            NeuralNetworkDiff& operator/=(float scalar);

            NeuralNetworkDiff operator+(const NeuralNetworkDiff& other) const noexcept;
            NeuralNetworkDiff operator-(const NeuralNetworkDiff& other) const noexcept;
            NeuralNetworkDiff operator*(float scalar) const noexcept;
            NeuralNetworkDiff operator/(float scalar) const;

            void invert() noexcept;

            [[nodiscard]] NeuralNetworkDiff inverted() const noexcept;

            friend class NeuralNetwork;

            template <class Archive>
            void serialize(Archive& archive) {
                archive(weight_diffs, bias_diffs, layer_sizes);
            }
        };

        constexpr static float GOOD_COST {0.1F};
        constexpr static std::size_t FIELD_COUNT {7};

        std::size_t iterations {};
        std::vector<std::size_t> layer_sizes;

        std::vector<Eigen::MatrixXf> weights;
        std::vector<Eigen::VectorXf> biases;

        NeuralNetworkDiff most_recent_diff;
        float most_recent_cost {};
        std::size_t diff_improvement_streak {};

        NeuralNetwork() = default;
        NeuralNetwork(NeuralNetwork&& other) noexcept = default;
        NeuralNetwork(const NeuralNetwork& other) noexcept = default;

        NeuralNetwork& operator=(NeuralNetwork&& other) noexcept = default;
        NeuralNetwork& operator=(const NeuralNetwork& other) noexcept = default;

        explicit NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize = true);

        void randomize();
        void modify(NeuralNetworkDiff diff, bool apply_biases = true, bool apply_weights = true);

        void train(float cost);

        [[nodiscard]] Eigen::VectorXf compute(Eigen::VectorXf input) const noexcept;
        [[nodiscard]] NeuralNetworkDiff random_diff() const noexcept;

        void save_file(const std::filesystem::path& filepath) const;

        template <class Archive>
        void serialize(Archive& archive) {
            archive(iterations, layer_sizes, weights, biases, most_recent_diff, most_recent_cost,
                    diff_improvement_streak);
        }

        static NeuralNetwork load_file(const std::filesystem::path& filepath);
        static float sigmoid_abs(float value);
    };
} // namespace lc

std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork& network);
std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork::NeuralNetworkDiff& diff);

#endif // LEXOCRAFT_NEURAL_NETWORK_HPP

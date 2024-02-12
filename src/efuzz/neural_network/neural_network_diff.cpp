#include <algorithm>
#include <vector>

#include <Eigen/Eigen>

#include <efuzz/neural_network/neural_network.hpp>

namespace lc {
    // Give each random float values between -1 and 1.
    NeuralNetwork::NeuralNetworkDiff::NeuralNetworkDiff(
        const std::vector<std::size_t>& layer_sizes) :
        weight_diffs(std::max(std::size_t {0}, layer_sizes.size() - 1)),
        bias_diffs(std::max(std::size_t {0}, layer_sizes.size() - 1)), layer_sizes {layer_sizes} {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] =
                Eigen::MatrixXf::Random(layer_sizes [index + 1], layer_sizes [index]);
            bias_diffs [index] = Eigen::VectorXf::Random(layer_sizes [index + 1]);
        }
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator+=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] += other.weight_diffs [index];
            bias_diffs [index] += other.bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator-=(
        const NeuralNetwork::NeuralNetworkDiff& other) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] -= other.weight_diffs [index];
            bias_diffs [index] -= other.bias_diffs [index];
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff&
        NeuralNetwork::NeuralNetworkDiff::operator*=(float scalar) noexcept {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] *= scalar;
            bias_diffs [index] *= scalar;
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff& NeuralNetwork::NeuralNetworkDiff::operator/=(float scalar) {
        for (std::size_t index {0}; index < layer_sizes.size() - 1; index++) {
            weight_diffs [index] /= scalar;
            bias_diffs [index] /= scalar;
        }
        return *this;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::operator+(
        const NeuralNetwork::NeuralNetworkDiff& other) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this += other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::operator-(
        const NeuralNetwork::NeuralNetworkDiff& other) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this -= other;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::operator*(float scalar) const noexcept {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this *= scalar;

        return copy_of_this;
    }

    NeuralNetwork::NeuralNetworkDiff
        NeuralNetwork::NeuralNetworkDiff::operator/(float scalar) const {
        NeuralNetwork::NeuralNetworkDiff copy_of_this {*this};

        copy_of_this /= scalar;

        return copy_of_this;
    }

    void NeuralNetwork::NeuralNetworkDiff::invert() noexcept {
        *this *= -1;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::NeuralNetworkDiff::inverted() const noexcept {
        return *this * -1;
    }
} // namespace lc

// Print neural network diff

std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork::NeuralNetworkDiff& diff) {
    stream << "diff:\n";
    stream << "  layer_sizes: ";

    for (const auto& layer_size: diff.layer_sizes) {
        stream << layer_size << " ";
    }

    stream << "\n  weight_diffs:\n";

    for (std::size_t weight_diff_number {0}; weight_diff_number < diff.weight_diffs.size();
         weight_diff_number++) {
        const auto& weight_diff = diff.weight_diffs [weight_diff_number];

        stream << "    weight diff number " << weight_diff_number << ":\n";
        stream << weight_diff << "\n";
    }

    stream << "  bias_diffs:\n";

    for (std::size_t bias_diff_number {0}; bias_diff_number < diff.bias_diffs.size();
         bias_diff_number++) {
        const auto& bias_diff = diff.bias_diffs [bias_diff_number];

        stream << "    bias diff number " << bias_diff_number << ":\n";
        stream << bias_diff << "\n";
    }

    return stream;
}

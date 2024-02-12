#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include <cereal/cereal.hpp>

#include <efuzz/neural_network/neural_network.hpp>

namespace lc {
    NeuralNetwork::NeuralNetwork(std::vector<std::size_t> layer_sizes, bool randomize) :
        layer_sizes(layer_sizes), most_recent_diff(layer_sizes) {
        for (std::size_t index {1}; index < layer_sizes.size(); ++index) {
            weights.emplace_back(layer_sizes [index], layer_sizes [index - 1]);
            biases.emplace_back(layer_sizes [index]);
        }

        if (randomize) {
            for (auto& weight: weights) {
                weight = Eigen::MatrixXf::Random(weight.rows(), weight.cols());
            }

            for (auto& bias: biases) {
                bias = Eigen::VectorXf::Random(bias.rows());
            }
        }
    }

    Eigen::VectorXf NeuralNetwork::compute(Eigen::VectorXf input) const noexcept {
        for (std::size_t index {0}; index < weights.size(); ++index) {
            input = weights [index] * input + biases [index];
            input = input.unaryExpr([](float value) { return sigmoid_abs(value); });
        }

        return input;
    }

    NeuralNetwork::NeuralNetworkDiff NeuralNetwork::random_diff() const noexcept {
        return NeuralNetworkDiff(layer_sizes);
    }

    float NeuralNetwork::sigmoid_abs(float value) {
        return 0.5F + value / (2 * (1 + std::abs(value)));
    }

    void NeuralNetwork::randomize() {
        for (auto& weight: weights) {
            weight = Eigen::MatrixXf::Random(weight.rows(), weight.cols());
        }

        for (auto& bias: biases) {
            bias = Eigen::VectorXf::Random(bias.rows());
        }
    }

    void NeuralNetwork::modify(NeuralNetwork::NeuralNetworkDiff diff, bool apply_biases,
                               bool apply_weights) {
        if (apply_biases) {
            for (std::size_t index {0}; index < biases.size(); ++index) {
                biases [index] += diff.bias_diffs [index];
            }
        }

        if (apply_weights) {
            for (std::size_t index {0}; index < weights.size(); ++index) {
                weights [index] += diff.weight_diffs [index];
            }
        }
    }

    /* Example usage:
     repeat {
         NeuralNetwork nn {...};

         auto res = nn.compute(...);

         float cost = get_cost(res, ...);

         nn.train(cost);
     }
    */

    void NeuralNetwork::train(float cost) { /* cost is between 0 and 1 */
        // If the cost is worse than GOOD_COST or the previous cost, revert the NN using
        // _most_recent_diff, reset the streak, and apply a different NeuralNetworkDiff. Otherwise,
        // continue to apply the same NeuralNetworkDiff and increment the streak. Based on the
        // streak and most recent cost, modify the NeuralNetworkDiff and apply it each time. If
        // _iterations is zero, choose a random starting NeuralNetworkDiff.

        if (iterations == 0) {
            most_recent_diff = NeuralNetworkDiff(layer_sizes);

            return;
        }

        if (cost < GOOD_COST || cost < most_recent_cost) {
            diff_improvement_streak = 0;
            most_recent_diff.invert();
            modify(most_recent_diff);
            most_recent_diff = NeuralNetworkDiff(layer_sizes);
        }

        else {
            ++diff_improvement_streak;
            most_recent_diff *= 1.0F + diff_improvement_streak / 10.0F;
        }
    }

    void NeuralNetwork::save_file(const std::filesystem::path& filepath) const {
        std::ofstream file {filepath};

        cereal::BinaryOutputArchive oarchive {file};

        oarchive(*this);
    }

    NeuralNetwork NeuralNetwork::load_file(const std::filesystem::path& filepath) {
        std::ifstream file {filepath};

        cereal::BinaryInputArchive iarchive {file};

        NeuralNetwork network;

        iarchive(network);

        return network;
    }
} // namespace lc

std::ostream& operator<<(std::ostream& stream, const lc::NeuralNetwork& network) {
    stream << "NeuralNetwork:\n";
    stream << "  layer_sizes: ";

    for (const auto& layer_size: network.layer_sizes) {
        stream << layer_size << " ";
    }

    stream << "\n  weights:\n";

    for (std::size_t weight_number {0}; weight_number < network.weights.size(); ++weight_number) {
        const auto& weight = network.weights [weight_number];

        stream << "    weight number " << weight_number << ":\n";
        // print the weight matrix 6 spaces to the right
        stream << weight << "\n";
    }

    stream << "  biases:\n";

    for (std::size_t bias_number {0}; bias_number < network.biases.size(); ++bias_number) {
        const auto& bias = network.biases [bias_number];

        stream << "    bias number " << bias_number << ":\n";
        stream << bias << "\n";
    }

    stream << "  iterations: " << network.iterations << "\n";
    stream << "  diff_improvement_streak: " << network.diff_improvement_streak << "\n";
    stream << "  most_recent_cost: " << network.most_recent_cost << "\n";
    stream << "  most_recent_diff:\n" << network.most_recent_diff;

    return stream;
}

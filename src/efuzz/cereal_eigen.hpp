#ifndef LEXOCRAFT_CEREAL_EIGEN_HPP
#define LEXOCRAFT_CEREAL_EIGEN_HPP

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cstddef>
#include <Eigen/Dense>

namespace cereal {
    template <class Archive, class Scalar, int Rows, int Cols, int Options, int MaxRows,
              int MaxCols>
    inline void save(Archive& archive,
                     const Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix)
        requires traits::is_output_serializable<BinaryData<Scalar>, Archive>::value {
        std::size_t rows = matrix.rows();
        std::size_t cols = matrix.cols();
        archive(rows);
        archive(cols);
        archive(binary_data(matrix.data(), rows * cols * sizeof(Scalar)));
    }

    template <class Archive, class Scalar, int Rows, int Cols, int Options, int MaxRows,
              int MaxCols>
    inline void load(Archive& archive,
                     Eigen::Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& matrix)
        requires traits::is_input_serializable<BinaryData<Scalar>, Archive>::value {
        std::size_t rows = 0;
        std::size_t cols = 0;
        archive(rows);
        archive(cols);

        matrix.resize(rows, cols);

        archive(binary_data(matrix.data(), static_cast<std::size_t>(rows * cols * sizeof(Scalar))));
    }
} // namespace cereal

#endif

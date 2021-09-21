#pragma once
#include <cmath>
#include <cassert>
#include <vector>
#include <stdio.h>

namespace vul {
template <typename T>
class DynamicMatrix {
  public:
    DynamicMatrix(int num_rows, int num_columns) : rows(num_rows), columns(num_columns), m(rows * columns, 0.0) {}
    DynamicMatrix() : rows(0), columns(0), m(0) {}

    T& operator()(int row, int column) { return m[column + row * columns]; }
    const T& operator()(int row, int column) const { return m[column + row * columns]; }

    T& operator[](int index) { return m[index]; }
    const T& operator[](int index) const { return m[index]; }

    void operator+=(const DynamicMatrix<T>& matrix) {
        assert(matrix.rows == rows && matrix.columns == columns);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < columns; ++col) {
                (*this)(row, col) += matrix(row, col);
            }
        }
    }

    DynamicMatrix operator+(const DynamicMatrix<T>& matrix) const {
        auto output = *this;
        output += matrix;
        return output;
    }

    void operator-=(const DynamicMatrix<T>& matrix) {
        assert(matrix.rows == rows && matrix.columns == columns);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < columns; ++col) {
                (*this)(row, col) -= matrix(row, col);
            }
        }
    }

    DynamicMatrix operator-(const DynamicMatrix<T>& matrix) const {
        auto output = *this;
        output -= matrix;
        return output;
    }

    DynamicMatrix operator*(const DynamicMatrix<T>& matrix) const {
        assert(columns == matrix.rows);
        DynamicMatrix<T> output(rows, matrix.columns);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < matrix.columns; ++col) {
                for (int r = 0; r < matrix.rows; ++r) {
                    output(row, col) += (*this)(row, r) * matrix(r, col);
                }
            }
        }
        return output;
    }

    void operator*=(const T& scalar) {
        for (auto& v : m) v *= scalar;
    }
    DynamicMatrix operator*(const T& scalar) const {
        auto output = *this;
        output *= scalar;
        return output;
    }

    DynamicMatrix transpose() const {
        auto output = DynamicMatrix<T>(columns, rows);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < columns; ++col) {
                output(col, row) = (*this)(row, col);
            }
        }
        return output;
    }

    void print() const {
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < columns; ++c) {
                printf("%e ", (*this)(r, c));
            }
            printf("\n");
        }
    }

    void fill(const T& z) {
        for (auto& v : m) v = z;
    }
    void zero() { fill(T(0.0)); }

    T norm() const {
        T norm(0.0);
        for (const auto& v : m) norm += v * v;
        using std::sqrt;
        return sqrt(norm);
    }

    double absmax() const {
        double norm(0.0);
        for (const auto& v : m) norm = std::max(norm, std::fabs(v));
        return norm;
    }

    static DynamicMatrix<T> Identity(int num_rows) {
        DynamicMatrix<T> I(num_rows, num_rows);
        for (int row = 0; row < num_rows; ++row) {
            I(row, row) = T(1.0);
        }
        return I;
    }

    int rows;
    int columns;
    std::vector<T> m;
};

template <typename T>
DynamicMatrix<T> operator*(const T& scalar, const DynamicMatrix<T>& matrix) {
    return matrix * scalar;
}

// These are just syntactic sugar to help in linear algebra functions
template <typename T>
class DynamicRowVector : public DynamicMatrix<T> {
  public:
    using Base = DynamicMatrix<T>;
    explicit DynamicRowVector(int rows) : Base(rows, 1) {}
    DynamicRowVector<T>& operator=(const Base& matrix) {
        assert(Base::rows == matrix.rows);
        assert(Base::columns == matrix.columns);
        Base::m = matrix.m;
        return *this;
    }
};
template <typename T>
class DynamicColumnVector : public DynamicMatrix<T> {
  public:
    using Base = DynamicMatrix<T>;
    DynamicColumnVector(int columns) : Base(1, columns) {}
    DynamicColumnVector<T>& operator=(const Base& matrix) {
        assert(Base::rows == matrix.rows);
        assert(Base::columns == matrix.columns);
        Base::m = matrix.m;
        return *this;
    }
};
}

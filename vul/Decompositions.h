#pragma once
#include "DynamicMatrix.h"
#include <tuple>

namespace vul {

template <typename T>
auto backsolve(const DynamicMatrix<T>& R, const DynamicMatrix<T>& b) {
    auto x = b;
    for (int row = R.rows - 1; row >= 0; --row) {
        for (int column = row + 1; column < R.columns; ++column) {
            x[row] -= R(row, column) * x[column];
        }
        x[row] /= R(row, row);
    }
    return x;
}
template <typename T>
auto forwardsolve(const DynamicMatrix<T>& R, const DynamicMatrix<T>& b) {
    auto x = b;
    for (int row = 0; row < R.rows; ++row) {
        for (int column = 0; column < row; ++column) {
            x[row] -= R(row, column) * x[column];
        }
        x[row] /= R(row, row);
    }
    return x;
}

template <typename MatrixType>
void factorLU(MatrixType& A, int num_eqns) {
    for (int k = 1; k < num_eqns; ++k) {
        for (int i = k; i < num_eqns; ++i) {
            A(i, k - 1) /= A(k - 1, k - 1);
            for (int j = k; j < num_eqns; ++j) {
                A(i, j) -= A(i, k - 1) * A(k - 1, j);
            }
        }
    }
}

template <typename MatrixType, typename VectorType>
VectorType solveLU(MatrixType A, VectorType b, int num_eqns) {
    factorLU(A, num_eqns);

    for (int k = 1; k < num_eqns; ++k) {
        for (int i = k; i < num_eqns; ++i) {
            b[i] -= A(i, k - 1) * b[k - 1];
        }
    }
    for (int i = num_eqns - 1; i >= 0; --i) {
        if (i < num_eqns - 1) {
            for (int j = i + 1; j < num_eqns; ++j) {
                b[i] -= A(i, j) * b[j];
            }
        }
        b[i] /= A(i, i);
    }
    return b;
}

template <typename T>
std::tuple<DynamicMatrix<T>, DynamicMatrix<T>> householderDecomposition(const DynamicMatrix<T>& A) {
    int m = A.rows;
    int n = A.columns;
    DynamicMatrix<T> u(m, 1);
    DynamicMatrix<T> v(m, 1);
    auto I = DynamicMatrix<T>::Identity(m);
    auto P = I;
    auto Q = I;
    auto R = A;
    T eps(1e-10);
    for (int i = 0; i < n; ++i) {
        u.zero();
        v.zero();
        for (int j = i; j < m; ++j) {
            u[j] = R(j, i);
        }
        auto alpha = u[i] < T(0.0) ? u.norm() : -u.norm();
        for (int j = i; j < m; ++j) {
            v[j] = j == i ? u[j] + alpha : u[j];
        }
        if (v.norm() < eps) continue;

        auto v_mag = v.norm();
        for (int j = i; j < m; ++j) {
            v[j] /= v_mag;
        }

        P = I - 2.0 * v * v.transpose();

        R = P * R;
        Q = Q * P;
    }

    if (m == n) return {Q, R};

    // Resize dimensions make Q (m x n) and R (n x n)
    auto Q_out = DynamicMatrix<T>(m, n);
    for (int row = 0; row < m; ++row)
        for (int col = 0; col < n; ++col) Q_out(row, col) = Q(row, col);

    auto R_out = DynamicMatrix<T>(n, n);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col) R_out(row, col) = R(row, col);

    return {Q_out, R_out};
}

template <typename T>
DynamicMatrix<T> calcUpperTriangularInverse(const DynamicMatrix<T>& R) {
    auto m = R.rows;
    auto n = R.columns;
    auto Rinv = DynamicMatrix<T>::Identity(m);
    for (int row = m - 1; row >= 0; row--) {
        Rinv(row, row) /= R(row, row);
        for (int col = row + 1; col < n; ++col) {
            for (int c = col; c < n; ++c) {
                Rinv(row, c) -= R(row, col) / R(row, row) * Rinv(col, c);
            }
        }
    }
    return Rinv;
}

template <typename T>
DynamicMatrix<T> calcPseudoInverse(const DynamicMatrix<T>& Q, const DynamicMatrix<T>& R) {
    return calcUpperTriangularInverse(R) * Q.transpose();
}

template <typename T>
DynamicRowVector<T> solve(const DynamicMatrix<T>& A, const DynamicRowVector<T>& b) {
    DynamicRowVector<T> x(A.columns);
    if (A.rows == A.columns) {
        x = solveLU(A, b, A.rows);
    } else if (A.rows > A.columns) {
        // Overdetermined problem
        DynamicMatrix<T> Q, R;
        std::tie(Q, R) = householderDecomposition(A);
        x = backsolve(R, Q.transpose() * b);
    } else {
        // Underdetermined problem
        DynamicMatrix<T> Q, R;
        std::tie(Q, R) = householderDecomposition(A.transpose());
        x = Q * forwardsolve(R.transpose(), b);
    }
    return x;
}
}

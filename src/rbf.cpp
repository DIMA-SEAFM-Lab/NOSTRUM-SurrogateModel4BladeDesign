#include "rbf.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Legacy Gaussian-elimination solver retained as backup (currently unused).
std::vector<double> solveLinearSystem(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n = a.size();
    if (n == 0) {
        return {};
    }

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t pivot = i;
        double maxVal = std::abs(a[i][i]);
        for (std::size_t row = i + 1; row < n; ++row) {
            const double candidate = std::abs(a[row][i]);
            if (candidate > maxVal) {
                maxVal = candidate;
                pivot = row;
            }
        }

        if (maxVal < std::numeric_limits<double>::epsilon()) {
            throw std::runtime_error("Singular matrix in RBF solver");
        }

        if (pivot != i) {
            std::swap(a[pivot], a[i]);
            std::swap(b[pivot], b[i]);
        }

        const double diag = a[i][i];
        for (std::size_t col = i; col < n; ++col) {
            a[i][col] /= diag;
        }
        b[i] /= diag;

        for (std::size_t row = i + 1; row < n; ++row) {
            const double factor = a[row][i];
            if (factor == 0.0) {
                continue;
            }
            for (std::size_t col = i; col < n; ++col) {
                a[row][col] -= factor * a[i][col];
            }
            b[row] -= factor * b[i];
        }
    }

    std::vector<double> x(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = 0.0;
        for (std::size_t col = i + 1; col < n; ++col) {
            sum += a[i][col] * x[col];
        }
        x[i] = b[i] - sum;
    }

    return x;
}

extern "C" {
void dgesv_(
    int* n,
    int* nrhs,
    double* a,
    int* lda,
    int* ipiv,
    double* b,
    int* ldb,
    int* info);
}

std::vector<double> solveLinearSystemLapack(
    const std::vector<std::vector<double>>& a,
    const std::vector<double>& b) {
    const std::size_t n = a.size();
    if (n == 0) {
        return {};
    }

    for (const auto& row : a) {
        if (row.size() != n) {
            throw std::invalid_argument("Matrix must be square for LAPACK solve.");
        }
    }

    if (b.size() != n) {
        throw std::invalid_argument("RHS dimension mismatch for LAPACK solve.");
    }

    std::vector<double> aFlat(n * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            aFlat[j * n + i] = a[i][j];
        }
    }

    std::vector<double> rhs = b;
    std::vector<int> ipiv(n, 0);

    int nInt = static_cast<int>(n);
    int nrhs = 1;
    int lda = nInt;
    int ldb = nInt;
    int info = 0;

    dgesv_(&nInt, &nrhs, aFlat.data(), &lda, ipiv.data(), rhs.data(), &ldb, &info);

    if (info != 0) {
        throw std::runtime_error("LAPACK dgesv_ failed with info = " + std::to_string(info));
    }

    return rhs;
}

}  // namespace

std::vector<double> rbf(
    std::size_t nt,
    const std::vector<std::vector<double>>& xt,
    const std::vector<double>& yt,
    double eps) {
    if (nt == 0) {
        return {};
    }

    std::vector<std::vector<double>> a(nt, std::vector<double>(nt, 0.0));
    for (std::size_t i = 0; i < nt; ++i) {
        for (std::size_t j = 0; j < nt; ++j) {
            if (i == j) {
                continue;
            }

            double d2 = 0.0;
            for (std::size_t p = 0; p < xt[i].size(); ++p) {
                const double diff = xt[i][p] - xt[j][p];
                d2 += diff * diff;
            }
            const double d = std::sqrt(d2);
            a[i][j] = std::pow(d, eps);
        }
    }

    std::vector<double> rhs(nt, 0.0);
    for (std::size_t i = 0; i < nt; ++i) {
        rhs[i] = yt[i];
    }

    return solveLinearSystemLapack(a, rhs);
}

#include "compute_srbf.hpp"

#include "rbf.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <stdexcept>
#include <vector>

SRBFResult computeSRBF(
    std::size_t numParams,
    int kshift,
    const std::vector<std::vector<double>>& predictionPoints,
    int nk,
    std::size_t nt,
    const std::vector<std::vector<double>>& trainingPoints,
    const std::vector<double>& trainingValues) {
    if (nk <= 0) {
        throw std::invalid_argument("nk must be positive");
    }

    const std::size_t np = predictionPoints.size();
    std::vector<double> expectedValues(np, 0.0);
std::vector<double> variances(np, 0.0);

    int mpiRank = 0;
    int mpiSize = 1;
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (mpiInitialized) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpiRank);
        MPI_Comm_size(MPI_COMM_WORLD, &mpiSize);
    }

    const double epsMin = 1.0;
    const double epsMax = 3.0;
    const double epsStep = nk > 1 ? (epsMax - epsMin) / static_cast<double>(nk - 1) : 0.0;

    for (std::size_t ip = 0; ip < np; ++ip) {
        if (((ip + 1) % 1000) == 0) {
            const std::size_t globalIndex =
                ip + 1 + static_cast<std::size_t>(mpiRank) * np;
            std::cout << "Rank " << mpiRank << " computing prediction number: "
                      << globalIndex << " (local " << (ip + 1) << ")\n";
        }

        std::vector<double> fp(static_cast<std::size_t>(nk), 0.0);
        for (int k = 0; k < nk; ++k) {
            double eps = epsMin + epsStep * static_cast<double>(k);
            if (std::abs(eps - 2.0) < std::numeric_limits<double>::epsilon()) {
                eps += (2.0 / static_cast<double>(nk - 1)) / 2.0;
            }

            const auto weights = rbf(nt, trainingPoints, trainingValues, eps);

            double prediction = 0.0;
            for (std::size_t j = 0; j < nt; ++j) {
                double d2 = 0.0;
                for (std::size_t p = 0; p < numParams; ++p) {
                    const double diff = predictionPoints[ip][p] - trainingPoints[j][p];
                    d2 += diff * diff;
                }
                const double d = std::sqrt(d2);
                prediction += weights[j] * std::pow(d, eps);
            }
            fp[static_cast<std::size_t>(k)] = prediction;
        }

        auto fpOrdered = fp;
        std::sort(fpOrdered.begin(), fpOrdered.end());
        expectedValues[ip] = std::accumulate(fpOrdered.begin(), fpOrdered.end(), 0.0) /
                             static_cast<double>(nk);

        std::size_t lowerIdx = 0;
        if (kshift > 0) {
            lowerIdx = static_cast<std::size_t>(std::max(0, kshift - 1));
        }
        lowerIdx = std::min(lowerIdx, fpOrdered.size() - 1);

        std::size_t upperIdx = fpOrdered.size() - 1;
        if (nk - kshift > 0) {
            upperIdx = static_cast<std::size_t>(nk - kshift - 1);
            upperIdx = std::min(upperIdx, fpOrdered.size() - 1);
        }
        if (lowerIdx > upperIdx) {
            std::swap(lowerIdx, upperIdx);
        }

        const double fpMin = fpOrdered[lowerIdx];
        const double fpMax = fpOrdered[upperIdx];
        variances[ip] = (fpMax - fpMin) / 2.0;
    }

    const auto minIt = std::min_element(expectedValues.begin(), expectedValues.end());
    const auto maxIt = std::max_element(variances.begin(), variances.end());

    SRBFResult result{};
    if (minIt != expectedValues.end()) {
        result.minValue = *minIt;
    }
    if (maxIt != variances.end()) {
        result.maxVariance = *maxIt;
        result.indexOfMaxVariance = static_cast<std::size_t>(std::distance(variances.begin(), maxIt));
    }

    return result;
}

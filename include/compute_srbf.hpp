#pragma once

#include <cstddef>
#include <vector>

/*
PROJECT: NOSTRUM, https://nostrum-project.it
AUTHORS: Alessio Castorrini, Riccardo Broglia, Rosalba Cardamone
DATE: 2026-01-01

DESCRIPTION:
In this header file, we declare the computeSRBF function, which is responsible for computing the Surrogate Radial Basis Function (SRBF) predictions based on the provided training data and prediction points.
*/

struct SRBFResult {
    double minValue;
    double maxVariance;
    std::size_t indexOfMaxVariance;
};

SRBFResult computeSRBF(
    std::size_t numParams,
    int kshift,
    const std::vector<std::vector<double>>& predictionPoints,
    int nk,
    std::size_t nt,
    const std::vector<std::vector<double>>& trainingPoints,
    const std::vector<double>& trainingValues);

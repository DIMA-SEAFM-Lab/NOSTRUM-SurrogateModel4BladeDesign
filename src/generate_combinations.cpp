#include "generate_combinations.hpp"

#include <vector>

static std::vector<std::vector<double>> baseCombinations(
    const std::vector<std::vector<double>>& paramVectors,
    std::size_t numParams) {
    if (numParams == 0) {
        return {};
    }

    if (numParams == 1) {
        std::vector<std::vector<double>> combinations;
        combinations.reserve(paramVectors.front().size());
        for (double value : paramVectors.front()) {
            combinations.push_back({value});
        }
        return combinations;
    }

    auto prevCombinations = baseCombinations(paramVectors, numParams - 1);
    const auto& currentVector = paramVectors[numParams - 1];

    std::vector<std::vector<double>> combinations;
    combinations.reserve(prevCombinations.size() * currentVector.size());
    for (const auto& prev : prevCombinations) {
        for (double value : currentVector) {
            auto row = prev;
            row.push_back(value);
            combinations.push_back(std::move(row));
        }
    }

    return combinations;
}

std::vector<std::vector<double>> generateCombinations(
    const std::vector<std::vector<double>>& paramVectors,
    std::size_t numParams) {
    if (paramVectors.empty() || numParams == 0) {
        return {};
    }

    return baseCombinations(paramVectors, numParams);
}

#include "generate_param_grid.hpp"

#include "generate_combinations.hpp"

std::vector<std::vector<double>> generateParamGrid(
    const std::vector<std::vector<double>>& paramVectors) {
    if (paramVectors.empty()) {
        return {};
    }

    return generateCombinations(paramVectors, paramVectors.size());
}

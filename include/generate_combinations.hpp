#pragma once

#include <vector>
/*
PROJECT: NOSTRUM, https://nostrum-project.it
AUTHORS: Alessio Castorrini, Riccardo Broglia, Rosalba Cardamone
DATE: 2026-01-01

DESCRIPTION:
In this header file, we declare the generateCombinations function, which is responsible for generating all possible combinations of parameters from a set of parameter vectors.
*/

std::vector<std::vector<double>> generateCombinations(
    const std::vector<std::vector<double>>& paramVectors,
    std::size_t numParams);

#pragma once

#include <vector>

/*
PROJECT: NOSTRUM, https://nostrum-project.it
AUTHORS: Alessio Castorrini, Riccardo Broglia, Rosalba Cardamone
DATE: 2026-01-01

DESCRIPTION:
In this header file, we declare the generateParamGrid function, which is responsible for generating a parameter grid from a set of parameter vectors.
*/

std::vector<std::vector<double>> generateParamGrid(
    const std::vector<std::vector<double>>& paramVectors);

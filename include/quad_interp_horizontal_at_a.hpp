#pragma once

#include <array>
#include <vector>

/*
PROJECT: NOSTRUM, https://nostrum-project.it
AUTHORS: Alessio Castorrini, Riccardo Broglia, Rosalba Cardamone
DATE: 2026-01-01

DESCRIPTION:
Interpolation helpers
*/

struct QuadInterpResult {
    std::vector<double> values;
    std::array<double, 3> coefficients;
};

QuadInterpResult quadInterpHorizontalAtA(
    const std::vector<double>& x,
    double xA,
    double yA,
    double xB,
    double yB);

#pragma once

#include <cstddef>
#include <vector>

/*
NOSTRUM, https://nostrum-project.it
AUTHORS: Alessio Castorrini, Riccardo Broglia, Rosalba Cardamone
DATE: 2026-01-01

DESCRIPTION:
In this header file, we declare the rbf function, which is responsible for computing the weights for Radial Basis Function (RBF) interpolation. The function takes as input the number of training points, the training points themselves, the corresponding training values, and a shape parameter (eps) that controls the smoothness of the RBF interpolation. The output is a vector of weights that can be used to make predictions at new points based on the training data.
*/

std::vector<double> rbf(
    std::size_t nt,
    const std::vector<std::vector<double>>& xt,
    const std::vector<double>& yt,
    double eps);

#include "quad_interp_horizontal_at_a.hpp"

#include <stdexcept>

QuadInterpResult quadInterpHorizontalAtA(
    const std::vector<double>& x,
    double xA,
    double yA,
    double xB,
    double yB) {
    if (xB == xA) {
        throw std::invalid_argument("Points A and B must have different x-coordinates.");
    }

    const double k = (yB - yA) / ((xB - xA) * (xB - xA));
    QuadInterpResult result;
    result.values.reserve(x.size());
    for (double xi : x) {
        const double diff = xi - xA;
        result.values.push_back(yA + k * diff * diff);
    }

    result.coefficients = {k, -2.0 * k * xA, yA + k * xA * xA};
    return result;
}

#include "HF_Model.hpp"

#include "quad_interp_horizontal_at_a.hpp"
#include "read_fast_binary.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> solveDenseSystem(std::vector<std::vector<double>> a, std::vector<double> b) {
    const std::size_t n = a.size();
    if (b.size() != n) {
        throw std::invalid_argument("Linear system dimension mismatch.");
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

        if (maxVal == 0.0) {
            throw std::runtime_error("Singular matrix encountered in spline setup.");
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
    for (std::size_t idx = n; idx-- > 0;) {
        double sum = 0.0;
        for (std::size_t col = idx + 1; col < n; ++col) {
            sum += a[idx][col] * x[col];
        }
        x[idx] = b[idx] - sum;
    }

    return x;
}

std::vector<double> computeSecondDerivatives(
    const std::vector<double>& x,
    const std::vector<double>& y) {
    const std::size_t n = x.size();
    if (n != y.size() || n < 2) {
        throw std::invalid_argument("Invalid spline data.");
    }

    if (n == 2) {
        return std::vector<double>(n, 0.0);
    }

    std::vector<double> h(n - 1, 0.0);
    for (std::size_t i = 0; i < n - 1; ++i) {
        h[i] = x[i + 1] - x[i];
        if (h[i] == 0.0) {
            throw std::invalid_argument("Duplicate x values in spline.");
        }
    }

    std::vector<std::vector<double>> A(n, std::vector<double>(n, 0.0));
    std::vector<double> rhs(n, 0.0);

    // Not-a-knot at the first two intervals
    A[0][0] = -h[1];
    A[0][1] = h[0] + h[1];
    A[0][2] = -h[0];
    rhs[0] = 0.0;

    // Interior equations
    for (std::size_t i = 1; i < n - 1; ++i) {
        A[i][i - 1] = h[i - 1];
        A[i][i] = 2.0 * (h[i - 1] + ((i < n - 1) ? h[i] : 0.0));
        if (i + 1 < n) {
            A[i][i + 1] = h[i];
        }
        rhs[i] = 6.0 *
                 ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    // Not-a-knot at the last two intervals
    const std::size_t last = n - 1;
    A[last][last - 2] = h[last - 1];
    A[last][last - 1] = -(h[last - 2] + h[last - 1]);
    A[last][last] = h[last - 2];
    rhs[last] = 0.0;

    return solveDenseSystem(A, rhs);
}

std::vector<double> evaluateSpline(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& secondDerivatives,
    const std::vector<double>& query) {
    std::vector<double> result;
    result.reserve(query.size());

    for (double q : query) {
        auto it = std::upper_bound(x.begin(), x.end(), q);
        std::size_t idx = std::distance(x.begin(), it);
        if (idx == 0) {
            idx = 1;
        }
        if (idx >= x.size()) {
            idx = x.size() - 1;
        }

        const std::size_t i = idx - 1;
        const double h = x[i + 1] - x[i];
        const double a = (x[i + 1] - q) / h;
        const double b = (q - x[i]) / h;
        const double term = ((a * a * a - a) * secondDerivatives[i] +
                             (b * b * b - b) * secondDerivatives[i + 1]) *
                            (h * h) / 6.0;
        result.push_back(a * y[i] + b * y[i + 1] + term);
    }

    return result;
}

std::vector<double> splineInterpolate(
    const std::vector<double>& x,
    const std::vector<double>& y,
    const std::vector<double>& query) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Spline input size mismatch.");
    }

    if (x.size() == 2) {
        // Linear interpolation fallback
        std::vector<double> result;
        result.reserve(query.size());
        for (double q : query) {
            double t = (q - x[0]) / (x[1] - x[0]);
            t = std::clamp(t, 0.0, 1.0);
            result.push_back(y[0] * (1.0 - t) + y[1] * t);
        }
        return result;
    }

    const auto secondDerivatives = computeSecondDerivatives(x, y);
    return evaluateSpline(x, y, secondDerivatives, query);
}

void writeBladeFile(
    const std::string& path,
    const std::vector<std::string>& headerLines,
    const std::vector<std::vector<double>>& data) {
    std::ofstream file(path);
    if (!file) {
        throw std::runtime_error("Unable to open blade file for writing: " + path);
    }

    for (const auto& line : headerLines) {
        file << line << '\n';
    }

    file << std::fixed << std::setprecision(6);
    constexpr std::size_t kIntegerColumnIndex = 6;  // seventh column must remain integer
    for (const auto& row : data) {
        for (std::size_t col = 0; col < row.size(); ++col) {
            if (col == kIntegerColumnIndex) {
                file << static_cast<long long>(std::llround(row[col]));
            } else {
                file << row[col];
            }

            if (col + 1 < row.size()) {
                file << '\t';
            }
        }
        file << '\n';
    }
}

double computeChannelMean(
    const std::vector<std::vector<double>>& channels,
    std::size_t startRow,
    std::size_t column) {
    if (channels.empty() || column >= channels.front().size()) {
        throw std::runtime_error("Invalid channel data.");
    }

    startRow = std::min(startRow, channels.size() - 1);

    double sum = 0.0;
    std::size_t count = 0;
    for (std::size_t i = startRow; i < channels.size(); ++i) {
        sum += channels[i][column];
        ++count;
    }

    if (count == 0) {
        throw std::runtime_error("No data to compute mean.");
    }

    return sum / static_cast<double>(count);
}

int runCommand(const std::string& command) {
    return std::system(command.c_str());
}

std::string quote(const std::filesystem::path& path) {
    std::string quoted = path.string();
    quoted.insert(0, "\"");
    quoted.push_back('"');
    return quoted;
}

std::filesystem::path pickExistingPath(
    const std::filesystem::path& preferred,
    const std::filesystem::path& fallback) {
    if (std::filesystem::exists(preferred)) {
        return preferred;
    }
    if (std::filesystem::exists(fallback)) {
        return fallback;
    }
    return preferred;
}

}  // namespace

std::pair<double, double> HF_Model::computeCPWithTorsionDOF(
    const std::vector<double>& xt,
    const std::vector<double>& xcTheta,
    const std::vector<double>& xs,
    int torsionStartIndex,
    const std::vector<std::vector<double>>& dataBladeBaseline,
    const std::vector<std::string>& headerLines,
    int twistColumnIndex,
    int numberBladeElements,
    int rotCpChannelIndex,
    const std::filesystem::path& projectRoot,
    const std::filesystem::path& aerodynExecutable,
    const std::filesystem::path& aerodynCaseRoot) {
    const int nTheta = static_cast<int>(xcTheta.size());
    if (static_cast<int>(xt.size()) != nTheta + 1) {
        throw std::invalid_argument("Unexpected number of design variables.");
    }

    std::vector<double> thetaOpt(xt.begin(), xt.begin() + nTheta);
    const double torsionOpt = xt.back();

    const auto thetas = splineInterpolate(xcTheta, thetaOpt, xs);

    std::vector<double> torsion(static_cast<std::size_t>(numberBladeElements), 0.0);
    const int torsionIdx = std::clamp(torsionStartIndex, 0, numberBladeElements - 1);
    std::vector<double> torsionX;
    torsionX.reserve(static_cast<std::size_t>(numberBladeElements - torsionIdx));
    for (int i = torsionIdx; i < numberBladeElements; ++i) {
        torsionX.push_back(dataBladeBaseline[static_cast<std::size_t>(i)][0]);
    }

    const auto torsionInterp = quadInterpHorizontalAtA(
        torsionX,
        dataBladeBaseline[static_cast<std::size_t>(torsionIdx)][0],
        0.0,
        dataBladeBaseline.back()[0],
        torsionOpt);

    for (std::size_t i = torsionIdx; i < static_cast<std::size_t>(numberBladeElements); ++i) {
        torsion[i] = torsionInterp.values[i - torsionIdx];
    }

    auto dataBladeTest = dataBladeBaseline;
    for (std::size_t i = 0; i < thetas.size(); ++i) {
        dataBladeTest[i][static_cast<std::size_t>(twistColumnIndex)] = thetas[i];
    }

    const auto logDir = projectRoot / "LOG_Files";
    std::filesystem::create_directories(logDir);

    const auto bladeTestPath = aerodynCaseRoot / "blade_test.dat";
    writeBladeFile(bladeTestPath.string(), headerLines, dataBladeTest);

    const auto aeroDynRe1Drv = aerodynCaseRoot / "AeroDynRe1.drv";
    const auto logAerodyn1 = logDir / "log.Aerodyn1";
    const std::string aerodynRe1Command =
        quote(aerodynExecutable) + " " + quote(aeroDynRe1Drv) + " > " + quote(logAerodyn1) + " 2>&1";
    if (runCommand(aerodynRe1Command) != 0) {
        std::cout << "Command executed: " << aerodynRe1Command << std::endl;
        throw std::runtime_error("aerodyn_driver execution failed for Re1.");
    }

    const auto aeroDynRe1Out =
        pickExistingPath(projectRoot / "AeroDynRe1.1.outb", aerodynCaseRoot / "AeroDynRe1.1.outb");
    const auto resultRe1 = readFASTbinary(aeroDynRe1Out.string());
    std::size_t initRow = static_cast<std::size_t>(
        std::llround(0.8 * static_cast<double>(resultRe1.channels.size())));
    if (!resultRe1.channels.empty()) {
        initRow = std::min(initRow, resultRe1.channels.size() - 1);
    }
    const std::size_t rotIndex = static_cast<std::size_t>(std::max(0, rotCpChannelIndex - 1));
    const double cp1 = computeChannelMean(resultRe1.channels, initRow, rotIndex);

    for (std::size_t i = 0; i < thetas.size(); ++i) {
        dataBladeTest[i][static_cast<std::size_t>(twistColumnIndex)] = thetas[i] + torsion[i];
    }

    writeBladeFile(bladeTestPath.string(), headerLines, dataBladeTest);

    const auto aeroDynRe2Drv = aerodynCaseRoot / "AeroDynRe2.drv";
    const auto logAerodyn2 = logDir / "log.Aerodyn2";
    const std::string aerodynRe2Command =
        quote(aerodynExecutable) + " " + quote(aeroDynRe2Drv) + " > " + quote(logAerodyn2) + " 2>&1";
    if (runCommand(aerodynRe2Command) != 0) {
        throw std::runtime_error("aerodyn_driver execution failed for Re2.");
    }

    const auto aeroDynRe2Out =
        pickExistingPath(projectRoot / "AeroDynRe2.1.outb", aerodynCaseRoot / "AeroDynRe2.1.outb");
    const auto resultRe2 = readFASTbinary(aeroDynRe2Out.string());
    initRow = static_cast<std::size_t>(
        std::llround(0.8 * static_cast<double>(resultRe2.channels.size())));
    if (!resultRe2.channels.empty()) {
        initRow = std::min(initRow, resultRe2.channels.size() - 1);
    }
    const double cp2 = computeChannelMean(resultRe2.channels, initRow, rotIndex);

    return {cp1, cp2};
}

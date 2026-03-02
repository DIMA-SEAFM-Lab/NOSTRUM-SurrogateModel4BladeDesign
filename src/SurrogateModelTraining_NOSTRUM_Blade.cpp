#include "compute_srbf.hpp"
#include "generate_param_grid.hpp"
#include "HF_Model.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mpi.h>

namespace {

// Data structures
struct Params {
    std::filesystem::path projectRoot;
    int pointsPerParam = 0;
    int nk = 0;
    int iterMax = 0;
    double tolerance = 0.0;
};


std::filesystem::path resolveConfigPath(
    const std::filesystem::path& requestedPath,
    const char* argv0) {
    if (requestedPath.is_absolute()) {
        if (std::filesystem::exists(requestedPath)) {
            return requestedPath;
        }
        throw std::runtime_error(
            "Configuration file does not exist: " + requestedPath.string());
    }

    std::vector<std::filesystem::path> candidates;
    candidates.push_back(requestedPath);
    candidates.push_back(std::filesystem::current_path() / requestedPath);

    if (argv0 != nullptr && argv0[0] != '\0') {
        std::error_code ec;
        const auto executablePath =
            std::filesystem::absolute(std::filesystem::path(argv0), ec);
        if (!ec) {
            const auto executableDir = executablePath.parent_path();
            candidates.push_back(executableDir / requestedPath);
            const auto parentDir = executableDir.parent_path();
            if (!parentDir.empty()) {
                candidates.push_back(parentDir / requestedPath);
            }
        }
    }

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec) && !ec) {
            const auto canonical = std::filesystem::weakly_canonical(candidate, ec);
            return ec ? std::filesystem::absolute(candidate) : canonical;
        }
    }

    std::ostringstream oss;
    oss << "Failed to locate configuration file '" << requestedPath.string()
        << "'. Current working directory: " << std::filesystem::current_path().string()
        << ". Checked paths:";
    for (const auto& candidate : candidates) {
        oss << "\n  - " << candidate.string();
    }
    throw std::runtime_error(oss.str());
}


Params loadParametersFromConfigFile(const std::filesystem::path& filename){
    std::ifstream file(filename);
    Params params;
    std::string line;

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open the configuration file: " + filename.string());
    }

    while (std::getline(file, line)) {
        // Trim leading whitespace
        line.erase(line.begin(), std::find_if(line.begin(), line.end(), [](unsigned char ch) {
            return !std::isspace(ch);
            }));

        // Skip empty lines and comment lines
        if (line.empty() || line[0] == '#' || line.substr(0, 2) == "//") {
            continue;
        }

        std::istringstream lineStream(line);
        std::string key;
        if (std::getline(lineStream, key, '=')) {
            std::string value;
            if (std::getline(lineStream, value)) {
                // Set the corresponding parameter based on the key
                if (key == "project_path") params.projectRoot = value;
                else if (key == "pointsPerParam") params.pointsPerParam = std::stoi(value);
                else if (key == "nk") params.nk = std::stoi(value);
                else if (key == "iterMax") params.iterMax = std::stoi(value);
                else if (key == "tolerance") params.tolerance = std::stod(value);
            }
        }
    }
    file.close();

    // Check that all parameters have been given in input
    bool valid = true;
    if (params.projectRoot.empty()) {
        std::cerr << "Error: project_path is missing in the configuration file." << std::endl;
        valid = false;
    }
    if (params.pointsPerParam <= 0) {
        std::cerr << "Error: pointsPerParam must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (params.nk <= 0) {
        std::cerr << "Error: nk must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (params.iterMax <= 0) {
        std::cerr << "Error: iterMax must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (params.tolerance <= 0.0) {
        std::cerr << "Error: tolerance must be a positive number in the configuration file." << std::endl;
        valid = false;
    }

    if (!valid) {
        throw std::runtime_error(
            "Invalid or incomplete configuration file: " + filename.string());
    }
    return params;
}



std::vector<std::vector<double>> buildParamVectors(
    const std::vector<double>& lowerBounds,
    const std::vector<double>& upperBounds,
    std::size_t pointsPerParam) {
    if (pointsPerParam < 2) {
        throw std::runtime_error("At least two grid points per parameter are required.");
    }

    const std::size_t numParams = lowerBounds.size();
    std::vector<std::vector<double>> paramVectors(numParams);

    for (std::size_t i = 0; i < numParams; ++i) {
        const double lb = lowerBounds[i];
        const double ub = upperBounds[i];
        const double step = (ub - lb) / static_cast<double>(pointsPerParam - 1);
        paramVectors[i].reserve(pointsPerParam);
        for (std::size_t j = 0; j < pointsPerParam; ++j) {
            paramVectors[i].push_back(lb + step * static_cast<double>(j));
        }
    }

    return paramVectors;
}

void appendTrainingData(
    const std::filesystem::path& filePath,
    const std::vector<std::vector<double>>& xt,
    const std::vector<std::pair<double, double>>& ft) {
    std::ofstream file(filePath, std::ios::app);
    if (!file) {
        throw std::runtime_error("Unable to open training log file: " + filePath.string());
    }

    for (std::size_t i = 0; i < xt.size(); ++i) {
        for (double value : xt[i]) {
            file << value << ' ';
        }
        file << ft[i].first << ' ' << ft[i].second << '\n';
    }
}

void saveSurrogateModel(
    const std::filesystem::path& outputPath,
    std::size_t nt,
    int nk,
    int kshift,
    const std::vector<std::vector<double>>& xt,
    const std::vector<std::pair<double, double>>& ft) 
    {
    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to write surrogate model file: " + outputPath.string());
    }

    file << "nt " << nt << '\n';
    file << "nk " << nk << '\n';
    file << "kshift " << kshift << '\n';

    file << "xt\n";
    for (std::size_t i = 0; i < nt; ++i) {
        for (double value : xt[i]) {
            file << value << ' ';
        }
        file << '\n';
    }

    file << "ft\n";
    for (std::size_t i = 0; i < nt; ++i) {
        file << ft[i].first << ' ' << ft[i].second << '\n';
    }
    file << '\n';
}

std::vector<double> extractTrainingValues(
    const std::vector<std::pair<double, double>>& ft,
    std::size_t column) {
    if (column > 1) {
        throw std::out_of_range("extractTrainingValues column must be 0 or 1");
    }
    std::vector<double> values;
    values.reserve(ft.size());
    for (const auto& row : ft) {
        values.push_back(column == 0 ? row.first : row.second);
    }
    return values;
}

std::string formatVectorLine(
    const std::vector<double>& values,
    int precision = 6) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        oss << values[i];
        if (i + 1 < values.size()) {
            oss << ", ";
        }
    }
    oss << "]";
    return oss.str();
}


}  // namespace

int main(int argc, char* argv[]) {



    MPI_Init(&argc, &argv);
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // HEADER OUTPUT
    if(rank == 0) {
        std::cout << "=====================================================\n";
        std::cout << "          NOSTRUM-PROJECT Blade Optimization  \n";
        std::cout << "              Surrogate Model Training\n";
        std::cout << "=====================================================\n\n";
        std::cout << "Authors: Castorrini A., Broglia R., Cardamone R.\n\n";
        std::cout << "Version           : 1.0, Date: 28/02/2026 \n";
        std::cout << "-----------------------------------------------------\n";
        std::cout << "\n";
        std::cout << "MPI processes      : " << size << '\n';
        std::cout << "Current directory  : " << std::filesystem::current_path().string() << '\n';
        std::cout << "HF evaluations     : rank 0 only\n";
        std::cout << "-----------------------------------------------------\n";
    }

    Params params;
    const std::filesystem::path defaultConfigFile = "config_main.txt";
    std::filesystem::path loadedConfigPath;
    try {
        if(rank==0){
            const auto requestedConfigPath =
                (argc > 1) ? std::filesystem::path(argv[1]) : defaultConfigFile;
            loadedConfigPath = resolveConfigPath(requestedConfigPath, argv[0]);
            std::cout << "Loading configuration from: " << loadedConfigPath << '\n';
            params = loadParametersFromConfigFile(loadedConfigPath);
        }
        // Broadcast parameters to all processes
        {
            int projectRootLength = 0;
            std::string projectRootString;
            if(rank == 0){
                projectRootString = params.projectRoot.string();
                projectRootLength = static_cast<int>(projectRootString.size());
            }
            MPI_Bcast(&projectRootLength, 1, MPI_INT, 0, MPI_COMM_WORLD);
            std::vector<char> projectRootBuffer(projectRootLength + 1, '\0');
            if(rank == 0){
                std::copy(projectRootString.begin(), projectRootString.end(), projectRootBuffer.begin());
            }
            MPI_Bcast(projectRootBuffer.data(), projectRootLength + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
            if(rank != 0){
                params.projectRoot = std::string(projectRootBuffer.data());
            }
            MPI_Bcast(&params.pointsPerParam, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&params.nk, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&params.iterMax, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&params.tolerance, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }   
        MPI_Barrier(MPI_COMM_WORLD);

        const auto projectRoot = params.projectRoot;
        const auto trainingLogPath = projectRoot / "train_log.dat";
        const auto resultsDir = projectRoot / "Results/SM_training";
        const auto surrogateModelPath = resultsDir / "SM_Database.txt";
        const auto hfModelPath = resultsDir / "HF_Model_info.txt";

        HF_Model hfModel;
        if(rank == 0){
            hfModel.initialize(projectRoot);
            //hfModel.defineBounds(); // Deprecated since bounds are now read from config file
            hfModel.saveDataOnFile(hfModelPath);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        std::vector<double> lb;
        std::vector<double> ub;
        //Broadcast bounds to all processes
        {
            std::size_t numParams = 0;
            if(rank == 0){
                lb = hfModel.lb;
                ub = hfModel.ub;
                numParams = lb.size();
            }
            MPI_Bcast(&numParams, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if(rank != 0){
                lb.resize(numParams);
                ub.resize(numParams);
            }
            MPI_Bcast(lb.data(), static_cast<int>(numParams), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Bcast(ub.data(), static_cast<int>(numParams), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
        if (rank == 0) {
            std::cout << "Run configuration:\n";
            std::cout << "  Config file      : " << loadedConfigPath.string() << '\n';
            std::cout << "  Project root     : " << projectRoot.string() << '\n';
            std::cout << "  pointsPerParam   : " << params.pointsPerParam << '\n';
            std::cout << "  nk               : " << params.nk << '\n';
            std::cout << "  iterMax          : " << params.iterMax << '\n';
            std::cout << "  tolerance        : " << params.tolerance << '\n';
            std::cout << "  Design variables : " << lb.size() << '\n';
            std::cout << "  LB               : " << formatVectorLine(lb) << '\n';
            std::cout << "  UB               : " << formatVectorLine(ub) << '\n';
            std::cout << "  Outputs:\n";
            std::cout << "    - " << trainingLogPath.string() << '\n';
            std::cout << "    - " << surrogateModelPath.string() << '\n';
            std::cout << "    - " << hfModelPath.string() << '\n';
            std::cout << "-----------------------------------------------------\n";
        }

        std::vector<std::vector<double>> xt;
        std::vector<std::pair<double, double>> ft;
        xt.reserve(10000);
        ft.reserve(10000);

        xt.push_back(lb);
        if(rank == 0){
            ft.push_back(hfModel.computeFT(xt.back()));
        }

        xt.push_back(ub);
        if(rank == 0){
            ft.push_back(hfModel.computeFT(xt.back()));
        }

        std::vector<double> xt_init;
        if (rank == 0) {
            xt_init = hfModel.thetaRef;
            xt_init.push_back(hfModel.initTorsion);
        }
        {
            std::size_t xtInitSize = xt_init.size();
            MPI_Bcast(&xtInitSize, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if (rank != 0) {
                xt_init.resize(xtInitSize);
            }
            if (xtInitSize > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("xt_init size exceeds int range required by MPI_Bcast.");
            }
            if (xtInitSize > 0) {
                MPI_Bcast(
                    xt_init.data(),
                    static_cast<int>(xtInitSize),
                    MPI_DOUBLE,
                    0,
                    MPI_COMM_WORLD);
            }
        }
        xt.push_back(xt_init);
        if(rank == 0){
            ft.push_back(hfModel.computeFT(xt.back()));
        }
        // Broadcast initial function evaluations to all processes
        {
            std::size_t numInitial = ft.size();
            MPI_Bcast(&numInitial, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
            if(rank != 0){
                ft.resize(numInitial);
            }
            for(std::size_t i = 0; i < numInitial; ++i){
                double cp1 = 0.0;
                double cp2 = 0.0;
                if(rank == 0){
                    cp1 = ft[i].first;
                    cp2 = ft[i].second;
                }
                MPI_Bcast(&cp1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&cp2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if(rank != 0){
                    ft[i] = std::make_pair(cp1, cp2);
                }
            }
        }

        const std::size_t numParams = lb.size();

        const std::size_t pointsPerParam = params.pointsPerParam > 0 ? params.pointsPerParam : 6;
        const auto paramVectors = buildParamVectors(lb, ub, pointsPerParam);
        const auto xp = generateParamGrid(paramVectors);

        const int nk = params.nk;
        const int kshift = nk / 20;

        if (rank == 0) std::filesystem::create_directories(resultsDir);

        int iter = 1;
        int iterMax = params.iterMax;
        double tolerance = params.tolerance;
        bool done1 = false;
        bool done2 = false;
        double maxVar1 = 0.0;
        double maxVar2 = 0.0;
        std::size_t j1 = 0;
        std::size_t j2 = 0;

        while (iter < iterMax) {
            if(rank == 0)  appendTrainingData(trainingLogPath, xt, ft);

            const std::size_t nt = xt.size();

            if (!done1) {
                const auto values = extractTrainingValues(ft, 0);

                // We now compute the SRBF with a different process dealing with a portion of xp
                MPI_Barrier(MPI_COMM_WORLD);
                // Decompose xp among processes
                std::size_t numXp = xp.size();
                std::size_t chunkSize = (numXp + size - 1) / size;
                std::size_t startIdx = rank * chunkSize;
                std::size_t endIdx = std::min(startIdx + chunkSize, numXp);
                std::vector<std::vector<double>> xpChunk;
                for(std::size_t i = startIdx; i < endIdx; ++i){
                    xpChunk.push_back(xp[i]);
                }
                const auto srbfResult = computeSRBF(numParams, kshift, xpChunk, nk, nt, xt, values);
                MPI_Barrier(MPI_COMM_WORLD);
                // Gather max variances and indices from all processes
                struct {
                    double maxVariance;
                    int index;
                } localData, globalData;
                localData.maxVariance = srbfResult.maxVariance;
                const std::size_t globalIdx1 = srbfResult.indexOfMaxVariance + startIdx;
                if (globalIdx1 > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("Global index exceeds int range required by MPI_MAXLOC.");
                }
                localData.index = static_cast<int>(globalIdx1); // Adjust index
                MPI_Allreduce(&localData, &globalData, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            
                maxVar1 = globalData.maxVariance;
                j1 = static_cast<std::size_t>(globalData.index);
                if(rank == 0) {
                    std::cout << "Iter: " << iter << ", Error/tol (1): "
                              << std::abs(maxVar1) / tolerance << ", nt: " << nt << '\n';
                }
                if (std::abs(maxVar1) <= tolerance) {
                    done1 = true;
                }
            }

            if (!done2) {
                const auto values = extractTrainingValues(ft, 1);
                MPI_Barrier(MPI_COMM_WORLD);
                // Decompose xp among processes
                std::size_t numXp = xp.size();
                std::size_t chunkSize = (numXp + size - 1) / size;
                std::size_t startIdx = rank * chunkSize;
                std::size_t endIdx = std::min(startIdx + chunkSize, numXp);
                std::vector<std::vector<double>> xpChunk;
                for(std::size_t i = startIdx; i < endIdx; ++i){
                    xpChunk.push_back(xp[i]);
                }
                const auto srbfResult = computeSRBF(numParams, kshift, xpChunk, nk, nt, xt, values);
                MPI_Barrier(MPI_COMM_WORLD);
                // Gather max variances and indices from all processes
                struct {
                    double maxVariance;
                    int index;
                } localData, globalData;
                localData.maxVariance = srbfResult.maxVariance;
                const std::size_t globalIdx2 = srbfResult.indexOfMaxVariance + startIdx;
                if (globalIdx2 > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                    throw std::runtime_error("Global index exceeds int range required by MPI_MAXLOC.");
                }
                localData.index = static_cast<int>(globalIdx2); // Adjust index
                MPI_Allreduce(&localData, &globalData, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
                maxVar2 = globalData.maxVariance;
                j2 = static_cast<std::size_t>(globalData.index);
                if(rank == 0) {
                    std::cout << "Iter: " << iter << ", Error/tol (2): "
                              << std::abs(maxVar2) / tolerance << ", nt: " << nt << '\n';
                }
                if (std::abs(maxVar2) <= tolerance) {
                    done2 = true;
                }
            }

            if (done1 && done2) {
                break;
            }

            const std::size_t j = (std::abs(maxVar1) > std::abs(maxVar2)) ? j1 : j2;
            if (j >= xp.size()) {
                throw std::runtime_error("Prediction index out of bounds.");
            }

            xt.push_back(xp[j]);
            if(rank == 0){
                ft.push_back(hfModel.computeFT(xt.back()));

                saveSurrogateModel(
                        surrogateModelPath,
                        xt.size(),
                        nk,
                        kshift,
                        xt,
                        ft);
            }
            // Broadcast new function evaluation to all processes
            {
                double cp1 = 0.0;
                double cp2 = 0.0;
                if(rank == 0){
                    cp1 = ft.back().first;
                    cp2 = ft.back().second;
                }
                MPI_Bcast(&cp1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                MPI_Bcast(&cp2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                if(rank != 0){
                    ft.push_back(std::make_pair(cp1, cp2));
                }
            }
            ++iter;
        }

        if(rank == 0) {
            std::cout << "Adaptive sampling completed after " << iter << " iterations." << std::endl;
        }
    } catch (const std::exception& ex) {
        if(rank == 0) {
            std::cerr << "Error: " << ex.what() << std::endl;
        }
        return 1;
    }
    MPI_Finalize();

    return 0;
}

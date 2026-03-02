#include "HF_Model.hpp"

// Implementation of HF_Model member functions

// Initialization and read files
void HF_Model::initialize(const std::filesystem::path& projectRoot) {
    //Read info
    loadParametersFromConfigFileHF(projectRoot / "config_HF_Model.txt");
    // Paths
    aerodynCaseRoot = projectRoot / "AeroDynFiles";
    // Load blade data
    bladeData = loadBladeData(aerodynCaseRoot / "Input_Files" / bladeFileName);

    const auto& dataBladeBaseline = bladeData.numericData;
    const auto& headerLines = bladeData.headerLines;

    xs = extractColumn(dataBladeBaseline, static_cast<std::size_t>(spanID - 1));

    const std::vector<std::size_t> thetaControlPointIndices = CP_indices;
    xcTheta = gatherByIndices(xs, thetaControlPointIndices);
    
    thetaRef.reserve(thetaControlPointIndices.size());
    for (std::size_t idx : thetaControlPointIndices) {
        thetaRef.push_back(dataBladeBaseline[idx][static_cast<std::size_t>(twistID - 1)]);
    }
    numberBladeElements = static_cast<int>(dataBladeBaseline.size());

}


void HF_Model::defineBounds() { //Deprecated function
    // Define upper and lower bounds for the surrogate model parameters
    // This function was originally implemented for the NOSTRUM project. Now a more general formulation takes the list of UB and LB directly from input
    deltaMaxRoot = 3.0;
    deltaMaxTip = 2.0;

    const auto deltaInterp = interpolateLinearProfile(
        xcTheta,
        xcTheta.front(),
        xcTheta.back(),
        deltaMaxRoot,
        deltaMaxTip);

    std::vector<double> lbTheta(thetaRef.size());
    std::vector<double> ubTheta(thetaRef.size());
    for (std::size_t i = 0; i < thetaRef.size(); ++i) {
        lbTheta[i] = thetaRef[i] - deltaInterp[i];
        ubTheta[i] = thetaRef[i] + deltaInterp[i];
    }

    deltaMaxTorsion = 1.5;
    initTorsion = 1.5;
    const double lbTorsion = initTorsion - deltaMaxTorsion;
    const double ubTorsion = initTorsion + deltaMaxTorsion;

    lb = lbTheta;
    lb.push_back(lbTorsion);
    ub = ubTheta;
    ub.push_back(ubTorsion);

}

BladeData HF_Model::loadBladeData(const std::filesystem::path& bladeFilePath) {
    std::ifstream input(bladeFilePath);
    if (!input) {
        throw std::runtime_error("Unable to open blade file: " + bladeFilePath.string());
    }

    BladeData result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            result.headerLines.push_back(line);
            continue;
        }

        std::istringstream iss(line);
        std::vector<double> row;
        bool numeric = true;
        std::string token;
        while (iss >> token) {
            char* end = nullptr;
            const double value = std::strtod(token.c_str(), &end);
            if (end == token.c_str() || *end != '\0') {
                numeric = false;
                break;
            }
            row.push_back(value);
        }

        if (numeric && !row.empty()) {
            result.numericData.push_back(std::move(row));
        } else {
            result.headerLines.push_back(line);
        }
    }

    if (result.numericData.empty()) {
        throw std::runtime_error("Blade file does not contain numeric data: " + bladeFilePath.string());
    }

    return result;
}

std::vector<double> HF_Model::extractColumn(const std::vector<std::vector<double>>& data, std::size_t column) {
    std::vector<double> columnData;
    columnData.reserve(data.size());
    for (const auto& row : data) {
        if (column >= row.size()) {
            throw std::runtime_error("Column index out of range while extracting data.");
        }
        columnData.push_back(row[column]);
    }
    return columnData;
}

std::vector<double> HF_Model::gatherByIndices(
    const std::vector<double>& values,
    const std::vector<std::size_t>& indices) {
    std::vector<double> subset;
    subset.reserve(indices.size());
    for (const auto idx : indices) {
        if (idx >= values.size()) {
            throw std::runtime_error("Index out of bounds while gathering control points.");
        }
        subset.push_back(values[idx]);
    }
    return subset;
}

std::vector<double> HF_Model::interpolateLinearProfile(
    const std::vector<double>& x,
    double xStart,
    double xEnd,
    double valueStart,
    double valueEnd) {
    if (std::abs(xEnd - xStart) < std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error("Invalid interpolation bounds (division by zero).");
    }
    std::vector<double> result;
    result.reserve(x.size());
    for (double xi : x) {
        const double t = (xi - xStart) / (xEnd - xStart);
        result.push_back(valueStart + t * (valueEnd - valueStart));
    }
    return result;
}

// compute the HF model output
std::pair<double, double> HF_Model::computeFT(const std::vector<double>& xt) {

    const auto& dataBladeBaseline = bladeData.numericData;
    const auto& headerLines = bladeData.headerLines;

    return computeCPWithTorsionDOF(
                xt,
                xcTheta,
                xs,
                torsionStartID - 1,
                dataBladeBaseline,
                headerLines,
                twistID - 1,
                numberBladeElements,
                rotCpChannel,
                projectRoot,
                aerodynExecutable,
                aerodynCaseRoot);

}

void HF_Model::saveDataOnFile(
    const std::filesystem::path& outputPath) {
    std::ofstream file(outputPath);
    if (!file) {
        throw std::runtime_error("Unable to write data model file: " + outputPath.string());
    }
    file << "xc_theta\n";
    for (double value : xcTheta) {
        file << value << ' ';
    }
    file << '\n';
}


void HF_Model::loadParametersFromConfigFileHF(const std::string& filename) {
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open the configuration file: " << filename << std::endl;
        return;
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
                if (key == "aerodyn_path") aerodynExecutable = value;
                else if (key == "blade_file_name") bladeFileName = value;
                else if (key == "twistID") twistID = std::stoi(value);
                else if (key == "chordID") chordID = std::stoi(value);
                else if (key == "spanID") spanID = std::stoi(value);
                else if (key == "torsionStartID") torsionStartID = std::stoi(value);
                else if (key == "rotCpChannel") rotCpChannel = std::stoi(value);
                        
                //Variables               
                else if (key == "CP_indices") {
                    int elements = std::stoi(value);
                    std::string line;
                    std::getline(file, line);
                    std::istringstream lineStream(line);
                    std::string value;
                    if(elements>0){
                        while (std::getline(lineStream, value, ',')) {
                            CP_indices.push_back(std::stoi(value));
                        }
                    }
                }
                else if( key == "UB") {
                    int elements = std::stoi(value);
                    std::string line;
                    std::getline(file, line);
                    std::istringstream lineStream(line);
                    std::string value;
                    if(elements>0){
                        while (std::getline(lineStream, value, ',')) {
                            ub.push_back(std::stod(value));
                        }
                    }
                }
                else if (key == "LB") {
                    int elements = std::stoi(value);
                    std::string line;
                    std::getline(file, line);
                    std::istringstream lineStream(line);
                    std::string value;
                    if(elements>0){
                        while (std::getline(lineStream, value, ',')) {
                            lb.push_back(std::stod(value));
                        }
                    }
                }
            }
        }
    }

    file.close();

    // Check that all required parameters have been set
    bool valid = true;
    if (aerodynExecutable.empty()) {
        std::cerr << "Error: aerodyn_path is missing in the configuration file." << std::endl;
        valid = false;  
    }
    if (bladeFileName.empty()) {
        std::cerr << "Error: blade_file_name is missing in the configuration file." << std::endl;
        valid = false;
    }
    if (twistID <= 0) {
        std::cerr << "Error: twistID must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (chordID <= 0) {
        std::cerr << "Error: chordID must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (spanID <= 0) {
        std::cerr << "Error: spanID must be a positive integer in the configuration file." << std::endl;
        valid = false;
    }
    if (torsionStartID <= 0) {
        std::cerr << "Error: torsionStartID must be a positive integer in the configuration file." << std::endl;
        valid = false;  
    }
    if (rotCpChannel <= 0) {
        std::cerr << "Error: rotCpChannel must be a positive integer in the configuration file." << std::endl;
        valid = false;  
    }
    if (CP_indices.empty()) {
        std::cerr << "Error: CP_indices must contain at least one index in the configuration file." << std::endl;
        valid = false; 
    }else {
        std::cout << "Control point indices: ";
        for (std::size_t idx : CP_indices) {
            std::cout << idx << ' ';
        }
        std::cout << std::endl;
    }
    if (ub.empty()) {
        std::cerr << "Error: UB must contain at least one value in the configuration file." << std::endl;
        valid = false; 
    }else {
        std::cout << "Upper bounds: ";
        for (double value : ub) {
            std::cout << value << ' ';
        }
        std::cout << std::endl;
    }
    if (lb.empty()) {
        std::cerr << "Error: LB must contain at least one value in the configuration file." << std::endl;
        valid = false; 
    }else {
        std::cout << "Lower bounds: ";
        for (double value : lb) {
            std::cout << value << ' ';  
        }
        std::cout << std::endl;
    }
    if (ub.size() != lb.size()) {
        std::cerr << "Error: UB and LB must have the same number of values in the configuration file." << std::endl;
        valid = false; 
    }
    if (!valid) {
        throw std::runtime_error(
            "Invalid or incomplete configuration file: " + filename);       
        }
    

}

#include "read_fast_binary.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class FileFmtID : int16_t {
    WithTime = 1,
    WithoutTime = 2,
    NoCompressWithoutTime = 3,
    ChanLen_In = 4
};

template <typename T>
T readValue(std::ifstream& stream) {
    T value{};
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!stream) {
        throw std::runtime_error("Failed to read FAST binary file.");
    }
    return value;
}

std::string readFixedString(std::ifstream& stream, std::size_t length) {
    std::string buffer(length, '\0');
    stream.read(buffer.data(), static_cast<std::streamsize>(length));
    if (!stream) {
        throw std::runtime_error("Failed to read FAST binary string.");
    }
    const auto end = buffer.find_last_not_of(' ');
    if (end == std::string::npos) {
        return std::string{};
    }
    return buffer.substr(0, end + 1);
}

}  // namespace

FASTBinaryResult readFASTbinary(const std::string& fileName) {
    std::ifstream input(fileName, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Unable to open FAST binary file: " + fileName);
    }

    const auto fileId = static_cast<FileFmtID>(readValue<int16_t>(input));

    int16_t lenName = 10;
    if (fileId == FileFmtID::ChanLen_In) {
        lenName = readValue<int16_t>(input);
    }

    const int32_t numOutChans = readValue<int32_t>(input);
    const int32_t numTimeSteps = readValue<int32_t>(input);

    double timeScale = 1.0;
    double timeOffset = 0.0;
    double timeOut1 = 0.0;
    double timeIncrement = 0.0;

    if (fileId == FileFmtID::WithTime) {
        timeScale = readValue<double>(input);
        timeOffset = readValue<double>(input);
    } else {
        timeOut1 = readValue<double>(input);
        timeIncrement = readValue<double>(input);
    }

    std::vector<float> columnScale(static_cast<std::size_t>(numOutChans), 1.0f);
    std::vector<float> columnOffset(static_cast<std::size_t>(numOutChans), 0.0f);

    if (fileId != FileFmtID::NoCompressWithoutTime) {
        for (int32_t i = 0; i < numOutChans; ++i) {
            columnScale[static_cast<std::size_t>(i)] = readValue<float>(input);
        }
        for (int32_t i = 0; i < numOutChans; ++i) {
            columnOffset[static_cast<std::size_t>(i)] = readValue<float>(input);
        }
    }

    const int32_t descLength = readValue<int32_t>(input);
    std::string description(descLength, '\0');
    if (descLength > 0) {
        input.read(description.data(), descLength);
        if (!input) {
            throw std::runtime_error("Failed to read FAST description text.");
        }
    }

    const std::size_t numChannelsPlusTime = static_cast<std::size_t>(numOutChans) + 1;

    std::vector<std::string> channelNames(numChannelsPlusTime);
    for (std::size_t i = 0; i < numChannelsPlusTime; ++i) {
        channelNames[i] = readFixedString(input, static_cast<std::size_t>(lenName));
    }

    std::vector<std::string> channelUnits(numChannelsPlusTime);
    for (std::size_t i = 0; i < numChannelsPlusTime; ++i) {
        channelUnits[i] = readFixedString(input, static_cast<std::size_t>(lenName));
    }

    std::vector<int32_t> packedTime;
    if (fileId == FileFmtID::WithTime) {
        packedTime.resize(static_cast<std::size_t>(numTimeSteps));
        input.read(reinterpret_cast<char*>(packedTime.data()),
                   static_cast<std::streamsize>(packedTime.size() * sizeof(int32_t)));
        if (!input) {
            throw std::runtime_error("Failed to read FAST time data.");
        }
    }

    const std::size_t numPoints = static_cast<std::size_t>(numOutChans) *
                                  static_cast<std::size_t>(numTimeSteps);

    std::vector<double> packedData(numPoints, 0.0);
    if (fileId == FileFmtID::NoCompressWithoutTime) {
        input.read(reinterpret_cast<char*>(packedData.data()),
                   static_cast<std::streamsize>(numPoints * sizeof(double)));
    } else {
        std::vector<int16_t> temp(numPoints, 0);
        input.read(reinterpret_cast<char*>(temp.data()),
                   static_cast<std::streamsize>(temp.size() * sizeof(int16_t)));
        if (!input) {
            throw std::runtime_error("Failed to read FAST packed channel data.");
        }
        for (std::size_t i = 0; i < numPoints; ++i) {
            packedData[i] = static_cast<double>(temp[i]);
        }
    }
    if (!input) {
        throw std::runtime_error("Failed to read FAST packed channel data.");
    }

    FASTBinaryResult result;
    result.fileId = static_cast<int16_t>(fileId);
    result.description = description;
    result.channelNames = std::move(channelNames);
    result.channelUnits = std::move(channelUnits);
    result.channels.assign(static_cast<std::size_t>(numTimeSteps),
                           std::vector<double>(numChannelsPlusTime, 0.0));

    for (int32_t it = 0; it < numTimeSteps; ++it) {
        for (int32_t ic = 0; ic < numOutChans; ++ic) {
            const std::size_t idx = static_cast<std::size_t>(ic + numOutChans * it);
            const double scaled = (packedData[idx] - columnOffset[static_cast<std::size_t>(ic)]) /
                                  columnScale[static_cast<std::size_t>(ic)];
            result.channels[static_cast<std::size_t>(it)][static_cast<std::size_t>(ic + 1)] = scaled;
        }
    }

    if (fileId == FileFmtID::WithTime) {
        for (int32_t it = 0; it < numTimeSteps; ++it) {
            const double time = (static_cast<double>(packedTime[static_cast<std::size_t>(it)]) - timeOffset) /
                                timeScale;
            result.channels[static_cast<std::size_t>(it)][0] = time;
        }
    } else {
        for (int32_t it = 0; it < numTimeSteps; ++it) {
            result.channels[static_cast<std::size_t>(it)][0] =
                timeOut1 + timeIncrement * static_cast<double>(it);
        }
    }

    return result;
}

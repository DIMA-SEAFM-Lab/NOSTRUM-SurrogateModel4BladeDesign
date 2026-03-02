#pragma once

#include <cstdint>
#include <string>
#include <vector>

/*
C++ simplified translation of the readFASTbinary function.
*/

struct FASTBinaryResult {
    std::vector<std::vector<double>> channels;
    std::vector<std::string> channelNames;
    std::vector<std::string> channelUnits;
    int16_t fileId;
    std::string description;
};

FASTBinaryResult readFASTbinary(const std::string& fileName);

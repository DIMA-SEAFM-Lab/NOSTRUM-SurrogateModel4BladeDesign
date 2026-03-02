cmake_minimum_required (VERSION 3.8)

project ("NOSTRUM-SurrogateModelGeneration" LANGUAGES C CXX)

# Specify the CMake build type as Release if it is not specified on the command line
if(NOT CMAKE_BUILD_TYPE)   
  set(CMAKE_BUILD_TYPE Release)
  #set(CMAKE_BUILD_TYPE Debug)
endif()

# Get the user's home directory
set(HOME_DIR $ENV{HOME})

# Find MPI package
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Specify the MPI compilers if necessary
# These lines can be commented out if find_package(MPI) sets them automatically
set(CMAKE_C_COMPILER ${MPI_C_COMPILER})
set(CMAKE_CXX_COMPILER ${MPI_CXX_COMPILER})

# Set compiler optimization flags for Release mode
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

add_definitions(-Wall)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Set the source and include directories
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set the source and include directories
set(SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src")
set(INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Collect all source files from the "src" directory
file(GLOB_RECURSE SOURCES ${SOURCE_DIR}/*.cpp)

# Add the include directory to the include path
include_directories(${INCLUDE_DIR} /usr/include/)

# Find BLAS and LAPACK packages
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Add source to this project's executable.
add_executable(TrainSurrogateModel ${SOURCES})

# Link BLAS, LAPACK, METIS, and MPI libraries
target_link_libraries(TrainSurrogateModel ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${MPI_C_LIBRARIES}) 

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET TrainSurrogateModel PROPERTY CXX_STANDARD 20)
endif()

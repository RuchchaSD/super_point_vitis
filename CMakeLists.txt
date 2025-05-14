# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SuperPointProject)

# Set output directories for libraries and executables
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message("Build type: " ${CMAKE_BUILD_TYPE})


if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -ffast-math -ftree-vectorize")
    message(STATUS "Building with aggressive optimization")
endif()


option(PROFILE "Build with profiling" OFF)
if(PROFILE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
endif()

option(BUILD_EXAMPLES "Build examples" ON)

set(GLOG_PATH "/root/glog_build/")
set(CMAKE_PREFIX_PATH ${GLOG_PATH} ${CMAKE_PREFIX_PATH})
find_package(Glog REQUIRED HINTS ${GLOG_PATH})

# Find packages
find_package(OpenCV REQUIRED)
find_package(VART 3.5.0 REQUIRED)
find_package(UNILOG 3.5.0 REQUIRED)
find_package(Glog 0.5.0 REQUIRED)
find_package(Threads REQUIRED)
find_package(XIR 3.5.0 REQUIRED)
find_package(vitis_ai_library REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "VART version: ${VART_VERSION}")
message(STATUS "UNILOG version: ${UNILOG_VERSION}")
message(STATUS "GLOG version: ${GLOG_VERSION}")
message(STATUS "XIR version: ${XIR_VERSION}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)

set(MODEL
    src/SuperPointFast.cpp
)

# Create SuperPointLib library
add_library(SuperPointLib SHARED
    ${MODEL}
)

# Set output name and properties for SuperPointLib
set_target_properties(SuperPointLib PROPERTIES
    OUTPUT_NAME "SuperPointLib"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
)

target_link_libraries(SuperPointLib
    ${OpenCV_LIBS}
    vart::runner
    vart::util
    xir::xir
    unilog::unilog
    glog::glog
    Threads::Threads
    vitis_ai_library::dpu_task
    vitis_ai_library::math
)

if(BUILD_EXAMPLES)
    message(STATUS "Building FeatureIO library - used for extract_features example")

    add_library(FeatureIO SHARED
        src/FeatureIO.cpp
    )

    set_target_properties(FeatureIO PROPERTIES
        OUTPUT_NAME "FeatureIO"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
    )

    target_link_libraries(FeatureIO
        ${OpenCV_LIBS}
    )

    target_include_directories(FeatureIO PUBLIC
        ${OpenCV_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
    )

    message(STATUS "Building examples")
    # Create executables
    add_executable(demo src/demo_superpoint.cpp)
    add_executable(demo_continuous src/demo_continuous.cpp)
    add_executable(extract_features src/extract_features.cpp)

    # Set output properties for executables
    set_target_properties(demo demo_continuous extract_features PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
    )

    # Link libraries for executables
    target_link_libraries(demo SuperPointLib)
    target_link_libraries(demo_continuous SuperPointLib)
    target_link_libraries(extract_features SuperPointLib FeatureIO)
endif()

# CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SuperPointProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")

set(GLOG_PATH "/root/glog_build/")

set(CMAKE_PREFIX_PATH ${GLOG_PATH} ${CMAKE_PREFIX_PATH})
find_package(Glog REQUIRED HINTS ${GLOG_PATH})

# Find packages
# set(OpenCV_DIR "/root/opencv_build/opencv/build") 
find_package(OpenCV REQUIRED)
# find_package(OpenCV 4.6.0 REQUIRED)
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
    # src/SuperPoint.cpp 
    # src/SuperPointSingleImp.cpp
    src/SuperPointFast.cpp
)

add_library(FeatureIO SHARED
    src/FeatureIO.cpp
)

# Link libraries for the FeatureIO library
target_link_libraries(FeatureIO
    ${OpenCV_LIBS}
)

# Specify include directories for the FeatureIO library
target_include_directories(FeatureIO PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)


# Add the source files to the library
add_library(SuperPointLib SHARED
    ${MODEL}
)

# Link libraries for the SuperPointLib library
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


# add_executable(demo
#     src/demo_superpoint.cpp
# )

# # Add the continuous demo executable
# add_executable(demo_continuous
#     src/demo_continuous.cpp
# )

# # Add the feature extraction executable
# add_executable(extract_features
#     src/extract_features.cpp
# )

# # add_executable(test_performance
# #     ${MODEL}
# #     src/test_performance_superpoint.cpp
# # )

# # Link libraries for the demo executable
# target_link_libraries(demo
#     SuperPointLib
# )

# # Link libraries for the continuous demo executable
# target_link_libraries(demo_continuous
#     SuperPointLib
# )

# # Link libraries for the feature extraction executable
# target_link_libraries(extract_features
#     SuperPointLib
#     FeatureIO
# )


## testing cruz
add_executable(superpoint_benchmark
    src/superpoint_benchmark.cpp
)

target_link_libraries(superpoint_benchmark
    SuperPointLib
    ${OpenCV_LIBS}
    glog::glog
)
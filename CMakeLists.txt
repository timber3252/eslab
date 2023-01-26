cmake_minimum_required(VERSION 3.10)
project(eslab)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS_DEBUG "-fPIC -O0 -g -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "-fPIC -O2 -Wall")

add_definitions(-DENABLE_DVPP_INTERFACE)
set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/acllib/lib64/stub/")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

if (${CMAKE_CXX_COMPILER} MATCHES "aarch")
    set(ARCH "arm")
else ()
    set(ARCH "x86")
endif()

# Header path
include_directories(
    /home/HwHiAiUser/ascend_ddk/arm/include/opencv4/
    /home/HwHiAiUser/ascend_ddk/arm/include/
    ${INC_PATH}/acllib/include/
    /usr/include/
    /usr/local/include/alibabacloud/
    ../inc/
    header
)

# add host lib path
link_directories(
    ${LIB_PATH}
    /home/HwHiAiUser/ascend_ddk/arm/lib/
    ${INC_PATH}/atc/lib64
    /home/HwHiAiUser/Ascend/driver
    /usr/bin/
    /usr/local/bin/
    /usr/local/lib/
    /usr/lib/aarch64-linux-gnu/
)

add_executable(eslab main.cpp
    model/face_detect.cpp model/face_detect.hpp model/face_feature_mask.cpp model/face_feature_mask.hpp
    ascend/model.cpp ascend/model.hpp ascend/acl.cpp ascend/acl.hpp ascend/acl_device.cpp ascend/acl_device.hpp ascend/face_feature_train_mean.hpp ascend/face_feature_train_std.hpp
    util/image.cpp util/image.hpp)

target_link_libraries(eslab ascendcl atlasutil acl_dvpp stdc++ opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio protobuf presenteragent avcodec swresample)



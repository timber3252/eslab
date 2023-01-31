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
    face_recognition/face_detect.cpp face_recognition/face_detect.hpp face_recognition/face_feature_mask.cpp face_recognition/face_feature_mask.hpp face_recognition/face_feature_train_mean.hpp face_recognition/face_feature_train_std.hpp face_recognition/face_utils.cpp face_recognition/face_utils.hpp
    ascend/model.cpp ascend/model.hpp ascend/acl.cpp ascend/acl.hpp ascend/acl_device.cpp ascend/acl_device.hpp ascend/presenter.cpp ascend/presenter.hpp ascend/pca_9557.cpp ascend/pca_9557.hpp ascend/ssd_1306.cpp ascend/ssd_1306.hpp ascend/button.cpp ascend/button.hpp
    util/image.cpp util/image.hpp face_recognition/face_recognition.cpp face_recognition/face_recognition.hpp util/math.cpp util/math.hpp
    ascend/3rdparty/font.hpp ascend/3rdparty/i2c.cpp ascend/3rdparty/i2c.hpp ascend/3rdparty/pca9557.cpp ascend/3rdparty/pca9557.hpp ascend/3rdparty/ssd1306.cpp ascend/3rdparty/ssd1306.hpp ascend/3rdparty/key.cpp ascend/3rdparty/key.hpp ascend/3rdparty/led.cpp ascend/3rdparty/led.hpp ascend/3rdparty/multi_button.cpp ascend/3rdparty/multi_button.hpp)

target_link_libraries(eslab ascendcl atlasutil acl_dvpp stdc++ opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_videoio protobuf presenteragent avcodec swresample pthread)



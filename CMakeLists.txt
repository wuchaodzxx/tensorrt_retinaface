cmake_minimum_required(VERSION 3.10)
project(onnx_tensorrt)

add_definitions("-std=c++1y")


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
#set(CMAKE_BUILD_TYPE Release)


# Enable compiler warnings
if (CMAKE_COMPILER_IS_GNUCC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
endif ()

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(CUDA REQUIRED)
find_package(CUDNN QUIET)
find_package(Protobuf REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system python regex filesystem)
find_package(gflags REQUIRED)
#find_package(glog REQUIRED)


# TensorRT
set(TENSORRT_ROOT $ENV{HOME}/TensorRT/)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES include)
MESSAGE(STATUS "Found TensorRT headers at ${TENSORRT_INCLUDE_DIR}")
find_library(TENSORRT_LIBRARY_INFER nvinfer
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_PARSER nvparsers
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNX_PARSER nvonnxparser
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_LIBRARY_ONNX_PARSER_RUNTIME nvonnxparser_runtime
        HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
set(TENSORRT_LIBRARY ${TENSORRT_LIBRARY_INFER} ${TENSORRT_LIBRARY_INFER_PLUGIN} ${TENSORRT_LIBRARY_PARSER} ${TENSORRT_LIBRARY_ONNX_PARSER} ${TENSORRT_LIBRARY_ONNX_PARSER_RUNTIME})
MESSAGE(STATUS "Find TensorRT libs at ${TENSORRT_LIBRARY}")
find_package_handle_standard_args(
        TENSORRT DEFAULT_MSG TENSORRT_INCLUDE_DIR TENSORRT_LIBRARY)

# CUDA Configuration for Pascal and Volta/Turing cards
if (CUDA_FOUND)
    list(APPEND GPU_ARCHS 61 70)
    set(CUDA_VERBOSE_BUILD ON)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11")
    # Generate SASS for each architecture
    foreach (arch ${GPU_ARCHS})
        set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
    endforeach ()
    # Generate PTX for the last architecture
    list(GET GPU_ARCHS -1 LATEST_GPU_ARCH)
    set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_GPU_ARCH},code=compute_${LATEST_GPU_ARCH}")

    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -cudart static \
    -lineinfo \
    -g \
    --expt-extended-lambda \
    ${GENCODES} \
")

    # Specify the cuda host compiler to use the same compiler as cmake.
    set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
    include_directories("/usr/local/cuda/include")

    if (TENSORRT_FOUND)
        include_directories(${TENSORRT_INCLUDE_DIR})
        message(STATUS "TensorRT found: " ${TENSORRT_LIBRARY})

        include_directories(include/)

        # cuda_add_library(upsample_layer_lib SHARED
        #         src/upsample_layer.cu
        #         src/upsample_layer.cc
        #         include/upsample_layer.h)
        # cuda_add_library(yolo_layer_lib SHARED
        #         src/yolo_layer.cu
        #         include/yolo_layer.h)

         add_library(tensorrt_engine_lib SHARED
                 include/all_plugin_factory.h
                 include/tensorrt_engine.h
                 src/tensorrt_engine.cc
                 src/image.cc
                 include/image.h
                 include/batch_stream.h
                 include/entropy_calibrator.h)
         target_link_libraries(tensorrt_engine_lib
                 ${TENSORRT_LIBRARY}
                 ${upsample_layer_lib}
                 ${yolo_layer_lib})

        add_subdirectory(examples)
    else ()
        message(FATAL_ERROR "TensorRT not found, nothing will build. installed it into /home/user/tensorrt5.1.5")
    endif (TENSORRT_FOUND)
else ()
    message(FATAL_ERROR "CUDA not found, do nothing.")
endif ()
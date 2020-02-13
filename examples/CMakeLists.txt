add_executable(trt_engine_retinaface trt_engine_retinaface.cc)
target_link_libraries(trt_engine_retinaface
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        tensorrt_engine_lib
        ${OpenCV_LIBS}
        thor)

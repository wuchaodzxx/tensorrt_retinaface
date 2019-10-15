add_executable(demo_trt_refinenet_seg demo_trt_refinenet_seg.cc trt_engine_yolov3.cc)
target_link_libraries(demo_trt_refinenet_seg
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        tensorrt_engine_lib
        ${OpenCV_LIBS}
        thor)


add_executable(trt_engine_refinenetseg trt_engine_refinenetseg.cc)
target_link_libraries(trt_engine_refinenetseg
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        tensorrt_engine_lib
        ${OpenCV_LIBS}
        thor)


add_executable(trt_engine_retinaface_eigen trt_engine_retinaface_eigen.cc)
target_link_libraries(trt_engine_retinaface_eigen
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        tensorrt_engine_lib
        ${OpenCV_LIBS}
        thor)


add_executable(trt_engine_retinaface trt_engine_retinaface.cc)
target_link_libraries(trt_engine_retinaface
        ${CUDA_LIBRARIES}
        ${CUDNN_LIBRARY}
        ${TENSORRT_LIBRARY}
        tensorrt_engine_lib
        ${OpenCV_LIBS}
        thor)

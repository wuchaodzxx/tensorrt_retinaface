#include <assert.h>
#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "NvInfer.h"
#include "NvOnnxParser.h"

#include <opencv2/opencv.hpp>
#include "image.h"
#include <sys/time.h>
#include <iterator>
#include <tensorrt_engine.h>
#include "thor/timer.h"
#include "thor/logging.h"
#include "thor/os.h"

#include "../include/batch_stream.h"
#include "../include/tensorrt_utils.h"
#include "../include/entropy_calibrator.h"


/**
 *
 * Inference on a new onnx converted trt model
 * using standalone TensorRT engine
 *
 */

using namespace thor::log;
using namespace nvinfer1;
static tensorrt::Logger gLogger;

std::stringstream gieModelStream;
//static const int INPUT_H = 678;
//static const int INPUT_W = 1024;
static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 114688;
static const int INPUT_SIZE = 786432;

static int gUseDLACore{-1};
const char* INPUT_BLOB_NAME = "0";
// const char* OUTPUT_BLOB_NAME = "prob";

void doInference(IExecutionContext& context, float* input, float* output,
                 int batchSize) {
  const ICudaEngine& engine = context.getEngine();
  assert(engine.getNbBindings() == 2);
  void* buffers[2];
  int inputIndex, outputIndex1, outputIndex2, outputIndex3, outputIndex;
  for (int b = 0; b < engine.getNbBindings(); ++b) {
    if (engine.bindingIsInput(b))
      inputIndex = b;
    else
      outputIndex = b;
  }

  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream,nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        batchSize * OUTPUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

int run(int argc, char** argv) {
  string engine_f = argv[1];
  string data_f = argv[2];
  string mode = "fp32";
  string batch_f;
  if (argc >= 4) {
    mode = argv[3];
    batch_f = argv[4];
  }

  LOG(INFO) << "loading from engine file: " << engine_f;
  tensorrt::TensorRTEngine trtEngine(engine_f);
  ICudaEngine* engine = trtEngine.getEngine();
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  LOG(INFO) << "inference directly from engine.";

  cv::VideoCapture cap(data_f);
  if (!cap.isOpened()) {
    std::cout << "Error: video-stream can't be opened! \n";
    return 1;
  }

  cv::Mat frame;
  float prob[OUTPUT_SIZE];
  float* data;
  float fps = 0;

  cv::Mat out;
  out.create(128, 128, CV_32FC(7));
  cv::Mat real_out;
  real_out.create(512, 512, CV_32FC(7));
  cv::Mat real_out_;
  real_out_.create(512, 512, CV_8UC3);
  int w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  int h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

  thor::Timer timer(20);
  timer.on();

  LOG(INFO) << "start to inference.";
  while (1) {
    cap >> frame;
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
    cv::Mat dst =
        cv::Mat::zeros(512, 512, CV_32FC3);  //新建一张512x512尺寸的图片Mat
    cv::resize(img, dst, dst.size());
    LOG(INFO) << dst.at<cv::Vec3f>(224, 556);
    LOG(INFO) << img.at<cv::Vec3b>(224, 556);
    data = normal(dst);

    timer.lap();
    doInference(*context, data, prob, 1);  // chw
    out = read2mat(prob, out);
    cv::resize(out, real_out, real_out.size());
    real_out_ = map2threeunchar(real_out, real_out_);
    cv::Mat mask_resized;
    cv::resize(real_out_, mask_resized, cv::Size(w, h));

    double cost = timer.lap();
    cout << "fps: " << 1 / cost << endl;

    // this inference is very fast, but these codes need optimization
    cv::Mat combined;
    cv::addWeighted(frame, 0.7, mask_resized, 0.7, 0.6, combined);
    cv::putText(combined, "FPS: " + to_string(1 / cost), cv::Point(10, 20), cv::FONT_HERSHEY_TRIPLEX, .6,
                cv::Scalar(0, 255, 0));

    cv::imshow("ori", combined);
    std::free(data);
    if (cv::waitKey(10) == 27) break;
  }

  cv::destroyAllWindows();
  cap.release();
  // destroy the engine
  context->destroy();
  engine->destroy();
  LOG(INFO) << "shut down!";
  return 0;
}

int main(int argc, char** argv) {
  // trt_file data_file
  run(argc, argv);
  return 0;
}

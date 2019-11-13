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
#include "thor/structures.h"

#include "../include/batch_stream.h"
#include "../include/tensorrt_utils.h"
#include "../include/entropy_calibrator.h"

// we will using eigen so something
#include "eigen3/Eigen/Eigen"
#include "eigen3/Eigen/Core"


/**
 *
 * Inference on a new onnx converted trt model
 * using standalone TensorRT engine
 *
 */

using namespace thor::log;
using namespace nvinfer1;
using namespace Eigen;

static tensorrt::Logger gLogger;
std::stringstream gieModelStream;
static const int INPUT_H = 678;
static const int INPUT_W = 1024;
// once image size certain, priors_n also certain, how does this calculated?
static const int priors_n = 28672;

// BGR order
static const float kMeans[3] = {104.f, 117.f, 123.f};
// using for Priors
static const vector<vector<int>> min_sizes = {{16, 32}, {64, 128}, {256, 512}};
static const vector<int> steps = {8, 16, 32};
const char *INPUT_BLOB_NAME = "0";

struct Box {
  float x1;
  float y1;
  float x2;
  float y2;
};

struct Landmark {
  float x[5];
  float y[5];
};

struct FaceInfo {
  float score;
  Box box;
  Landmark landmark;
};

vector<Box> createPriors(vector<vector<int>> min_sizes, vector<int> steps, cv::Size img_size) {
  vector<Box> anchors;
  // 8, 16, 32
  for (int j = 0; j < steps.size(); ++j) {
    int step = steps[j];
    // featuremap sizes
    int fm_h = ceil(INPUT_H * 1.0 / step);
    int fm_w = ceil(INPUT_W * 1.0 / step);
    vector<int> min_sizes_k = min_sizes[j];
    // iter one featuremap
    for (int fi = 0; fi < fm_h; ++fi) {
      for (int fj = 0; fj < fm_w; ++fj) {
        for (int k = 0; k < min_sizes_k.size(); ++k) {
          int min_size = min_sizes_k[k];
          float s_kx = (float) min_size / INPUT_W;
          float s_ky = (float) min_size / INPUT_H;
          float cx = (float) ((fj + 0.5) * step) / INPUT_W;
          float cy = (float) ((fi + 0.5) * step) / INPUT_H;

          Box rect;
          rect.x1 = cx;
          rect.y1 = cy;
          rect.x2 = s_kx;
          rect.y2 = s_ky;
          anchors.emplace_back(rect);
        }
      }
    }
  }
  for (int kI = 0; kI < 5; ++kI) {
    LOG(INFO) << anchors[kI].x1 << " " << anchors[kI].y1 << " " << anchors[kI].x2 << " " << anchors[kI].y2;
  }
  return anchors;
}

bool CompareBBox(const FaceInfo &a, const FaceInfo &b) {
  return a.score > b.score;
}

std::vector<FaceInfo> nms(std::vector<FaceInfo> &bboxes,
                          float threshold) {
  std::vector<FaceInfo> bboxes_nms;
  std::sort(bboxes.begin(), bboxes.end(), CompareBBox);
  int32_t select_idx = 0;
  int32_t num_bbox = static_cast<int32_t>(bboxes.size());
  std::vector<int32_t> mask_merged(num_bbox, 0);
  bool all_merged = false;

  while (!all_merged) {
    while (select_idx < num_bbox && mask_merged[select_idx] == 1) select_idx++;
    if (select_idx == num_bbox) {
      all_merged = true;
      continue;
    }
    bboxes_nms.push_back(bboxes[select_idx]);
    mask_merged[select_idx] = 1;
    Box select_bbox = bboxes[select_idx].box;
    float area1 = static_cast<float>((select_bbox.x2 - select_bbox.x1 + 1) *
        (select_bbox.y2 - select_bbox.y1 + 1));
    float x1 = static_cast<float>(select_bbox.x1);
    float y1 = static_cast<float>(select_bbox.y1);
    float x2 = static_cast<float>(select_bbox.x2);
    float y2 = static_cast<float>(select_bbox.y2);

    select_idx++;
    for (int32_t i = select_idx; i < num_bbox; i++) {
      if (mask_merged[i] == 1) continue;

      Box &bbox_i = bboxes[i].box;
      float x = std::max<float>(x1, static_cast<float>(bbox_i.x1));
      float y = std::max<float>(y1, static_cast<float>(bbox_i.y1));
      float w = std::min<float>(x2, static_cast<float>(bbox_i.x2)) - x + 1;  //<- float 型不加1
      float h = std::min<float>(y2, static_cast<float>(bbox_i.y2)) - y + 1;
      if (w <= 0 || h <= 0) continue;

      float area2 = static_cast<float>((bbox_i.x2 - bbox_i.x1 + 1) *
          (bbox_i.y2 - bbox_i.y1 + 1));
      float area_intersect = w * h;

      if (static_cast<float>(area_intersect) /
          (area1 + area2 - area_intersect) >
          threshold) {
        mask_merged[i] = 1;
      }
    }
  }
  return bboxes_nms;
}

Box decodeBox(Box anchor, cv::Vec4f regress, const int w, const int h) {
  Box rect;
  rect.x1 = w * (anchor.x1 + regress[0] * 0.1 * anchor.x2);
  rect.y1 = h * (anchor.y1 + regress[1] * 0.1 * anchor.y2);
  rect.x2 = w * (anchor.x2 * exp(regress[2] * 0.2));
  rect.y2 = h * (anchor.y2 * exp(regress[3] * 0.2));
  rect.x1 -= (rect.x2 / 2);
  rect.y1 -= (rect.y2 / 2);
  rect.x2 += rect.x1;
  rect.y2 += rect.y1;
  return rect;
}

Landmark decodeLandmark(Box anchor, Landmark facePts, const int w, const int h) {
  Landmark landmark;
  for (int k = 0; k < 5; ++k) {
    landmark.x[k] = w * (anchor.x1 + facePts.x[k] * 0.1 * anchor.x2);
    landmark.y[k] = h * (anchor.y1 + facePts.y[k] * 0.1 * anchor.y2);
  }
  return landmark;
}

vector<FaceInfo> doPostProcess(float *out_box, float *out_landmark, float *out_conf,
    const vector<Box> &priors, cosnt int w, const in h, float nms_threshold) {
  // 28672x4, 28672x2, 28672x10
  vector<FaceInfo> all_faces;
  for (int i = 0; i < priors_n; ++i) {
    // first column is background
    float conf_i = out_conf[2 * i + 1];
    if (conf_i >= 0.8) {
      // only score >= 0.5
      cv::Vec4f regress;
      float dx = out_box[4 * i];
      float dy = out_box[4 * i + 1];
      float dw = out_box[4 * i + 2];
      float dh = out_box[4 * i + 3];
      regress = cv::Vec4f(dx, dy, dw, dh);
      Box box = decodeBox(priors[i], regress, w, h);

      Landmark pts;
      for (size_t k = 0; k < 5; k++) {
        pts.x[k] = out_landmark[i * 10 + k * 2];
        pts.y[k] = out_landmark[i * 10 + k * 2 + 1];
      }
      Landmark landmark = decodeLandmark(priors[i], pts, w, h);
      FaceInfo one_face;
      one_face.box = box;
      one_face.score = conf_i;
      one_face.landmark = landmark;
      all_faces.emplace_back(one_face);
    }
  }
  // do nms here
  all_faces = nms(all_faces, nms_threshold);
  return all_faces;
}

vector<FaceInfo> doInference(IExecutionContext &context, float *input, const vector<Box> &priors, int batchSize,
                             const int w, const int h, float nms_threshold = 0.86) {
  const ICudaEngine &engine = context.getEngine();
  // we have 4 bindings for retinaface
  assert(engine.getNbBindings() == 4);

  void *buffers[4];
  std::vector<int64_t> bufferSize;
  int nbBindings = engine.getNbBindings();
  bufferSize.resize(nbBindings);

  for (int kI = 0; kI < nbBindings; ++kI) {
    nvinfer1::Dims dims = engine.getBindingDimensions(kI);
    nvinfer1::DataType dt = engine.getBindingDataType(kI);
    int64_t totalSize = volume(dims) * 1 * getElementSize(dt);
    bufferSize[kI] = totalSize;
//    LOG(INFO) << "binding " << kI << " nodeName: " << engine.getBindingName(kI) << " total size: " << totalSize;
    CHECK(cudaMalloc(&buffers[kI], totalSize));
  }

  auto out1 = new float[bufferSize[1] / sizeof(float)];
  auto out2 = new float[bufferSize[2] / sizeof(float)];
  auto out3 = new float[bufferSize[3] / sizeof(float)];

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaMemcpyAsync(buffers[0], input, bufferSize[0], cudaMemcpyHostToDevice, stream));
//  context.enqueue(batchSize, buffers, stream,nullptr);
  context.enqueue(1, buffers, stream, nullptr);

  CHECK(cudaMemcpyAsync(out1, buffers[1], bufferSize[1], cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(out2, buffers[2], bufferSize[2], cudaMemcpyDeviceToHost, stream));
  CHECK(cudaMemcpyAsync(out3, buffers[3], bufferSize[3], cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // release the stream and the buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[0]));
  CHECK(cudaFree(buffers[1]));
  CHECK(cudaFree(buffers[2]));
  CHECK(cudaFree(buffers[3]));
  // box, landmark, conf
  // 28672x4, 28672x2, 28672x10
  // out1: 4 box, out2: 2 conf, out3: 10 landmark
  vector<FaceInfo> all_faces = doPostProcess(out1, out2, out3, priors, w, h, nms_threshold);
  return all_faces;
}

int run(int argc, char **argv) {
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
  ICudaEngine *engine = trtEngine.getEngine();
  CheckEngine(engine);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);
  LOG(INFO) << "inference directly from engine.";

  vector<Box> priors = createPriors(min_sizes, steps, cv::Size(INPUT_H, INPUT_W));

  if (thor::os::suffix(data_f) == "mp4") {
    cv::VideoCapture cap(data_f);
    if (!cap.isOpened()) {
      std::cout << "Error: video-stream can't be opened! \n";
      return 1;
    }

    cv::Mat frame;
    float *data;
    int ori_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int ori_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    thor::Timer timer(20);
    timer.on();
    LOG(INFO) << "start to inference on video file: " << data_f;
    while (1) {
      cap >> frame;
      if (frame.empty()) {
        cv::destroyAllWindows();
        cap.release();
        // destroy the engine
        context->destroy();
        engine->destroy();
        LOG(INFO) << "shut down!";
        break;
      }
      cv::Mat resizedImage = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
      cv::resize(frame, resizedImage, cv::Size(INPUT_W, INPUT_H));
      data = HWC2CHW(resizedImage, kMeans);

      timer.lap();
      vector<FaceInfo> all_faces = doInference(*context, data, priors, 1);
      double cost = timer.lap();
      LOG(INFO) << "fps: " << 1 / cost << ", cost: " << cost;

      for (size_t i = 0; i < all_faces.size(); i++) {
        FaceInfo one_face = all_faces[i];
        int x1 = (int) (one_face.box.x1 * ori_w);
        int y1 = (int) (one_face.box.y1 * ori_h);
        int x2 = (int) (one_face.box.x2 * ori_w);
        int y2 = (int) (one_face.box.y2 * ori_h);
        float conf = one_face.score;
        if (conf > 0.5) {
          char conf_str[128];
          sprintf(conf_str, "%.3f", conf);
          cv::putText(frame, conf_str, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 0.6,
                      cv::Scalar(0, 255, 0));
          cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);

          for (size_t j = 0; j < 5; j++) {
            cv::Point2f pt = cv::Point2f(one_face.landmark.x[j] * ori_w,
                                         one_face.landmark.y[j] * ori_h);
            cv::circle(frame, pt, 1, cv::Scalar(0, 255, 0), 2);
          }
        }
      }

      // also we need log FPS here
      char fps_str[128];
      sprintf(fps_str, "FPS: %.3f", 1 / cost);
      cv::putText(frame, fps_str, cv::Point(40, 40), cv::FONT_HERSHEY_COMPLEX, 0.6,
                  cv::Scalar(0, 0, 225));
      std::free(data);
      cv::imshow("RetinaFace TensorRT 1280x960", frame);
      cv::waitKey(1);
    }

  } else {
    // on image
    float *data;
    cv::Mat frame = cv::imread(data_f);
    int ori_w = frame.cols;
    int ori_h = frame.rows;
    cv::Mat resizedImage = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    cv::resize(frame, resizedImage, cv::Size(INPUT_W, INPUT_H));
    data = HWC2CHW(resizedImage, kMeans);

    vector<FaceInfo> all_faces = doInference(*context, data, priors, 1, ori_w, ori_h);

    for (size_t i = 0; i < all_faces.size(); i++) {
      FaceInfo one_face = all_faces[i];
      int x1 = (int) (one_face.box.x1);
      int y1 = (int) (one_face.box.y1);
      int x2 = (int) (one_face.box.x2);
      int y2 = (int) (one_face.box.y2);
      float conf = one_face.score;
      if (conf > 0.5) {
        char conf_str[128];
        sprintf(conf_str, "%.3f", conf);
        cv::putText(frame, conf_str, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 0.6,
                    cv::Scalar(255, 0, 225));
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255));
        for (size_t j = 0; j < 5; j++) {
          cv::Point2f pt = cv::Point2f(one_face.landmark.x[j],
                                       one_face.landmark.y[j]);
          cv::circle(frame, pt, 1, cv::Scalar(0, 255, 0), 2);
        }
      }
    }
    cv::imshow("RetinaFace TensorRT 1280x960", frame);
    cv::waitKey(0);
  }

  return 0;
}

int main(int argc, char **argv) {
  // trt_file data_file
  run(argc, argv);
  return 0;
}

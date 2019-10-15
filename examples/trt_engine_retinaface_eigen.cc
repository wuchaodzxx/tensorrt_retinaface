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
 *
 * NOTE: eigen version is slow, we have tried using eigen
 * to do matrix job, but actually slow in construct the MatrixXf from data*
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
const char* INPUT_BLOB_NAME = "0";


struct anchor_box
{
  float x1;
  float y1;
  float x2;
  float y2;
};

struct FacePts
{
  float x[5];
  float y[5];
};

struct FaceDetectInfo
{
  float score;
  anchor_box rect;
  FacePts pts;
};


void createPriors(vector<vector<int>> min_sizes, vector<int> steps, cv::Size img_size, Eigen::MatrixXf &priors) {
  // create a PriorBox array with shape [28672, 4]
  // tensor([[0.0039, 0.0058, 0.0156, 0.0233],
  //        [0.0039, 0.0058, 0.0312, 0.0466],
  //        [0.0117, 0.0058, 0.0156, 0.0233],
  //        ...,
  //        [0.9531, 1.0015, 0.5000, 0.7453],
  //        [0.9844, 1.0015, 0.2500, 0.3726],
  //        [0.9844, 1.0015, 0.5000, 0.7453]])
//  float* data = new float[priors_n * 4 * sizeof(float)];
  vector<float> data_v;
  // 8, 16, 32
  int i = 0;
  for (int j = 0; j < steps.size(); ++j) {
    int step = steps[j];
    // featuremap sizes
    int fm_h = ceil(INPUT_H*1.0/step);
    int fm_w = ceil(INPUT_W*1.0/step);
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
          data_v.push_back(cx);
          data_v.push_back(cy);
          data_v.push_back(s_kx);
          data_v.push_back(s_ky);
          i++;
        }
      }
    }
  }
  priors = Map<Matrix<float, priors_n, 4, RowMajor> >(data_v.data());
}

bool CompareBBox(const FaceDetectInfo &a, const FaceDetectInfo &b) {
  return a.score > b.score;
}

MatrixXf nms(MatrixXf &bboxes, float threshold) {
  MatrixXf bboxes_nms;
  return bboxes_nms;
}


MatrixXf doPostProcess(float* out_box, float* out_landmark, float* out_conf, Eigen::MatrixXf &priors) {
  // 28672x4, 28672x2, 28672x10
  // decode boxes and landmarks using priors
  // convert box to MatrixXf, decode box and landmarks using priors
  thor::Timer timer(20);
  timer.on();

  MatrixXf m_box = Map<Matrix<float, priors_n, 4, RowMajor>>(out_box);
  MatrixXf m_landmark = Map<Matrix<float, priors_n, 10, RowMajor>>(out_landmark);
  MatrixXf m_conf = Map<Matrix<float, priors_n, 2, RowMajor>>(out_conf);
  LOG(INFO) << "construct time: " << timer.lap();

  cout << m_conf.topRows(5) << endl;

  const float variances[2] = {0.1, 0.2};
  MatrixXf box_left_part = priors.leftCols(2) + priors.rightCols(2).cwiseProduct(m_box.leftCols(2)*variances[0]);
  MatrixXf tmp = m_box.rightCols(2)*variances[1];
  MatrixXf tmpExp = tmp.array().exp();
  MatrixXf box_right_part = priors.rightCols(2).cwiseProduct(tmpExp);
  LOG(INFO) << "prior time: " << timer.lap();

  box_left_part = (box_left_part - box_right_part/2);
  box_right_part = box_right_part+box_left_part;
  MatrixXf box_res(m_conf.rows(), 5);
  box_res << box_left_part, box_right_part, m_conf.rightCols(1);
  LOG(INFO) << "reconstruct time: " << timer.lap();

  cout << box_res.topRows(5) << endl;

  // filter out conf less than 0.5
  VectorXi is_selected = (box_res.col(4).array() > 0.5).cast<int>();
  MatrixXf mat_sel(is_selected.sum(), box_res.cols());
  int rownew = 0;
  for (int i = 0; i < box_res.rows(); ++i) {
    if (is_selected[i]) {       
       mat_sel.row(rownew) = box_res.row(i);
       rownew++;
    }
  }
  LOG(INFO) << "select time: " << timer.lap();
  return mat_sel;
}

MatrixXf doInference(IExecutionContext& context, float* input, Eigen::MatrixXf &priors, int batchSize, float threshold=0.8) {
  double t = (double) cv::getTickCount();

  const ICudaEngine& engine = context.getEngine();
  // we have 4 bindings for retinaface
  assert(engine.getNbBindings() == 4);

  void* buffers[4];
  std::vector<int64_t> bufferSize;
  int nbBindings = engine.getNbBindings();
  bufferSize.resize(nbBindings);

  for (int kI = 0; kI < nbBindings; ++kI) {
    nvinfer1::Dims dims = engine.getBindingDimensions(kI);
    nvinfer1::DataType dt = engine.getBindingDataType(kI);
    int64_t totalSize = volume(dims)*1*getElementSize(dt);
    bufferSize[kI] = totalSize;
//    LOG(INFO) << "binding " << kI << " nodeName: " << engine.getBindingName(kI) << " total size: " << totalSize;
    CHECK(cudaMalloc(&buffers[kI], totalSize));
  }

  auto out1 = new float[bufferSize[1]/ sizeof(float)];
  auto out2 = new float[bufferSize[2]/ sizeof(float)];
  auto out3 = new float[bufferSize[3]/ sizeof(float)];

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));
  CHECK(cudaMemcpyAsync(buffers[0], input, bufferSize[0], cudaMemcpyHostToDevice, stream));
//  context.enqueue(batchSize, buffers, stream,nullptr);
  context.enqueue(1, buffers, stream,nullptr);

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
  // 114688, 57344, 286720
  int64_t candidates_n = bufferSize[3] / (2*sizeof(float));
  MatrixXf all_faces;
  // box got right now, should convert coordinates now
  // out1: 4 box, out2: 2 conf, out3: 10 landmark
  double cost = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
  LOG(INFO) << "fps1: " << 1 / cost << ", cost: " << cost;

  double tic = (double) cv::getTickCount();
  all_faces = doPostProcess(out1, out2, out3, priors);
  cost = ((double) cv::getTickCount() - tic) / cv::getTickFrequency();
  LOG(INFO) << "fps2: " << 1 / cost << ", cost: " << cost;
  return all_faces;
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
  CheckEngine(engine);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  LOG(INFO) << "inference directly from engine.";

  MatrixXf priors;
  createPriors(min_sizes, steps, cv::Size(INPUT_H, INPUT_W), priors);
  // check priors
  cout << priors.topRows(5) << endl;
  cout << priors.bottomRows(5) << endl;
  cout << priors.rows() << "x" << priors.cols() << endl;

  if (thor::os::suffix(data_f) == "mp4") {
    cv::VideoCapture cap(data_f);
    if (!cap.isOpened()) {
      std::cout << "Error: video-stream can't be opened! \n";
      return 1;
    }

    cv::Mat frame;
    float* data;
    int ori_w = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int ori_h = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float scale_x = ori_w / INPUT_W;
    float scale_y = ori_h / INPUT_H;

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
      // this pixel value is normal
      // substract mean value
      cv::subtract(resizedImage, cv::Scalar(104, 117, 123), resizedImage);
      // HWC -> CHW, how to convert these channels?
      data = HWC2CHW(resizedImage, kMeans);

      timer.lap();
      MatrixXf all_faces = doInference(*context, data, priors, 1);
      double cost = timer.lap();
      LOG(INFO) << "fps: " << 1 / cost << ", cost: " << cost;

      // resolve these results
      // loc: 28672x4*size(float)
      // visualize these faces
      for (size_t i = 0; i < all_faces.rows(); i++)
      {
        // get rectangle and confs
        Eigen::VectorXf one_face = all_faces.row(i);
        int x1 = (int) (one_face[0]*ori_w);
        int y1 = (int) (one_face[1]*ori_h);
        int x2 = (int) (one_face[2]*ori_w);
        int y2 = (int) (one_face[3]*ori_h);
        float conf = one_face[4];
        if (conf > 0.6) {
          char conf_str[128];
          sprintf(conf_str, "%.3f", conf);
          cv::putText(frame, conf_str, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 0.6,
                      cv::Scalar(255, 0, 225));
          cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255));
        }
      }

      // also we need log FPS here
      char fps_str[128];
      sprintf(fps_str, "%.3f", 1/cost);
      cv::putText(frame, fps_str, cv::Point(40, 40), cv::FONT_HERSHEY_COMPLEX, 0.6,
                  cv::Scalar(255, 0, 225));
      std::free(data);
      cv::imshow("ori", frame);
      cv::waitKey(1);
    }

  } else {
    // on image
    float* data;
    cv::Mat frame = cv::imread(data_f);
    cv::Mat resizedImage = cv::Mat::zeros(INPUT_H, INPUT_W, CV_32FC3);
    cv::resize(frame, resizedImage, cv::Size(INPUT_W, INPUT_H));
    // this pixel value is normal
    // substract mean value
    cv::subtract(resizedImage, cv::Scalar(104, 117, 123), resizedImage);
    // HWC -> CHW, how to convert these channels?
    data = HWC2CHW(resizedImage, kMeans);

    MatrixXf all_faces = doInference(*context, data, priors, 1);

    // resolve these results
    // loc: 28672x4*size(float)
    // visualize these faces
    for (size_t i = 0; i < all_faces.rows(); i++)
    {
      // get rectangle and confs
      Eigen::VectorXf one_face = all_faces.row(i);
      int x1 = (int) (one_face[0]*INPUT_W);
      int y1 = (int) (one_face[1]*INPUT_H);
      int x2 = (int) (one_face[2]*INPUT_W);
      int y2 = (int) (one_face[3]*INPUT_H);
      float conf = one_face[4];
      if (conf > 0.6) {
//        LOG(INFO) << x1 << " " << y1 << " " << x2 << " " << y2 << " " << conf;
        char conf_str[128];
        sprintf(conf_str, "%.3f", conf);
        cv::putText(frame, conf_str, cv::Point(x1, y1), cv::FONT_HERSHEY_COMPLEX, 0.6,
                    cv::Scalar(255, 0, 225));
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 255));
      }
    }
    cv::imshow("ori", frame);
    cv::waitKey(0);
  }

  return 0;
}

int main(int argc, char** argv) {
  // trt_file data_file
  run(argc, argv);
  return 0;
}

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
#include "thor/timer.h"
#include "thor/logging.h"
#include "thor/os.h"

#include "../include/batch_stream.h"
#include "../include/tensorrt_utils.h"
#include "../include/tensorrt_common.h"

using namespace thor::log;
using namespace nvinfer1;
static tensorrt::Logger gLogger;

std::stringstream gieModelStream;
static const int INPUT_H = 512;
static const int INPUT_W = 512;
static const int INPUT_C = 3;
static const int OUTPUT_SIZE = 114688;
static const int INPUT_SIZE = 786432;

static int gUseDLACore{-1};
const char* INPUT_BLOB_NAME = "0";
// const char* OUTPUT_BLOB_NAME = "prob";

class Int8EntropyCalibrator : public IInt8EntropyCalibrator {
 public:
  //（1，3，512，512）--》（500，3，512，512）
  Int8EntropyCalibrator(BatchStream& stream, bool readCache = true)
      : mStream(stream), mReadCache(readCache) {
    // DimsNCHW dims = mStream.getDims();
    mInputCount = 1 * 3 * 512 * 512;
    CHECK(cudaMalloc(&mDeviceInput, mInputCount * sizeof(float)));
    // mStream.reset(firstBatch);
  }

  virtual ~Int8EntropyCalibrator() { CHECK(cudaFree(mDeviceInput)); }

  int getBatchSize() const override { return mStream.getBatchSize(); }

  bool getBatch(void* bindings[], const char* names[],
                int nbBindings) override {
    if (!mStream.next()) return false;

    CHECK(cudaMemcpy(mDeviceInput, mStream.get_image(),
                     mInputCount * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], INPUT_BLOB_NAME));
    bindings[0] = mDeviceInput;
    return true;
  }

  const void* readCalibrationCache(size_t& length) override  //读缓存
  {
    mCalibrationCache.clear();
    std::ifstream input(calibrationTableName(), std::ios::binary);
    input >> std::noskipws;
    if (mReadCache && input.good())
      std::copy(std::istream_iterator<char>(input),
                std::istream_iterator<char>(),
                std::back_inserter(mCalibrationCache));

    length = mCalibrationCache.size();
    return length ? &mCalibrationCache[0] : nullptr;
  }

  void writeCalibrationCache(const void* cache, size_t length) override {
    std::ofstream output(calibrationTableName(), std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
  }

 private:
  static std::string calibrationTableName() {
    return std::string("CalibrationTable");
  }
  BatchStream mStream;
  bool mReadCache{true};

  size_t mInputCount;
  void* mDeviceInput{nullptr};
  std::vector<char> mCalibrationCache;
};

void onnxToTRTModel(
    const std::string& modelFile,  // name of the onnx model
    unsigned int maxBatchSize,  // batch size - NB must be at least as large as
    IHostMemory*& trtModelStream, DataType dataType,
    IInt8Calibrator* calibrator,
    std::string save_name)  // output buffer for the TensorRT model
{
  int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
  // create the builder
  IBuilder* builder = createInferBuilder(gLogger);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();

  auto parser = nvonnxparser::createParser(*network, gLogger);

  if (!parser->parseFromFile(modelFile.c_str(), verbosity)) {
    string msg("failed to parse onnx file");
    LOG(INFO) << msg.c_str();
    exit(EXIT_FAILURE);
  }
  if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8()))
    exit(EXIT_FAILURE);

  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(4_GB);  //不能超过你的实际能用的显存的大小，例如我的1060的可用为4.98GB，超过4.98GB会报错
  builder->setInt8Mode(dataType == DataType::kINT8);  //
  builder->setInt8Calibrator(calibrator);             //
  samplesCommon::enableDLA(builder, gUseDLACore);
  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);

  // we can destroy the parser
  parser->destroy();

  // serialize the engine, then close everything down  序列化
  trtModelStream = engine->serialize();

  gieModelStream.write((const char*)trtModelStream->data(),
                       trtModelStream->size());
  std::ofstream SaveFile(save_name, std::ios::out | std::ios::binary);
  SaveFile.seekp(0, std::ios::beg);
  SaveFile << gieModelStream.rdbuf();
  gieModelStream.seekg(0, gieModelStream.beg);

  engine->destroy();
  network->destroy();
  builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output,
                 int batchSize) {
  const ICudaEngine& engine = context.getEngine();
  // input and output buffer pointers that we pass to the engine - the engine
  // requires exactly IEngine::getNbBindings(), of these, but in this case we
  // know that there is exactly one input and one output.
  assert(engine.getNbBindings() == 2);
  void* buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  int inputIndex, outputIndex;
  for (int b = 0; b < engine.getNbBindings(); ++b) {
    if (engine.bindingIsInput(b))
      inputIndex = b;
    else
      outputIndex = b;
  }
  // create GPU buffers and a stream   创建GPU缓冲区和流
  CHECK(cudaMalloc(&buffers[inputIndex],
                   batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex],
                   batchSize * OUTPUT_SIZE * sizeof(float)));

  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it
  // back:
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(
      batchSize, buffers, stream,
      nullptr);  // TensorRT的执行通常是异步的，因此将核加入队列放在CUDA流上
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        batchSize * OUTPUT_SIZE * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // release the stream and the buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

int do_serialize(int argc, char** argv) {
  IHostMemory* trtModelStream{nullptr};
  // gUseDLACore = samplesCommon::parseDLA(argc, argv);
  // create a TensorRT model from the onnx model and serialize it to a stream
  std::string file_name = argv[3];
  std::string modelFile = argv[5];
  if (0 == strcmp(argv[2], "int8")) {
    if (argc != 7) {
      std::cout << "no_have_serialize_txt" << std::endl;
      std::cout << "int8" << std::endl;
      std::cout << "cam or video file" << std::endl;
      std::cout << "save serialize name" << std::endl;
      std::cout << "onnx name" << std::endl;
      std::cout << "batch calibration name" << std::endl;
      return 1;
    }
    std::string batchfile = argv[6];
    BatchStream calibrationStream(500, batchfile);
    Int8EntropyCalibrator calibrator(calibrationStream);
    std::cout << "using int8 mode" << std::endl;
    onnxToTRTModel(modelFile, 1, trtModelStream, DataType::kINT8, &calibrator,
                   file_name);  //读onnx模型,序列化引擎
  } else {
    if (argc != 6) {
      std::cout << "no_have_serialize_txt" << std::endl;
      std::cout << "float32" << std::endl;
      std::cout << "cam or video file" << std::endl;
      std::cout << "save serialize name" << std::endl;
      std::cout << "onnx name" << std::endl;
      return 1;
    }
    std::cout << "using float32 mode" << std::endl;
    onnxToTRTModel(modelFile, 1, trtModelStream, DataType::kFLOAT, nullptr,
                   file_name);  //读onnx模型,序列化引擎
  }
  std::cout << "rialize model ready" << std::endl;
  assert(trtModelStream != nullptr);

  // deserialize the engine    DLA加速
  //反序列化引擎
  IRuntime* runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  if (gUseDLACore >= 0) {
    runtime->setDLACore(gUseDLACore);
  }
  //反序列化
  ICudaEngine* engine = runtime->deserializeCudaEngine(
      trtModelStream->data(), trtModelStream->size(), nullptr);

  assert(engine != nullptr);
  trtModelStream->destroy();
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);

  int cam_index = 0;
  cv::VideoCapture cap;
  if (0 == strcmp(argv[4], "cam")) {
    cap.open(cam_index);
  } else {
    cap.open(argv[4]);
  }

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
  while (1) {
    struct timeval tval_before, tval_after, tval_result;
    gettimeofday(&tval_before, NULL);
    cap >> frame;
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
    cv::Mat dst =
        cv::Mat::zeros(512, 512, CV_32FC3);  //新建一张512x512尺寸的图片Mat
    cv::resize(img, dst, dst.size());
    data = normal(dst);
    doInference(*context, data, prob, 1);  // chw
    out = read2mat(prob, out);
    // hwc
    cv::resize(out, real_out, real_out.size());
    real_out_ = map2threeunchar(real_out, real_out_);
    cv::imshow("res map out?", real_out_);
    cv::imshow("ori", frame);
    // show_image(real_out, "Segmentation");   //显示图片
    std::free(data);
    // free_image(real_out);
    if (cv::waitKey(10) == 27) break;
    gettimeofday(&tval_after, NULL);
    timersub(&tval_after, &tval_before, &tval_result);
    float curr = 1000000.f / ((long int)tval_result.tv_usec);
    // std::cout << (float)tval_result.tv_usec << std::endl;
    // printf("\nFPS:%.0f\n", fps);
    cout << "fps: " << fps << endl;
    fps = .9 * fps + .1 * curr;
    // fps = curr;
  }
  cv::destroyAllWindows();
  cap.release();

  // destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();
  std::cout << "shut down" << std::endl;
  // nvcaffeparser1::shutdownProtobufLibrary();

  return 0;  //无法退出？解决
}

int run(int argc, char** argv) {
  gieModelStream.seekg(0, gieModelStream.beg);
  std::ifstream serialize_iutput_stream(argv[3],
                                        std::ios::in | std::ios::binary);
  if (!serialize_iutput_stream) {
    std::cout << "cannot find serialize file" << std::endl;
    return 1;
  }
  serialize_iutput_stream.seekg(0);

  gieModelStream << serialize_iutput_stream.rdbuf();
  gieModelStream.seekg(0, std::ios::end);
  const int modelSize = gieModelStream.tellg();
  gieModelStream.seekg(0, std::ios::beg);
  void* modelMem = malloc(modelSize);
  gieModelStream.read((char*)modelMem, modelSize);

  IHostMemory* trtModelStream{nullptr};
  IBuilder* builder = createInferBuilder(gLogger);
  if (0 == strcmp(argv[2], "int8")) {
    if (argc != 6) {
      std::cout << "have_serialize_txt" << std::endl;
      std::cout << "int8" << std::endl;
      std::cout << "cam or video file" << std::endl;
      std::cout << "saved serialize name" << std::endl;
      std::cout << "batch calibration name" << std::endl;
      return 1;
    }
    std::string batchfile = argv[5];
    BatchStream calibrationStream(500, batchfile);
    Int8EntropyCalibrator calibrator(calibrationStream);
    std::cout << "using int8 mode" << std::endl;
    builder->platformHasFastInt8();
    builder->setInt8Mode(true);               //
    builder->setInt8Calibrator(&calibrator);  //
  } else {
    if (argc != 5) {
      std::cout << "have_serialize_txt" << std::endl;
      std::cout << "float" << std::endl;
      std::cout << "cam or video file" << std::endl;
      std::cout << "saved serialize name" << std::endl;
      return 1;
    }
    std::cout << "using float32 mode" << std::endl;
  }
  builder->destroy();
  IRuntime* runtime = createInferRuntime(gLogger);

  ICudaEngine* engine =
      runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  std::free(modelMem);
  assert(engine != nullptr);
  IExecutionContext* context = engine->createExecutionContext();
  assert(context != nullptr);
  int cam_index = 0;
  char* filename = (argc > 3) ? argv[3] : 0;
  cout << "inference directly from engine.\n";

  cv::VideoCapture cap;
  if (0 == strcmp(argv[4], "cam")) {
    cap.open(cam_index);
  } else {
    cap.open(argv[4]);
  }
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

  while (1) {
    cap >> frame;
    cv::Mat img;
    cv::cvtColor(frame, img, cv::COLOR_BGR2RGB);
    cv::Mat dst =
        cv::Mat::zeros(512, 512, CV_32FC3);  //新建一张512x512尺寸的图片Mat
    cv::resize(img, dst, dst.size());
    data = normal(dst);
    doInference(*context, data, prob, 1);  // chw
    out = read2mat(prob, out);
    cv::resize(out, real_out, real_out.size());
    real_out_ = map2threeunchar(real_out, real_out_);
    cv::Mat mask_resized;
    cv::resize(real_out_, mask_resized, cv::Size(w, h));

    double cost = timer.lap();
    cout << "fps: " << 1 / cost << endl;

    // add weighted them and show
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
  runtime->destroy();
  std::cout << "shut down" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
  if (0 == strcmp(argv[1], "have_serialize_txt")) {
    run(argc, argv);
  } else {
    do_serialize(argc, argv);
  }
  return 0;
}

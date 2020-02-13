#ifndef __TRT_NET_H_
#define __TRT_NET_H_

#include <algorithm>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "NvOnnxParser.h"

#include "./all_plugin_factory.h"
#include "./tensorrt_utils.h"

using namespace std;

namespace tensorrt {

namespace RUN_MODE {
enum RUN_MODE { FLOAT32 = 0, FLOAT16 = 1, INT8 = 2 };
}

class TensorRTEngine {
 public:

  TensorRTEngine(const std::string &modelFile, const std::string &saveEngineFile,
      const std::vector<std::vector<float>> &calibratorData,
                                int mode /*= RUN_MODE::FLOAT32*/,
                                int maxBatchSize /*= 1*/);
  explicit TensorRTEngine(const std::string &engineFile);

  ~TensorRTEngine() {
    // Release the stream and the buffers
    cudaStreamSynchronize(mTrtCudaStream);
    cudaStreamDestroy(mTrtCudaStream);
    for (auto &item : mTrtCudaBuffer) cudaFree(item);
    mTrtPluginFactory.destroyPlugin();
    if (!mTrtRunTime) mTrtRunTime->destroy();
    if (!mTrtContext) mTrtContext->destroy();
    if (!mTrtEngine) mTrtEngine->destroy();
  };

  void saveEngine(std::string fileName) {
    if (mTrtEngine) {
      nvinfer1::IHostMemory *data = mTrtEngine->serialize();
      std::ofstream file;
      file.open(fileName, std::ios::binary | std::ios::out);
      if (!file.is_open()) {
        std::cout << "read create engine file" << fileName << " failed"
                  << std::endl;
        return;
      }
      file.write((const char *)data->data(), data->size());
      file.close();
    }
  };

  void doInference(const void *inputData, void *outputData, int batchSize = 1);
  inline size_t getInputSize() {
    return std::accumulate(mTrtBindBufferSize.begin(),
                           mTrtBindBufferSize.begin() + mTrtInputCount, 0);
  };
  inline size_t getOutputSize() {
    return std::accumulate(mTrtBindBufferSize.begin() + mTrtInputCount,
                           mTrtBindBufferSize.end(), 0);
  };
  void printTime() { mTrtProfiler.printLayerTimes(mTrtIterationTime); }
  inline int getBatchSize() { return mTrtBatchSize; };
  nvinfer1::ICudaEngine* getEngine() {
    return mTrtEngine;
  }

 public:
  // some net info config
  int input_width;
  int input_height;
  int input_channel;

 private:
//  nvinfer1::ICudaEngine *loadModelAndCreateEngine(
//      const std::string &modelFile,  // name of the onnx model
//      IHostMemory *&trtModelStream);

  void InitEngine();

  nvinfer1::IExecutionContext *mTrtContext;
  nvinfer1::ICudaEngine *mTrtEngine;
  nvinfer1::IRuntime *mTrtRunTime;
  PluginFactory mTrtPluginFactory;
  cudaStream_t mTrtCudaStream;
  Profiler mTrtProfiler;
  int mTrtRunMode;

  std::vector<void *> mTrtCudaBuffer;
  std::vector<int64_t> mTrtBindBufferSize;
  int mTrtInputCount;
  int mTrtIterationTime;
  int mTrtBatchSize;

  std::stringstream _gieModelStream;

};
}  // namespace tensorrt

#endif  //__TRT_NET_H_

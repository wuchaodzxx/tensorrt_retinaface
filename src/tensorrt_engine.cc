#include <cassert>
#include <chrono>
#include <cublas_v2.h>
#include <cudnn.h>
#include <iostream>
#include <sstream>
#include <string.h>
#include <time.h>
#include <unordered_map>
#include <memory>

#include "tensorrt_engine.h"
//#include "entroy_calibrator.h"

using namespace nvinfer1;
using namespace nvonnxparser;
using namespace plugin;

static tensorrt::Logger gLogger;

#define RETURN_AND_LOG(ret, severity, message)                                 \
    do                                                                         \
    {                                                                          \
        std::string error_message = "ssd_error_log: " + std::string(message); \
        gLogger.log(ILogger::Severity::k##severity, error_message.c_str());    \
        return (ret);                                                          \
    } while (0)

inline void *safeCudaMalloc(size_t memSize) {
  void *deviceMem;
  CUDA_CHECK(cudaMalloc(&deviceMem, memSize));
  if (deviceMem == nullptr) {
    std::cerr << "Out of memory" << std::endl;
    exit(1);
  }
  return deviceMem;
}

inline int64_t volume(const nvinfer1::Dims &d) {
  return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}

namespace tensorrt {
TensorRTEngine::TensorRTEngine(const std::string &modelFile,
                               const std::string &saveEngineFile,
                               const std::vector<std::vector<float>> &calibratorData,
                               int mode,
                               int maxBatchSize) {
  IBuilder* builder = createInferBuilder(gLogger);
  assert(builder != nullptr);
  nvinfer1::INetworkDefinition* network = builder->createNetwork();
  auto parser = nvonnxparser::createParser(*network, gLogger);

  if ( !parser->parseFromFile(modelFile.c_str(), static_cast<int>(gLogger.reportableSeverity) ) )
  {
    cerr << "Failure while parsing ONNX file" << std::endl;
  }


  IHostMemory *trtModelStream{nullptr};
  // Build the engine
  builder->setMaxBatchSize(maxBatchSize);
  builder->setMaxWorkspaceSize(1 << 30);

  if (mTrtRunMode == RUN_MODE::INT8) {
    std::cout << "setInt8Mode" << std::endl;
    if (!builder->platformHasFastInt8())
      std::cout << "Notice: the platform do not has fast for int8" << std::endl;
//    builder->setInt8Mode(true);
//    builder->setInt8Calibrator(calibrator);
    cerr << "int8 mode not supported for now.\n";
  } else if (mTrtRunMode == RUN_MODE::FLOAT16) {
    std::cout << "setFp16Mode" << std::endl;
    if (!builder->platformHasFastFp16())
      std::cout << "Notice: the platform do not has fast for fp16" << std::endl;
    builder->setFp16Mode(true);
  }

  ICudaEngine* engine = builder->buildCudaEngine(*network);
  assert(engine);
  // we can destroy the parser
  parser->destroy();
  // serialize the engine, then close everything down
  trtModelStream = engine->serialize();
  trtModelStream->destroy();
  InitEngine();
}

TensorRTEngine::TensorRTEngine(const std::string &engineFile)
    : mTrtContext(nullptr),
      mTrtEngine(nullptr),
      mTrtRunTime(nullptr),
      mTrtRunMode(RUN_MODE::FLOAT32),
      mTrtInputCount(0),
      mTrtIterationTime(0) {
  _gieModelStream.seekg(0, _gieModelStream.beg);
  std::ifstream serialize_iutput_stream(engineFile, std::ios::in | std::ios::binary);
  if (!serialize_iutput_stream) {
    std::cerr << "cannot find serialize file" << std::endl;
  }
  serialize_iutput_stream.seekg(0);

  _gieModelStream << serialize_iutput_stream.rdbuf();
  _gieModelStream.seekg(0, std::ios::end);
  const int modelSize = _gieModelStream.tellg();
  _gieModelStream.seekg(0, std::ios::beg);
  void* modelMem = malloc(modelSize);
  _gieModelStream.read((char*)modelMem, modelSize);

  IBuilder* builder = createInferBuilder(gLogger);
  builder->destroy();
  // todo: release runtime avoid memory leak
  IRuntime* runtime = createInferRuntime(gLogger);
  mTrtEngine = runtime->deserializeCudaEngine(modelMem, modelSize, NULL);
  std::free(modelMem);
  assert(mTrtEngine != nullptr);
}

void TensorRTEngine::InitEngine() {
  mTrtBatchSize = mTrtEngine->getMaxBatchSize();
  mTrtContext = mTrtEngine->createExecutionContext();
  assert(mTrtContext != nullptr);
  mTrtContext->setProfiler(&mTrtProfiler);

  // Input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings()
  int nbBindings = mTrtEngine->getNbBindings();
  cout << "we get the engine bindings: " << nbBindings << endl;

  mTrtCudaBuffer.resize(nbBindings);
  mTrtBindBufferSize.resize(nbBindings);
  for (int i = 0; i < nbBindings; ++i) {
    Dims dims = mTrtEngine->getBindingDimensions(i);
    DataType dtype = mTrtEngine->getBindingDataType(i);
    int64_t totalSize = volume(dims) * mTrtBatchSize * getElementSize(dtype);
    mTrtBindBufferSize[i] = totalSize;
    mTrtCudaBuffer[i] = safeCudaMalloc(totalSize);
    if (mTrtEngine->bindingIsInput(i))
      mTrtInputCount++;
  }

  CUDA_CHECK(cudaStreamCreate(&mTrtCudaStream));
}


void TensorRTEngine::doInference(const void *inputData, void *outputData, int batchSize /*= 1*/) {
  //static const int batchSize = 1;
  assert(mTrtInputCount == 1);
  assert(batchSize <= mTrtBatchSize);

  // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
  int inputIndex = 0;
  cout << "copy data to cuda\n";
  CUDA_CHECK(cudaMemcpyAsync(mTrtCudaBuffer[inputIndex],
                             inputData,
                             mTrtBindBufferSize[inputIndex],
                             cudaMemcpyHostToDevice,
                             mTrtCudaStream));
  cout << "start to execute.\n";
  mTrtContext->execute(batchSize, &mTrtCudaBuffer[inputIndex]);

  for (size_t bindingIdx = mTrtInputCount; bindingIdx < mTrtBindBufferSize.size(); ++bindingIdx) {
    auto size = mTrtBindBufferSize[bindingIdx];
    CUDA_CHECK(cudaMemcpyAsync(outputData, mTrtCudaBuffer[bindingIdx], size, cudaMemcpyDeviceToHost, mTrtCudaStream));
    outputData = (char *) outputData + size;
  }
}

}
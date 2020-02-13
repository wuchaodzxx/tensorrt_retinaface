//
// Created by fagangjin on 20/9/2019.
//

#ifndef ONNX_TENSORRT_INCLUDE_ENTROPY_CALIBRATOR_H_
#define ONNX_TENSORRT_INCLUDE_ENTROPY_CALIBRATOR_H_

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

#include "batch_stream.h"
#include "tensorrt_common.h"


/**
 * \ this is for onnx tensorrt int calibration
 * \ codes not verified yet, just borrowed here
 */


namespace tensorrt {

class Int8EntropyCalibrator : public IInt8EntropyCalibrator{

  const char* INPUT_BLOB_NAME = "0";


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


}
#endif //ONNX_TENSORRT_INCLUDE_ENTROPY_CALIBRATOR_H_

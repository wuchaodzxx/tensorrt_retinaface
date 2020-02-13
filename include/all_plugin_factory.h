#ifndef __PLUGIN_FACTORY_H_
#define __PLUGIN_FACTORY_H_

#include <vector>
#include <memory>
#include <regex>
#include "NvInferPlugin.h"
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvOnnxParserRuntime.h"
#include <unordered_map>
#include <list>

/**
 * 
 * We provide all plugin factory here
 * 
 * supports layers
 * 
 * UpsampleLayerPlugin
 * YoloLayerPlugin
 * ProposalPlugin
 * 
 * 
 * */

namespace tensorrt {

// codes borrowed from onnx-tensorrt
template<typename T>
using string_map = std::unordered_map<std::string, T>;
class Plugin;
typedef Plugin* (*plugin_deserializer)(const void* serialData,
                                       size_t serialLength);
struct IOwnable {
  virtual void destroy() = 0;
 protected:
  virtual ~IOwnable() {}
};
struct OwnableDeleter {
  void operator()(IOwnable* obj) const {
    obj->destroy();
  }
};
using UniqueOwnable = std::unique_ptr<IOwnable, OwnableDeleter>;


static constexpr float NEG_SLOPE = 0.1;
static constexpr float UPSAMPLE_SCALE = 2.0;
static constexpr int CUDA_THREAD_NUM = 512;

// Integration for serialization.

using nvinfer1::plugin::createPReLUPlugin;
using nvinfer1::plugin::INvPlugin;

class PluginFactory : public nvonnxparser::IPluginFactory {
 private:
  nvinfer1::ILogger* _logger;
  string_map<plugin_deserializer> _plugin_registry;
  std::list<UniqueOwnable> _owned_plugin_instances;

 public:

  PluginFactory() {};
  ~PluginFactory() {};

  // The application has to destroy the plugin when it knows it's safe to do so.
  void destroyPlugin() {
  }

  void registerPlugin(const char *plugin_type, plugin_deserializer func) {
    // Note: This allows existing importers to be replaced
    _plugin_registry[plugin_type] = func;
  }
  // This is used by TRT during engine deserialization
  nvinfer1::IPlugin *createPlugin(const char *layerName,
                                          const void *serialData,
                                          size_t serialLength) {

  };
  void destroy() override {

  }

};

} // namespace tensorrt

#endif
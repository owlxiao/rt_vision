#ifndef RT_VISION_TENSORRTDETECTOR_H
#define RT_VISION_TENSORRTDETECTOR_H

#include "vision/Logging.h"
#include "vision/VisionDetector.h"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <string>

namespace rt_vision {

#define CHECK(status)                                                          \
  do {                                                                         \
    auto ret = (status);                                                       \
    if (ret != 0) {                                                            \
      std::cerr << "Cuda failure: " << ret << std::endl;                       \
      abort();                                                                 \
    }                                                                          \
  } while (0)

class TensorRTDetector : public VisionDetector {
public:
  TensorRTDetector(std::string &engineFilePath, int device, float confNmsThresh,
                   float confBboxThresh, std::size_t confNumClasses);

public:
  std::vector<Object> inference(const cv::Mat &frame);
  void doInference(float *input, float *output);

private:
  Logger _gLogger;
  std::size_t _outputSize;
  std::unique_ptr<nvinfer1::IRuntime> _runtime{nullptr};
  std::unique_ptr<nvinfer1::ICudaEngine> _engine{nullptr};
  std::unique_ptr<nvinfer1::IExecutionContext> _context{nullptr};
};

} // namespace rt_vision

#endif
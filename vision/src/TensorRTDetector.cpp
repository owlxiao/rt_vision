#include "vision/TensorRTDetector.h"
#include "vision/VisionDetector.h"

#include <NvInferImpl.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>

namespace rt_vision {

/// FIXME
static const char *INPUT_BLOB_NAME = "input_0";
static const char *OUTPUT_BLOB_NAME = "output_0";

TensorRTDetector::TensorRTDetector(std::string &engineFilePath, int device,
                                   float confNmsThresh, float confBboxThresh,
                                   std::size_t confNumClasses)
    : VisionDetector(confNmsThresh, confBboxThresh, confNumClasses) {
  cudaSetDevice(device);

  /// create a model using the API directly and serialize it to a stream
  std::ifstream file(engineFilePath, std::ios::binary);
  std::unique_ptr<char[]> trtModelStream{nullptr};
  std::size_t size{0};

  if (file.good()) {
    file.seekg(0, file.end);
    size = file.tellg();

    file.seekg(0, file.beg);
    trtModelStream = std::make_unique<char[]>(size);

    assert(trtModelStream);
    file.read(trtModelStream.get(), size);

    file.close();
  } else {
    std::cerr << "Failed to open" << std::endl;
  }

  _runtime = std::unique_ptr<nvinfer1::IRuntime>{
      nvinfer1::createInferRuntime(_gLogger)};
  assert(_runtime != nullptr);

  _engine = std::unique_ptr<nvinfer1::ICudaEngine>{
      _runtime->deserializeCudaEngine(trtModelStream.get(), size)};
  assert(_engine != nullptr);

  _context = std::unique_ptr<nvinfer1::IExecutionContext>(
      this->_engine->createExecutionContext());
  assert(this->_context != nullptr);

  auto out_dims = _engine->getTensorShape(OUTPUT_BLOB_NAME);
  _outputSize = std::accumulate(out_dims.d, out_dims.d + out_dims.nbDims, 1,
                                std::multiplies<int>());
}

std::vector<Object> TensorRTDetector::inference(const cv::Mat &frame) {
  /// Perform necessart pre-processing on the fram
  auto pr_img = staticResize(frame);

  auto input_blob = std::make_unique<float[]>(pr_img.total() * 3);
  blobFromImage(pr_img, input_blob.get());

  /// Run inference
  auto output_blob = std::make_unique<float[]>(_outputSize);
  doInference(input_blob.get(), output_blob.get());

  float scale =
      std::min(_input_w / (frame.cols * 1.0), _input_h / (frame.rows * 1.0));

  std::vector<Object> objects;
  decodeOutput(output_blob.get(), objects, _bboxConfThresh, scale, frame.cols,
               frame.rows);

  return objects;
}

void TensorRTDetector::doInference(float *input, float *output) {
  const nvinfer1::ICudaEngine &engine = _context->getEngine();

  /// Pointers to input and output device buffers to pass to engine.
  /// Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbIOTensors() == 2);
  void *buffers[2];

  constexpr int inputIndex = 0;
  constexpr int outputIndex = 1;

  /// In order to bind the buffers, we need to know the names of the input and
  /// output tensors. Note that indices are guaranteed to be less than
  /// IEngine::getNbBindings()
  const auto inputType = engine.getTensorDataType(INPUT_BLOB_NAME);
  assert(inputType == nvinfer1::DataType::kFLOAT);

  const auto outputType = engine.getTensorDataType(OUTPUT_BLOB_NAME);
  assert(outputType == nvinfer1::DataType::kFLOAT);

  /// Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex],
                   3 * _input_h * _input_w * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], _outputSize * sizeof(float)));

  /// Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  /// DMA input batch data to device, infer on the batch asynchronously, and DMA
  /// output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input,
                        3 * _input_h * _input_w * sizeof(float),
                        cudaMemcpyHostToDevice, stream));

  _context->setTensorAddress(INPUT_BLOB_NAME, buffers[inputIndex]);
  _context->setTensorAddress(OUTPUT_BLOB_NAME, buffers[outputIndex]);

  _context->enqueueV3(stream);

  CHECK(cudaMemcpyAsync(output, buffers[outputIndex],
                        _outputSize * sizeof(float), cudaMemcpyDeviceToHost,
                        stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

} // namespace rt_vision
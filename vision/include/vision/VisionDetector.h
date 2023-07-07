#ifndef RT_VISION_VISIONDETECTOR_H
#define RT_VISION_VISIONDETECTOR_H

#include <opencv4/opencv2/opencv.hpp>

#include <vector>

namespace rt_vision {

struct Object {
  cv::Rect_<float> rect;
  std::size_t label;
  float prob;
};

struct GridAndStride {
  std::size_t grid0;
  std::size_t grid1;
  std::size_t stride;
};

class VisionDetector {
public:
  VisionDetector(float confNmsThresh, float confBboxThresh,
                 std::size_t confNumClasses)
      : _input_w(0), _input_h(0), _nmsThresh(confNmsThresh),
        _bboxConfThresh(confBboxThresh), _numClasses(confNumClasses){};

public:
  cv::Mat staticResize(const cv::Mat &img);
  void blobFromImage(const cv::Mat &img, float *blob_data);

  void decodeOutput(const float *prob, std::vector<Object> &objects,
                    const float bboxConfThresh, float scale, const int imgWidth,
                    const int imgHeight);

  virtual std::vector<Object> inference(const cv::Mat &frame) = 0;

private:
  void generateGridsAndStride(std::vector<std::size_t> &strides,
                              std::vector<GridAndStride> &gridStrides);

  void generateYoloxProposals(const std::vector<GridAndStride> &gridStrides,
                              const float *featBlob, const float probThreshold,
                              std::vector<Object> &objects);

  void qsortDescentInplace(std::vector<Object> &objects);
  void qsortDescentInplace(std::vector<Object> &faceobjects, int left,
                           int right);

  void nmsSortedBboxes(const std::vector<Object> &facebjects,
                       std::vector<int> &picked, const float nmsThreshold);

  float intersectionArea(const Object &a, const Object &b);

protected:
  std::size_t _input_w;
  std::size_t _input_h;
  float _nmsThresh;
  float _bboxConfThresh;
  std::size_t _numClasses;
  std::vector<std::size_t> _strides = {8, 16, 32};
};

} // namespace rt_vision

#endif
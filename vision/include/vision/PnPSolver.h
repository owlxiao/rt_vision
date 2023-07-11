#ifndef RT_VISION_PNPSOLVER_H
#define RT_VISION_PNPSOLVER_H

#include "vision/VisionDetector.h"

#include <opencv2/core.hpp>

#include <array>
#include <vector>

namespace rt_vision {

class PnPSolver {
public:
  PnPSolver(const std::array<double, 9> &cameraMatrix,
            const std::vector<double> &distortionCoefficients);

  bool solvePnP(const Object &obj, cv::Mat &rvec, cv::Mat &tvec);

private:
  cv::Mat _cameraMatrix;
  cv::Mat _distCoeffs;

  /// Unit: mm
  static constexpr float TINDER_WIDTH = 70;
  static constexpr float TINDER_HEIGHT = 125;

  std::vector<cv::Point3f> _tinderPoints;
};
} // namespace rt_vision

#endif
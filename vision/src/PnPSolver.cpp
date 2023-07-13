#include "vision/PnPSolver.h"
#include "vision/VisionDetector.h"

namespace rt_vision {

PnPSolver::PnPSolver(const std::array<double, 9> &camera_matrix,
                     const std::vector<double> &dist_coeffs)
    : _cameraMatrix(
          cv::Mat(3, 3, CV_64F, const_cast<double *>(camera_matrix.data()))
              .clone()),
      _distCoeffs(
          cv::Mat(1, 5, CV_64F, const_cast<double *>(dist_coeffs.data()))
              .clone()) {

  /// Unit: m
  constexpr double small_half_y = TINDER_WIDTH / 2.0 / 1000.0;
  constexpr double small_half_z = TINDER_HEIGHT / 2.0 / 1000.0;

  _tinderPoints.emplace_back(cv::Point3f(0, small_half_y, -small_half_z));
  _tinderPoints.emplace_back(cv::Point3f(0, small_half_y, small_half_z));
  _tinderPoints.emplace_back(cv::Point3f(0, -small_half_y, small_half_z));
  _tinderPoints.emplace_back(cv::Point3f(0, -small_half_y, -small_half_z));
}

bool PnPSolver::solvePnP(const Object &obj, cv::Mat &rvec, cv::Mat &tvec) {
  std::vector<cv::Point2f> image_points;

  // Fill in image points
  image_points.emplace_back(cv::Point2f(
      obj.rect.x, obj.rect.y + obj.rect.height)); /// Bottom-left corner
  image_points.emplace_back(
      cv::Point2f(obj.rect.x, obj.rect.y)); /// Top-left corner

  image_points.emplace_back(cv::Point2f(obj.rect.x + obj.rect.width,
                                        obj.rect.y)); /// Top-right corner
  image_points.emplace_back(
      cv::Point2f(obj.rect.x + obj.rect.width,
                  obj.rect.y + obj.rect.height)); /// Bottom-right corner

  return cv::solvePnP(_tinderPoints, image_points, _cameraMatrix, _distCoeffs,
                      rvec, tvec, false, cv::SOLVEPNP_IPPE);
}

} // namespace rt_vision
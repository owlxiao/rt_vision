#ifndef DETECTOR_DETECTORNODE_H
#define DETECTOR_DETECTORNODE_H

#include "vision/VisionDetector.h"

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

namespace rt_vision {

class DetectorNode : public rclcpp::Node {
public:
  DetectorNode(const rclcpp::NodeOptions &options);

private:
  void initializeParameters(void);
  void createPreviewWindow(void);
  void loadClassLabelsFile(void);

  void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &ptr);

private:
  bool _isPreview{};
  std::size_t _numClasses{};
  std::string _engineFilePath{};
  std::string _subImageTopicName{};
  std::string _pubObjectsTopicName{};
  std::string _classLabelsPath{};

  std::unique_ptr<VisionDetector> _inferEngine{nullptr};
  std::vector<std::string> classNames{};

  image_transport::Subscriber _subImage;
};

} // namespace rt_vision

#endif
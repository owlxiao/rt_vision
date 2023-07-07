#ifndef DETECTOR_DETECTORNODE_H
#define DETECTOR_DETECTORNODE_H

#include <rclcpp/rclcpp.hpp>

namespace rt_vision {

class DetectorNode : public rclcpp::Node {
public:
  DetectorNode(const rclcpp::NodeOptions &options);

private:
  void initializeParameters(void);
  void createPreviewWindow(void);

private:
  bool _isPreview{};
  std::size_t _numClasses{};
  std::string _engineFilePath{};
  std::string _subImageTopicName{};
  std::string _pubObjectsTopicName{};
};

} // namespace rt_vision

#endif
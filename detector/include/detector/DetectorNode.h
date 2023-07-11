#ifndef DETECTOR_DETECTORNODE_H
#define DETECTOR_DETECTORNODE_H

#include "rt_interfaces/msg/objects.hpp"
#include "vision/VisionDetector.h"

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/detail/header__struct.hpp>

namespace rt_vision {

class DetectorNode : public rclcpp::Node {
public:
  DetectorNode(const rclcpp::NodeOptions &options);

private:
  void initializeParameters(void);
  void createPreviewWindow(void);
  void loadClassLabelsFile(void);

  void colorImageCallback(const sensor_msgs::msg::Image::ConstSharedPtr &ptr);

  void bboxToObjectsMsg(rt_interfaces::msg::Objects &msg,
                        std::vector<rt_vision::Object> &objects,
                        std_msgs::msg::Header &header);

private:
  bool _isPreview{};
  std::size_t _numClasses{};
  std::string _engineFilePath{};
  std::string _subImageTopicName{};
  std::string _pubObjectsTopicName{};
  std::string _classLabelsPath{};
  float _confNmsThresh{0.0f};
  float _confBboxThresh{0.0f};

  std::unique_ptr<VisionDetector> _inferEngine{nullptr};
  std::vector<std::string> classNames{};

  image_transport::Subscriber _subImage;

  rclcpp::Publisher<rt_interfaces::msg::Objects>::SharedPtr _pubObjects;
};

} // namespace rt_vision

#endif
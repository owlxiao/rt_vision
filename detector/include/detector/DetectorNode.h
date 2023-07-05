#ifndef DETECTOR_DETECTORNODE_H
#define DETECTOR_DETECTORNODE_H

#include <rclcpp/rclcpp.hpp>

namespace rt_vision {

class DetectorNode : public rclcpp::Node {
public:
  DetectorNode(const rclcpp::NodeOptions &options);
};

} // namespace rt_vision

#endif
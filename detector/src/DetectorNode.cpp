#include "detector/DetectorNode.h"

namespace rt_vision {

DetectorNode::DetectorNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("detector", options) {
  RCLCPP_INFO(get_logger(), "Starting DetectorNode!");
}

} // namespace rt_vision

#include "rclcpp_components/register_node_macro.hpp"

/// Register the component with class_loader.
/// This acts as a sort of entry point, allowing the component to be
/// discoverable when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rt_vision::DetectorNode)

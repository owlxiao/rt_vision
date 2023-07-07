#include "detector/DetectorNode.h"

#include <opencv2/highgui.hpp>
#include <rcpputils/asserts.hpp>

namespace rt_vision {

DetectorNode::DetectorNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("detector", options) {
  RCLCPP_INFO(get_logger(), "Starting DetectorNode!");

  initializeParameters();

  if (this->_isPreview) {
    createPreviewWindow();
  }
}

void DetectorNode::initializeParameters() {
  _isPreview = this->declare_parameter<bool>("is_preview", true);
  RCLCPP_INFO(get_logger(), "Set parameter is_preview: %i", _isPreview);

  _engineFilePath =
      this->declare_parameter<std::string>("engine_file_path", "");
  RCLCPP_INFO(get_logger(), "Set parameter engine_file_path: `%s`",
              _engineFilePath.c_str());
  rcpputils::assert_true(!_engineFilePath.empty());

  int numClasses = this->declare_parameter<int>("num_classes", 1);
  rcpputils::assert_true(numClasses >= 0);
  _numClasses = numClasses;
  RCLCPP_INFO(get_logger(), "Set parameter num_classes: %zu", _numClasses);

  _subImageTopicName = this->declare_parameter<std::string>(
      "subscribe_image_topic_name", "camera/color/image_raw");
  RCLCPP_INFO(get_logger(), "Set parameter subscribe_image_topic_name: `%s`",
              _subImageTopicName.c_str());
  rcpputils::assert_true(!_subImageTopicName.empty());

  _pubObjectsTopicName = this->declare_parameter<std::string>(
      "publish_objects_topic_name", "detector/objects");
  RCLCPP_INFO(get_logger(), "Set parameter publish_objects_topic_name: `%s`",
              _pubObjectsTopicName.c_str());
  rcpputils::assert_true(!_pubObjectsTopicName.empty());
}

void DetectorNode::createPreviewWindow() {
  std::string windowName{"DetectorNode: Preview"};
  cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
}

} // namespace rt_vision

#include "rclcpp_components/register_node_macro.hpp"

/// Register the component with class_loader.
/// This acts as a sort of entry point, allowing the component to be
/// discoverable when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rt_vision::DetectorNode)

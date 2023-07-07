#include "detector/DetectorNode.h"
#include "vision/TensorRTDetector.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <rcpputils/asserts.hpp>

#include <fstream>

namespace rt_vision {

DetectorNode::DetectorNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("detector", options) {
  RCLCPP_INFO(get_logger(), "Starting DetectorNode!");

  initializeParameters();

  if (this->_isPreview) {
    createPreviewWindow();
  }

  loadClassLabelsFile();

  /// FIXME: Use parameter in bringup launch script
  int device = 0;
  float confNmsThresh{0.0f};
  float confBboxThresh{0.0f};

  _inferEngine = std::make_unique<TensorRTDetector>(
      _engineFilePath, device, confNmsThresh, confBboxThresh, _numClasses);

  this->_subImage = image_transport::create_subscription(
      this, this->_subImageTopicName,
      std::bind(&DetectorNode::colorImageCallback, this, std::placeholders::_1),
      "raw");
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

  _classLabelsPath = this->declare_parameter<std::string>("class_labels_path");
  rcpputils::assert_true(!_classLabelsPath.empty());
  RCLCPP_INFO(get_logger(), "Set parameter class_labels_path: `%s`",
              _classLabelsPath.c_str());
}

void DetectorNode::createPreviewWindow() {
  std::string windowName{"DetectorNode: Preview"};
  cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
}

void DetectorNode::loadClassLabelsFile() {
  std::ifstream file(_classLabelsPath);
  std::string buffer{};

  if (file.fail()) {
    RCLCPP_ERROR(get_logger(), "Failed to open %s", _classLabelsPath.c_str());
    rclcpp::shutdown();
  }

  while (std::getline(file, buffer)) {
    if (buffer == "")
      continue;
    classNames.push_back(buffer);
  }

  RCLCPP_INFO(get_logger(), "Loaded class labels path: %s",
              _classLabelsPath.c_str());
}

void DetectorNode::colorImageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &ptr) {
  auto img = cv_bridge::toCvCopy(ptr, "bgr8");
  cv::Mat frame = img->image;

  auto now = std::chrono::system_clock::now();

  auto objects = this->_inferEngine->inference(frame);

  auto end = std::chrono::system_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - now);
  RCLCPP_INFO(this->get_logger(), "Inference: %f FPS",
              1000.0f / elapsed.count());

  if (this->_isPreview) {
    cv::imshow("DetectorNode: Preview", frame);
    auto key = cv::waitKey(1);
    if (key == 'q') {
      rclcpp::shutdown();
    }
  }
}

} // namespace rt_vision

#include "rclcpp_components/register_node_macro.hpp"

/// Register the component with class_loader.
/// This acts as a sort of entry point, allowing the component to be
/// discoverable when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rt_vision::DetectorNode)

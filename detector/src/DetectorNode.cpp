#include "detector/DetectorNode.h"
#include "vision/TensorRTDetector.h"

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <rclcpp/qos.hpp>
#include <rcpputils/asserts.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <fstream>
#include <vision/PnPSolver.h>
#include <visualization_msgs/msg/detail/marker__struct.hpp>

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

  _inferEngine = std::make_unique<TensorRTDetector>(
      _engineFilePath, device, _confNmsThresh, _confBboxThresh, _numClasses);

  this->_subImage = image_transport::create_subscription(
      this, this->_subImageTopicName,
      std::bind(&DetectorNode::colorImageCallback, this, std::placeholders::_1),
      "raw");

  this->_pubObjects = this->create_publisher<rt_interfaces::msg::Objects>(
      "/detector/Objects", rclcpp::SensorDataQoS{});

  this->_camInfoSub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      this->_subCameraInfoTopicName, rclcpp::SensorDataQoS{},
      [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr cameraInfo) {
        this->_camInfo =
            std::make_shared<sensor_msgs::msg::CameraInfo>(*cameraInfo);
        this->_pnpSolver =
            std::make_unique<PnPSolver>(_camInfo->k, _camInfo->d);
        this->_camInfoSub.reset();
      });

  /// Visualization Marker Publisher
  /// See http://wiki.ros.org/rviz/DisplayTypes/Marker
  _objectMarker.ns = "objects";
  _objectMarker.action = visualization_msgs::msg::Marker::ADD;
  _objectMarker.scale.x = 0.05;
  _objectMarker.scale.z = 0.125;
  _objectMarker.color.a = 1.0;
  _objectMarker.color.g = 0.5;
  _objectMarker.color.b = 1.0;
  _objectMarker.lifetime = rclcpp::Duration::from_seconds(0.1);

  _textMarker.ns = "classification";
  _textMarker.action = visualization_msgs::msg::Marker::ADD;
  _textMarker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  _textMarker.scale.z = 0.1;
  _textMarker.color.a = 1.0;
  _textMarker.color.r = 1.0;
  _textMarker.color.g = 1.0;
  _textMarker.color.b = 1.0;
  _textMarker.lifetime = rclcpp::Duration::from_seconds(0.1);

  this->_pubMarker =
      this->create_publisher<visualization_msgs::msg::MarkerArray>(
          "/detector/marker", 10);
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

  _subCameraInfoTopicName =
      this->declare_parameter<std::string>("subscribe_camera_info_topic_name");
  rcpputils::assert_true(!_subImageTopicName.empty());
  RCLCPP_INFO(get_logger(),
              "Set parameter subscribe_camera_info_topic_name: `%s`",
              _subCameraInfoTopicName.c_str());

  _pubObjectsTopicName = this->declare_parameter<std::string>(
      "publish_objects_topic_name", "detector/objects");
  RCLCPP_INFO(get_logger(), "Set parameter publish_objects_topic_name: `%s`",
              _pubObjectsTopicName.c_str());
  rcpputils::assert_true(!_pubObjectsTopicName.empty());

  _classLabelsPath = this->declare_parameter<std::string>("class_labels_path");
  rcpputils::assert_true(!_classLabelsPath.empty());
  RCLCPP_INFO(get_logger(), "Set parameter class_labels_path: `%s`",
              _classLabelsPath.c_str());

  _confBboxThresh = this->declare_parameter<float>("conf_bbox_thresh", 0.0f);
  RCLCPP_INFO(get_logger(), "Set parameter conf_bbox_thresh: %f",
              _confBboxThresh);

  _confNmsThresh = this->declare_parameter<float>("conf_nms_thresh", 0.0f);
  RCLCPP_INFO(get_logger(), "Set parameter conf_nms_thresh: %f",
              _confNmsThresh);
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

  rt_interfaces::msg::Objects objectsMsg;
  bboxToObjectsMsg(objectsMsg, objects, ptr->header);
  _pubObjects->publish(objectsMsg);

  if (this->_isPreview) {
    _inferEngine->drawObjects(frame, objects, classNames);

    /// Draw inference time on the image
    std::string timeText =
        "Inference: " + std::to_string(elapsed.count()) + "ms";
    cv::putText(frame, timeText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
                1.0, cv::Scalar(255, 255, 255), 2);

    cv::imshow("DetectorNode: Preview", frame);
    auto key = cv::waitKey(1);
    if (key == 'q') {
      rclcpp::shutdown();
    }
  }
}

void DetectorNode::bboxToObjectsMsg(rt_interfaces::msg::Objects &msg,
                                    std::vector<rt_vision::Object> &objects,
                                    const std_msgs::msg::Header &header) {
  rt_interfaces::msg::Object objectMsg;

  msg.header = header;
  _objectMarker.header = header;
  _textMarker.header = header;

  _markerArray.markers.clear();
  _objectMarker.id = 0;
  _textMarker.id = 0;

  for (const auto &obj : objects) {
    objectMsg.probability = obj.prob;
    objectMsg.class_id = classNames[obj.label];

    cv::Mat rvec, tvec;
    _pnpSolver->solvePnP(obj, rvec, tvec);

    /// fill pose
    objectMsg.pose.position.x = tvec.at<double>(0);
    objectMsg.pose.position.y = tvec.at<double>(1);
    objectMsg.pose.position.z = tvec.at<double>(2);

    /// rvec to 3x3 rotation matrix
    cv::Mat rotationMatrix;
    cv::Rodrigues(rvec, rotationMatrix);

    /// rotation matrix to quaternion
    tf2::Matrix3x3 tf2RotationMatrix(
        rotationMatrix.at<double>(0, 0), rotationMatrix.at<double>(0, 1),
        rotationMatrix.at<double>(0, 2), rotationMatrix.at<double>(1, 0),
        rotationMatrix.at<double>(1, 1), rotationMatrix.at<double>(1, 2),
        rotationMatrix.at<double>(2, 0), rotationMatrix.at<double>(2, 1),
        rotationMatrix.at<double>(2, 2));
    tf2::Quaternion tf2Q;
    tf2RotationMatrix.getRotation(tf2Q);
    objectMsg.pose.orientation = tf2::toMsg(tf2Q);

    /// fill the markkers
    ++_objectMarker.id;
    _objectMarker.scale.y = 0.135; 
    _objectMarker.pose = objectMsg.pose;

    ++_textMarker.id;
    _textMarker.pose.position = objectMsg.pose.position;
    _textMarker.pose.position.y -= 0.1;
    _textMarker.text = objectMsg.class_id;

    _markerArray.markers.emplace_back(_objectMarker);
    _markerArray.markers.emplace_back(_textMarker);

    msg.objects.emplace_back(objectMsg);
  }

  publishMarkers();
}

void DetectorNode::publishMarkers() { _pubMarker->publish(_markerArray); }

} // namespace rt_vision

#include "rclcpp_components/register_node_macro.hpp"

/// Register the component with class_loader.
/// This acts as a sort of entry point, allowing the component to be
/// discoverable when its library is being loaded into a running process.
RCLCPP_COMPONENTS_REGISTER_NODE(rt_vision::DetectorNode)

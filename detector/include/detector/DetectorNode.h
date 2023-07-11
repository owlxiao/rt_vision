#ifndef DETECTOR_DETECTORNODE_H
#define DETECTOR_DETECTORNODE_H

#include "rt_interfaces/msg/objects.hpp"
#include "vision/PnPSolver.h"
#include "vision/VisionDetector.h"

#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <visualization_msgs/msg/detail/marker__struct.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

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

  void publishMarkers(void);

private:
  bool _isPreview{};
  std::size_t _numClasses{};
  std::string _engineFilePath{};
  std::string _subImageTopicName{};
  std::string _subCameraInfoTopicName{};
  std::string _pubObjectsTopicName{};
  std::string _classLabelsPath{};
  float _confNmsThresh{0.0f};
  float _confBboxThresh{0.0f};

  std::unique_ptr<VisionDetector> _inferEngine{nullptr};
  std::vector<std::string> classNames{};

  image_transport::Subscriber _subImage;

  // rt_interfaces::msg::Objects _pubObjectsMsg;
  rclcpp::Publisher<rt_interfaces::msg::Objects>::SharedPtr _pubObjects;

  /// Camera info
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr _camInfoSub;
  std::shared_ptr<sensor_msgs::msg::CameraInfo> _camInfo;
  std::unique_ptr<PnPSolver> _pnpSolver;

  /// Visualization marker publisher
  visualization_msgs::msg::Marker _objectMarker;
  visualization_msgs::msg::Marker _textMarker;
  visualization_msgs::msg::MarkerArray _markerArray;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr _pubMarker;
};

} // namespace rt_vision

#endif
cmake_minimum_required(VERSION 3.8)
project(detector)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(vision REQUIRED)
find_package(OpenCV REQUIRED)

set(dependencies rclcpp rclcpp_components vision OpenCV )

add_library(detector_node SHARED src/DetectorNode.cpp)
ament_target_dependencies(detector_node ${dependencies})
rclcpp_components_register_nodes(detector_node
                                 "rt_vision::DetectorNode")


target_include_directories(detector_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(detector_node PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS detector_node
  DESTINATION lib)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

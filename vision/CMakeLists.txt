cmake_minimum_required(VERSION 3.8)
project(vision)

set(CMAKE_EXPORT_COMPILE_COMMANDS on)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)

# find dependencies
find_package(ament_cmake REQUIRED)

# Opencv Package
find_package(OpenCV)
# Opencv package include directory
include_directories(${OpenCV_INCLUDE_DIRS})


# CUDA Package
find_package(CUDA REQUIRED)

if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found on this system.")
endif()

# CUDA package include directory
include_directories(${CUDA_INCLUDE_DIRS})

set(SRC ${SRC} src/TensorRTDetector.cpp src/VisionDetector.cpp)

add_library(vision SHARED 
  ${SRC}
)

target_link_libraries(vision nvinfer)
target_link_libraries(vision ${OpenCV_LIBS}  ${CUDA_LIBRARIES})


target_include_directories(vision PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(vision PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS vision
  DESTINATION lib/${PROJECT_NAME})

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

ament_export_include_directories(include)
ament_package()

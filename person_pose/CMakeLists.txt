cmake_minimum_required(VERSION 3.8)
project(person_pose)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(person_pose_interfaces REQUIRED)
find_package(OpenCV REQUIRED)

# MMDeploy C++ SDK. Set MMDeploy_DIR (or CMAKE_PREFIX_PATH) to the SDK's
# lib/cmake/MMDeploy directory — see README.md for install instructions.
find_package(MMDeploy REQUIRED)

add_executable(person_pose src/person_pose.cpp)
target_include_directories(person_pose PRIVATE ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(person_pose rclcpp builtin_interfaces std_msgs sensor_msgs cv_bridge person_pose_interfaces)
# The prebuilt/installed MMDeploy SDK exports a plain `mmdeploy` shared target;
# backend modules (ort/trt) are loaded as plugins at runtime via LD_LIBRARY_PATH.
target_link_libraries(person_pose mmdeploy ${OpenCV_LIBS} pthread)

install(TARGETS person_pose
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

cmake_minimum_required(VERSION 2.8.3)
project(allegro_hand)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  allegro_hand_controllers
  #allegro_hand_driver
  #bhand
  #libpcan
  roscpp
  sensor_msgs
# roslaunch  # Disabled due to testing
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES allegro_hand_joint_pd allegro_hand_sim desired_joint_state_pub
  CATKIN_DEPENDS allegro_hand_controllers allegro_hand_driver bhand libpcan roscpp sensor_msgs
#  DEPENDS
)

###########
## Build ##
###########
message( ${catkin_INCLUDE_DIRS} )
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

## Declare executables

add_library(allegro_hand_joint_pd src/allegro_hand_joint_pd.cpp)

add_library(allegro_hand_sim src/allegro_hand_sim.cpp)

add_executable(allegro_hand_node src/allegro_hand_node.cpp)
target_link_libraries(allegro_hand_node allegro_hand_joint_pd allegro_hand_sim ${catkin_LIBRARIES})

add_executable(desired_joint_state_pub src/desired_joint_state_pub.cpp)
target_link_libraries(desired_joint_state_pub ${catkin_LIBRARIES})

# This is disabled until roslaunch tests can pass required arguments.
# roslaunch_add_file_check(launch/allegro_hand.launch)

#############
## Install ##
#############

install(TARGETS
  allegro_hand_joint_pd
  allegro_hand_sim
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

## Install launch files
install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/
  FILES_MATCHING PATTERN "*.launch"
)

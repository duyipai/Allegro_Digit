#ifndef __DESIRED_JOINT_STATE_PUB_H__
#define __DESIRED_JOINT_STATE_PUB_H__
#include <stdio.h>
#include <boost/thread/thread.hpp>

#include "ros/ros.h"
#include "allegro_hand_controllers/allegro_node.h"
class DesiredJointStatePub{

 public:
  DesiredJointStatePub();

  ~DesiredJointStatePub();

  // Uses the String received command to set the hand into its home
  // position, or saves the grasp in order to go into PD control mode. Also
  // can turn the hand off.
  void cmdCallback(const std_msgs::String::ConstPtr &msg);

  void jointStateCallback(const sensor_msgs::JointState &msg);

  void publishDesiredJointState(double * controlInput, bool isTorqueControl=false);
  // controlInput is either desired position (in radians) or desired torque depending on isTorqueControl

 protected:
  ros::Subscriber cmd_sub;

  ros::Subscriber joint_state_sub;

  ros::Publisher desired_state_pub;

  ros::NodeHandle nh;

  ros::Time position_time;

  double current_position[DOF_JOINTS] = {0.0};

  double current_velocity[DOF_JOINTS] = {0.0};

  double current_effort[DOF_JOINTS] = {0.0};

  double saved_position[DOF_JOINTS] = {0.0};

  boost::mutex *mutex;

};

#endif  // __DESIRED_JOINT_STATE_PUB_H__
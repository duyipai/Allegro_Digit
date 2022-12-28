using namespace std;

#include "allegro_hand/allegro_hand_sim.h"
#include <stdio.h>

#include "ros/ros.h"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

double home_pose[DOF_JOINTS] =
        {
                // Default (HOME) position (degrees), set at system start if
                // no 'initial_position.yaml' parameter is loaded.
                0.0, -10.0, 45.0, 45.0,  0.0, -10.0, 45.0, 45.0,
                5.0, -5.0, 50.0, 45.0, 60.0, 25.0, 15.0, 45.0
        };

// Constructor subscribes to topics.
AllegroNodeSim::AllegroNodeSim()
        : AllegroNode(true) {
  for (int i = 0; i < DOF_JOINTS; i++)
  {
        current_position_filtered[i] = DEGREES_TO_RADIANS(home_pose[i]);
        desired_joint_state.position.push_back(DEGREES_TO_RADIANS(home_pose[i]));
        }
  ROS_INFO("Simulated hand controller");
  joint_cmd_sub = nh.subscribe(
          DESIRED_STATE_TOPIC, 1, &AllegroNodeSim::setJointCallback, this);
}

void AllegroNodeSim::setJointCallback(const sensor_msgs::JointState &msg) {
  ROS_WARN_COND(!control_hand_, "Setting control_hand_ to True because of "
                "received JointState message");
  if (msg.effort.size()>0)
  {
    ROS_ERROR("Received JointState message with effort, but this is not "
              "supported by the simulated controller");
  }
  else if (msg.position.size()>0)
  {
    mutex->lock();
    desired_joint_state = msg;
    mutex->unlock();
  }
  else
  {
    ROS_ERROR("Received JointState message with no position or effort");
  }
  control_hand_ = true;
}

AllegroNodeSim::~AllegroNodeSim() {
  ROS_INFO("Sim controller node is shutting down");
}

void AllegroNodeSim::computeDesiredTorque() {
  // Just set current = desired.
  for (int idx = 0; idx < DOF_JOINTS; ++idx) {
    current_position_filtered[idx] = desired_joint_state.position[idx];
  }
}

void AllegroNodeSim::updateController() {
  // Update the controller.
  // Calculate loop time;
  tnow = ros::Time::now();
  dt = 1e-9 * (tnow - tstart).nsec;

  // When running gazebo, sometimes the loop gets called *too* often and dt will
  // be zero. Ensure nothing bad (like divide-by-zero) happens because of this.
  if(dt <= 0) {
    ROS_DEBUG_STREAM_THROTTLE(1, "AllegroNodeSim::updateController dt is zero.");
    return;
  }

  tstart = tnow;
  computeDesiredTorque();
  publishData();
}
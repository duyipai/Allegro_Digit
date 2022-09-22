#ifndef __ALLEGRO_HAND_JOINT_PD_H__
#define __ALLEGRO_HAND_JOINT_PD_H__

#include "allegro_hand_controllers/allegro_node.h"
#include "bhand/BHand.h"

// Joint-space PD control of the Allegro hand.
//
// Allows you to save a position and command it to the hand controller.
// Controller gains are loaded from the ROS parameter server.
class AllegroNodePD : virtual public AllegroNode {

 public:
  AllegroNodePD();

  ~AllegroNodePD();

  // Main spin code: just waits for messages.
  void doIt();

  void setJointCallback(const sensor_msgs::JointState &msg);

  // Loads all gains and initial positions from the parameter server.
  void initController(const std::string &whichHand);

  // PD control happens here.
  void computeDesiredTorque();

 protected:
  // Subscribe to desired joint states, only so we can set control_hand_ to true
  // when we receive a desired command.
  ros::Subscriber joint_cmd_sub;

  // If this flag is true, the hand will be controlled (either in joint position
  // or joint torques). If false, desired torques will all be zero.
  bool control_hand_ = false;

  BHand *pBHand = NULL;

   double desired_position[DOF_JOINTS] = {0.0};
};

#endif  // __ALLEGRO_HAND_JOINT_PD_H__

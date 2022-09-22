#include <stdio.h>

#include "ros/ros.h"
#include "allegro_hand/allegro_hand_joint_pd.h"
#include "allegro_hand/allegro_hand_sim.h"

int main(int argc, char **argv) {
  ros::init(argc, argv, "allegro_hand_core");
  
  AllegroNodePD* allegroNodePtr;
  if (argv[1] == std::string("true")) {
    allegroNodePtr = new AllegroNodeSim();
    ROS_INFO("Start Allegro Hand simulation.");
  } else {
    allegroNodePtr = new AllegroNodePD();
    ROS_INFO("Start Allegro Hand controller.");
  }
  allegroNodePtr->doIt();
}

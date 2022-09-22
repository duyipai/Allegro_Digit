using namespace std;

#include <stdio.h>

#include "ros/ros.h"
#include "allegro_hand/allegro_hand_joint_pd.h"

#define RADIANS_TO_DEGREES(radians) ((radians) * (180.0 / M_PI))
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

// Default parameters.
double k_p[DOF_JOINTS] =
        {
                // Default P Gains for PD Controller, loaded if
                // 'gains_pd.yaml' file is not loaded.
                600.0, 600.0, 600.0, 1000.0, 600.0, 600.0, 600.0, 1000.0,
                600.0, 600.0, 600.0, 1000.0, 1000.0, 1000.0, 1000.0, 600.0
        };

double k_d[DOF_JOINTS] =
        {
                // Default D Gains for PD Controller, loaded if
                // 'gains_pd.yaml' file is not loaded.
                15.0, 20.0, 15.0, 15.0, 15.0, 20.0, 15.0, 15.0,
                15.0, 20.0, 15.0, 15.0, 30.0, 20.0, 20.0, 15.0
        };

double home_pose[DOF_JOINTS] =
        {
                // Default (HOME) position (degrees), set at system start if
                // no 'initial_position.yaml' parameter is loaded.
                0.0, -10.0, 45.0, 45.0,  0.0, -10.0, 45.0, 45.0,
                5.0, -5.0, 50.0, 45.0, 60.0, 25.0, 15.0, 45.0
        };

std::string pGainParams[DOF_JOINTS] =
        {
                "~gains_pd/p/j00", "~gains_pd/p/j01", "~gains_pd/p/j02",
                "~gains_pd/p/j03",
                "~gains_pd/p/j10", "~gains_pd/p/j11", "~gains_pd/p/j12",
                "~gains_pd/p/j13",
                "~gains_pd/p/j20", "~gains_pd/p/j21", "~gains_pd/p/j22",
                "~gains_pd/p/j23",
                "~gains_pd/p/j30", "~gains_pd/p/j31", "~gains_pd/p/j32",
                "~gains_pd/p/j33"
        };

std::string dGainParams[DOF_JOINTS] =
        {
                "~gains_pd/d/j00", "~gains_pd/d/j01", "~gains_pd/d/j02",
                "~gains_pd/d/j03",
                "~gains_pd/d/j10", "~gains_pd/d/j11", "~gains_pd/d/j12",
                "~gains_pd/d/j13",
                "~gains_pd/d/j20", "~gains_pd/d/j21", "~gains_pd/d/j22",
                "~gains_pd/d/j23",
                "~gains_pd/d/j30", "~gains_pd/d/j31", "~gains_pd/d/j32",
                "~gains_pd/d/j33"
        };

std::string initialPosition[DOF_JOINTS] =
        {
                "~initial_position/j00", "~initial_position/j01",
                "~initial_position/j02",
                "~initial_position/j03",
                "~initial_position/j10", "~initial_position/j11",
                "~initial_position/j12",
                "~initial_position/j13",
                "~initial_position/j20", "~initial_position/j21",
                "~initial_position/j22",
                "~initial_position/j23",
                "~initial_position/j30", "~initial_position/j31",
                "~initial_position/j32",
                "~initial_position/j33"
        };

// Constructor subscribes to topics.
AllegroNodePD::AllegroNodePD()
        : AllegroNode() {
  initController(whichHand);

  joint_cmd_sub = nh.subscribe(
          DESIRED_STATE_TOPIC, 1, &AllegroNodePD::setJointCallback, this);
}

AllegroNodePD::~AllegroNodePD() {
  ROS_INFO("PD controller node is shutting down");
  if (pBHand != NULL) {
    delete pBHand;
  }
}


void AllegroNodePD::setJointCallback(const sensor_msgs::JointState &msg) {
  ROS_WARN_COND(!control_hand_, "Setting control_hand_ to True because of "
                "received JointState message");
  control_hand_ = true;
  mutex->lock();
  desired_joint_state = msg;
  mutex->unlock();
}

void AllegroNodePD::computeDesiredTorque() {

  // No control: set torques to zero.
  if (!control_hand_) {
    //ROS_INFO_THROTTLE(1.0, "Hand control is false");
    for (int i = 0; i < DOF_JOINTS; i++) {
      desired_torque[i] = 0.0;
    }
    return;
  }

  // Sanity/defensive check: if *both* position and torques are set in the
  // message, do nothing.
  if (desired_joint_state.position.size() > 0 &&
      desired_joint_state.effort.size() > 0) {
    ROS_WARN("Error: both positions and torques are specified in the desired "
                     "state. You cannot control both at the same time.");
    return;
  }

  pBHand->SetJointPosition(current_position_filtered);
  {
    mutex->lock();

    if (desired_joint_state.position.size() == DOF_JOINTS) {
      // Control joint positions: compute the desired torques (PD control).
      for (int i = 0; i < DOF_JOINTS; i++) {
        desired_position[i] = desired_joint_state.position[i];
      }
      pBHand->SetJointDesiredPosition(desired_position);
      pBHand->UpdateControl((double)frame * ALLEGRO_CONTROL_TIME_INTERVAL);
      pBHand->GetJointTorque(desired_torque);
    }
    else if (desired_joint_state.effort.size() > 0)
    {
      // Control joint torques: set desired torques as the value stored in the
      // desired_joint_state message.
      for (int i = 0; i < DOF_JOINTS; i++) {
        desired_torque[i] = desired_joint_state.effort[i];
      }
    }
    mutex->unlock();
  }
  // ROS_INFO("Control delay is %f seconds", tnow.toSec()-desired_joint_state.header.stamp.toSec());
}

void AllegroNodePD::initController(const std::string &whichHand) {
  // Initialize BHand controller
  if (whichHand.compare("left") == 0) {
    pBHand = new BHand(eHandType_Left);
    ROS_WARN("CTRL: Left Allegro Hand controller initialized.");
  }
  else {
    pBHand = new BHand(eHandType_Right);
    ROS_WARN("CTRL: Right Allegro Hand controller initialized.");
  }
  pBHand->SetTimeInterval(ALLEGRO_CONTROL_TIME_INTERVAL);
  pBHand->SetMotionType(eMotionType_JOINT_PD);
  // set gains_pd via gains_pd.yaml or to default values
  if (ros::param::has("~gains_pd")) {
    ROS_INFO("CTRL: PD gains loaded from param server.");
    for (int i = 0; i < DOF_JOINTS; i++) {
      ros::param::get(pGainParams[i], k_p[i]);
      ros::param::get(dGainParams[i], k_d[i]);
    }
  }
  else {
    // gains will be loaded every control iteration
    ROS_WARN("CTRL: PD gains not loaded");
    ROS_WARN("Check launch file is loading /parameters/gains_pd.yaml");
    ROS_WARN("Loading default PD gains...");
  }
  // set initial position via initial_position.yaml or to default values
  if (ros::param::has("~initial_position")) {
    ROS_INFO("CTRL: Initial Pose loaded from param server.");
    double tmp;
    mutex->lock();
    desired_joint_state.position.resize(DOF_JOINTS);
    for (int i = 0; i < DOF_JOINTS; i++) {
      ros::param::get(initialPosition[i], tmp);
      desired_joint_state.position[i] = DEGREES_TO_RADIANS(tmp);
    }
    mutex->unlock();
  }
  else {
    ROS_WARN("CTRL: Initial position not loaded.");
    ROS_WARN("Check launch file is loading /parameters/initial_position.yaml");
    ROS_WARN("Loading Home position instead...");

    // Home position
    mutex->lock();
    desired_joint_state.position.resize(DOF_JOINTS);
    for (int i = 0; i < DOF_JOINTS; i++)
      desired_joint_state.position[i] = DEGREES_TO_RADIANS(home_pose[i]);
    mutex->unlock();
  }
  control_hand_ = false;

  printf("*************************************\n");
  printf("      Joint PD Control Method        \n");
  printf("-------------------------------------\n");
  printf("  Only 'H', 'O', 'S', 'Space' works. \n");
  printf("*************************************\n");
}

void AllegroNodePD::doIt() {
  // Main spin loop, uses the publisher/subscribers.
  while (ros::ok()) {
    updateController();
    ros::spinOnce();
  }
}

int main(int argc, char **argv) {
  ros::init(argc, argv, "allegro_hand_pd_controller");
  AllegroNodePD allegroNode;

  allegroNode.doIt();
}
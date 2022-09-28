#include "allegro_hand/desired_joint_state_pub.h"
#define DEGREES_TO_RADIANS(angle) ((angle) / 180.0 * M_PI)

double home_pose[DOF_JOINTS] =
        {
                // Default (HOME) position (degrees), set at system start if
                // no 'initial_position.yaml' parameter is loaded.
                0.0, DEGREES_TO_RADIANS(-10.0), DEGREES_TO_RADIANS(45.0), DEGREES_TO_RADIANS(45.0),  0.0, DEGREES_TO_RADIANS(-10.0),
                DEGREES_TO_RADIANS(45.0), DEGREES_TO_RADIANS(45.0), DEGREES_TO_RADIANS(5.0), DEGREES_TO_RADIANS(-5.0),
                DEGREES_TO_RADIANS(50.0), DEGREES_TO_RADIANS(45.0), DEGREES_TO_RADIANS(60.0), DEGREES_TO_RADIANS(25.0),
                DEGREES_TO_RADIANS(15.0), DEGREES_TO_RADIANS(45.0)
        };

double zero_torque[DOF_JOINTS] ={0.0};
double pre_grasp[DOF_JOINTS] = {0.0, 0.0, 0.0, 0.0, 0.3, 1.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.9, -0.16};
double grasp[DOF_JOINTS] = {0.0, 0.0, 0.0, 0.0, 0.3, 1.6, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.9, -0.16};
double roll1[DOF_JOINTS] = {0.0, 0.0, 0.0, 0.0, 0.3, 1.6, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.9, -0.16};
double roll2[DOF_JOINTS] = {0.0, 0.0, 0.0, 0.0, 0.3, 1.6, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 1.4, 0.0, 0.9, -0.16};
int status = 0;

DesiredJointStatePub::DesiredJointStatePub()
{
  mutex = new boost::mutex();
  cmd_sub = nh.subscribe(
    LIB_CMD_TOPIC, 1, &DesiredJointStatePub::cmdCallback, this);
  desired_state_pub = nh.advertise<sensor_msgs::JointState>(DESIRED_STATE_TOPIC, 3);
  joint_state_sub = nh.subscribe(
    JOINT_STATE_TOPIC, 1, &DesiredJointStatePub::jointStateCallback, this);
}

DesiredJointStatePub::~DesiredJointStatePub() {
  ROS_INFO("Control logic node is shutting down");
  delete mutex;
  nh.shutdown();
}

// Called when an external (string) message is received
void DesiredJointStatePub::cmdCallback(const std_msgs::String::ConstPtr &msg) {
  ROS_INFO("CTRL: Heard: [%s]", msg->data.c_str());
  const std::string lib_cmd = msg->data.c_str();

  if (lib_cmd.compare("pdControl") == 0) {
    // Desired position only necessary if in PD Control mode
    // publishDesiredJointState(saved_position, false);
    switch (status) {
      case 0:
        publishDesiredJointState(home_pose, false);
        status += 1;
        break;
      case 1:
        publishDesiredJointState(pre_grasp, false);
        status += 1;
        break;
      case 2:
        publishDesiredJointState(grasp, false);
        status += 1;
        break;
      case 3:
        publishDesiredJointState(roll1, false);
        status += 1;
        break;
      case 4:
        publishDesiredJointState(roll2, false);
        status -= 1;
        break;
      default:
        ROS_ERROR("CTRL: Invalid status of %d", status);
        break;
    }
  } else if (lib_cmd.compare("save") == 0) {
    mutex->lock();
    for (int i = 0; i < DOF_JOINTS; i++)
      saved_position[i] = current_position[i];
    mutex->unlock();
  } else if (lib_cmd.compare("off") == 0) {
    publishDesiredJointState(zero_torque, true);
  } else if (lib_cmd.compare("home") == 0) {
    publishDesiredJointState(home_pose, false);
    status = 1;
  } else {
    ROS_WARN("Unknown commanded grasp: %s.", lib_cmd.c_str());
    return;
  }
}

void DesiredJointStatePub::jointStateCallback(const sensor_msgs::JointState &msg) {
  // Store the current joint states.

  for (int i = 0; i < DOF_JOINTS; i++)
  {
    current_position[i] = msg.position[i];
    current_velocity[i] = msg.velocity[i];
    current_effort[i] = msg.effort[i];
  }

}

void DesiredJointStatePub::publishDesiredJointState(double *controlInput, bool isTorqueControl)
{
  // Publish the desired joint state.
  sensor_msgs::JointState desired_joint_state;
  desired_joint_state.header.stamp = ros::Time::now();
  mutex->lock();
  if (isTorqueControl)
  {
    for (int i = 0; i < DOF_JOINTS; i++)
    {
      desired_joint_state.effort.push_back(controlInput[i]);
    }
    desired_joint_state.position.resize(0);
  }
  else
  {
    for (int i = 0; i < DOF_JOINTS; i++)
    {
      desired_joint_state.position.push_back(controlInput[i]);
    }
    desired_joint_state.effort.resize(0);
  }
  mutex->unlock();
  desired_state_pub.publish(desired_joint_state);
}

int main(int argc, char **argv) {

  ros::init(argc, argv, "allegro_hand_control_logic");
  ros::NodeHandle nh;
  ros::Rate r(10);
  DesiredJointStatePub desired_joint_state_pub;

  while(ros::ok()) {
    ros::spinOnce();
    r.sleep();
  }
  
}

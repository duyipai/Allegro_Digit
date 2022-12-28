#ifndef __ALLEGRO_HAND_SIM_H__
#define __ALLEGRO_HAND_SIM_H__

#include "allegro_hand/allegro_hand_joint_pd.h"

// Simulated Allegro Hand.
//
// Simulated is probably a generous term, this is simply a pass-through for
// joint states: commanded -> current.
//
class AllegroNodeSim : public AllegroNodePD {

 public:
  AllegroNodeSim();

  ~AllegroNodeSim();

  void computeDesiredTorque();

  void updateController();

  void setJointCallback(const sensor_msgs::JointState &msg);

  protected:
};

#endif  // __ALLEGRO_HAND_SIM_H__
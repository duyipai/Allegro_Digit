# Allegro_Digit

Allegro Digit ROS Package
================================

This is from the official allegro hand ros package [1].

[1] https://github.com/simlabrobotics/allegro_hand_ros_v4

It simplifies the launch file structure, seperating the low level PD controller and the control logic by introducing a DesiredJointStatePub node that implements the high level control. It is also made to be compatiable with ROS neotic.

It also requires the BHand library installed [2].

[2] http://wiki.wonikrobotics.com/AllegroHandWiki/index.php/BHand_library_API


Launch file instructions:
------------------------

There is now a single file,
[allegro_hand.launch](allegro_hand/launch/allegro_hand.launch) that starts the hand.

Note on `AUTO_CAN`: There is a nice script `detect_pcan.py` which automatically
finds an open `/dev/pcanusb` file. If instead you specify the can device
manually (`CAN_DEVICE:=/dev/pcanusbN`), make sure you *also* specify
`AUTO_CAN:=false`. 

Packages
--------

 * **allegro_hand** Low level PD control and high level DesiredJointStatePub.

 * packages from the official ROS repo [1].


Installing the PCAN driver
--------------------------

Before using the hand, you must install the pcan drivers. This assumes you have
a peak-systems pcan to usb adapter.

1. Install these packages

    sudo apt-get install libpopt-dev ros-neotic-libpcan

2. Download latest drivers: http://www.peak-system.com/fileadmin/media/linux/index.htm#download

Install the drivers:

    make clean; make NET=NO_NETDEV_SUPPORT
    sudo make install
    sudo /sbin/modprobe pcan (if errors are showing, make sure secure boot is disabled)

Test that the interface is installed properly with:

     cat /proc/pcan

You should see some stuff streaming.

When the hand is connected, you should see pcanusb0 or pcanusb1 in the list of
available interfaces:

    ls -l /dev/pcan*


3.
Build the sources:

    catkin_make    
    source devel/setup.bash

4. 
quick start:

    roslaunch allegro_hand allegro_hand.launch


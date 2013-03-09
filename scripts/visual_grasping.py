#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import visual_grasping.visual_grasper
import openrave_bridge.env_operations
from sensor_msgs.msg import PointCloud2, PointCloud
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
import tabletop_actions
import sys
from pr2_control_utilities import pr2_joint_mover

if __name__ == "__main__":

    if len(sys.argv) == 1:
        print "Usage: %s [bolt | baseline | rest]"
        sys.exit(1)                
    elif sys.argv[1] not in ["bolt" , "baseline" , "rest"]:
        print "Usage: %s [bolt | baseline | rest]"
        sys.exit(1)        

    rospy.init_node("visual_grasping")
    if sys.argv[1] == "rest":
        control = pr2_joint_mover.PR2JointMover()
        control.open_left_gripper()
        control.open_right_gripper()
        control.set_arm_state([1.4169820643, 
                               -0.349340503599, 
                               0.860236384523, 
                               -2.0390671803, 
                               -4.12956559472, 
                               -0.96061705675, 
                               2.07003890695], "l_arm")
        control.set_arm_state([-1.82797811213,
                               -0.330749630007, 
                               -1.00800414828, 
                               -1.43693185785, 
                               5.18324409043, 
                               -1.99975352354, 
                               -21.9234625261],
                              "r_arm")
        sys.exit()
    
    
    rospy.loginfo("Creating the robot")
    robot = openrave_bridge.pr2model.PR2Robot()
    rospy.loginfo("Finding the table")
    detector = tabletop_actions.object_detector.GenericDetector()
    
    detector.detect();
    if detector.last_detection_msg is None:
        rospy.logerr("No table detected, the robot might hit the table!")
    else:    
        openrave_bridge.env_operations.add_table(
            detector.last_detection_msg, robot.env)
    
    vg = visual_grasping.visual_grasper.VisualGrasper(robot)    
    if sys.argv[1] == "bolt":    
        vg.work_on_bolt_pointcloud()
    elif sys.argv[1] == "baseline":
        vg.standard_grasping(pullup=True)
        
    


#!/usr/bin/env python
import roslib
roslib.load_manifest("visual_grasping")
import rospy
import utils
import openrave_bridge.pr2model
from geometry_msgs.msg import Pose, PoseStamped
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
import tf
import copy
import numpy as np
import time
from pr2_control_utilities import IKUtilities
from object_manipulation_msgs.srv import (GraspPlanning,
                                          GraspPlanningRequest, GraspPlanningResponse)

class VisualGrasper(object):
    def __init__(self, robot = None):
        if robot is None:
            self.robot = openrave_bridge.pr2model.PR2Robot()
        else:
            self.robot = robot
            
        self.gripper_pub = rospy.Publisher("~gripper_estimate", MarkerArray)    
        self.tf_listener = tf.TransformListener()
        #self.left_ik = IKUtilities("left",
                                   #tf_listener=self.tf_listener)
        self.grap_planning_srv = None
        
    
    def gripper_from_points(self, p0, p1, frame_id = "/base_link"):
        """
        p0: [x,y,z]
        p1: [x,y,z]
        
        returns: a geometry_msgs PoseStamped
        """
        
        p1 =  np.asarray(p1)
        p0 = np.asarray(p0)
        dest = p0
        T = utils.make_orth_basis_z_ax(p1 - p0)

        #moving the gripper backward so that the fingers touch the point
        T[:3,3] = dest
        fingertip = T.dot([-0.16,0,0,1])
        T[:,3] = fingertip

        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose = utils.matrixToPose(T)
        return ps
    
    def publish_gripper_pose(self, pose):
        assert isinstance(pose, PoseStamped)
        gripper = utils.makeGripperMarker(pose)
        self.gripper_pub.publish(gripper)
        
    
    def point_at_gripper(self, target,
                         desired_dist, 
                         gripper_x,
                         min_gripper_y, 
                         max_gripper_y,
                         n_attempts=1000,
                         n_poses = 100,
                         constrain_y = lambda _ :True,
                         constrain_z = lambda _: True                          
                          ):
        """Given a target, move the gripper so that it's always
        pointing at its pose while keeping the desired_dist.
        """
        assert isinstance(target, PoseStamped)
        
        if target.header.frame_id != "/base_link":
            rospy.loginfo("Changing the frame from %s to %s", target.header.frame_id,
                          "/base_link")
            self.tf_listener.waitForTransform("/base_link", target.header.frame_id,
                                              rospy.Time.now(), rospy.Duration(1))
            target = self.tf_listener.transformPose("/base_link", target)
            
        tx = target.pose.position.x
        ty = target.pose.position.y
        tz = target.pose.position.z

        gripper_xyz = utils.valid_sphere_given_one_coord(desired_dist,
                                                         gripper_x,
                                                         min_gripper_y,
                                                         max_gripper_y,
                                                         (tx, ty, tz),
                                                         n_poses, 
                                                         n_attempts,
                                                         constrain_y,
                                                         constrain_z
                                                         )
        if len(gripper_xyz) == 0:
            rospy.logerr("Could not find a valid solution")
            return []
        all_poses = []
        for gripper_x, gripper_y, gripper_z in gripper_xyz:
            vec = (-gripper_x + tx,
                   -gripper_y + ty,
                   -gripper_z + tz,
                   )
            
            rot_mat = utils.make_orth_basis(vec)
            
            M = np.identity(4)
            M[:3, :3] = rot_mat
            M[:3, 3] = (gripper_x, gripper_y, gripper_z)
            pos = utils.matrixToPose(M)
            
            ps = PoseStamped()
            ps.header.frame_id = "/base_link"
            ps.pose = pos
            all_poses.append(ps)
        return all_poses
    
    def grasp_bowl(self, p0, x_angle, frame_id ="/base_link",
                   side_grasp = False):
        dest = np.asarray(p0)
        
        if side_grasp:
            rospy.loginfo("Doing a side grasp")
            Q = utils.transformations.quaternion_from_euler(
                0, 
                0, 
                x_angle-np.pi/2)
        else:
            rospy.loginfo("Doing a top grasp")
            Q = utils.transformations.quaternion_from_euler(
                x_angle, 
                np.pi/2, 
                0)
        
        rospy.loginfo("Got a quaternion of \n%s", Q)
        T = utils.transformations.quaternion_matrix(Q)
        T[:3, 3] = dest            
        T[2,3] += 0.16
        
        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.pose = utils.matrixToPose(T)
        return ps        
    
    def visible_trajectory(self, target, desired_dist):
        self.robot.controller.time_to_reach = 5.0
        ddist = (desired_dist-0.01)*.7
        for gripper_x in np.linspace(target.pose.position.x - ddist, 
                                     target.pose.position.x + ddist,
                                     20):
        
            min_gripper_y = target.pose.position.y - ddist
            max_gripper_y = target.pose.position.y + ddist
            contrain_z = lambda z: z>target.pose.position.z+0.1
            all_gripper_ps = self.point_at_gripper(target, 
                                                   desired_dist, 
                                                   gripper_x, 
                                                   min_gripper_y,
                                                   max_gripper_y,
                                                   n_poses=20,
                                                   constrain_z=contrain_z)
            list_of_joints = []
            for gripper_ps in all_gripper_ps:
                M = utils.poseTomatrix(gripper_ps.pose)
                x,y,z = utils.transformations.euler_from_matrix(M)
                x += np.pi
                M2 = utils.transformations.euler_matrix(x,y,z)
                M[:3, :3] = M2[:3,:3]
                newpos = utils.matrixToPose(M)            
                gripper_ps.pose = newpos
                self.publish_gripper_pose(gripper_ps)

                curr_joint_angles = self.robot.controller.robot_state.left_arm_positions
                curr_joint_angles = np.array(curr_joint_angles)
                #joints, _ = self.left_ik.run_ik(gripper_ps,
                                                #curr_joint_angles,
                                                #"l_wrist_roll_link",
                                                #collision_aware=0)
                sols = self.robot.find_leftarm_ik(gripper_ps,
                                                            ignore_end_effector=False,
                                                            multiple_soluitions = True)
                if sols is not None and len(sols) != 0:
                    list_of_joints.extend(sols)
            list_of_joints = np.array(list_of_joints)
            if list_of_joints is not None and len(list_of_joints) != 0:
                print "Got %s solutions" % (list_of_joints.shape, )
                dists = np.sum((list_of_joints - curr_joint_angles)**2, 1)
                i = np.argmin(dists)
                joints = list_of_joints[i, :]
                
                self.robot.controller.time_to_reach = 1.5
                self.robot.controller.set_arm_state(joints, "left", wait=True)
        self.robot.controller.time_to_reach = 5.0

    def plan_grasp(self, graspable,
                   which_arm,
                          graspable_name = "",
                          table_name = "",
                          ):
        """Picks up a previously detected object.

        Parameters:
        graspable: an object_manipulation_msgs/GraspableObject msg instance.
         This usually comes from a Detector.call_collision_map_processing call.
        graspable_name: the name of the object to graps. It is provided by
         Detector.call_collision_map_processing.
        table_name: the name of the table. Again provided by Detector.call_collision_map_processing.
        which_arm: left_arm or right_arm            

        Return:
        a object_manipulation_msgs.GraspPlanningResponse msg
        """
        if self.grap_planning_srv is None:
            srv_name =  "/plan_point_cluster_grasp"
            rospy.loginfo("Waiting for service %s", srv_name)
            rospy.wait_for_service(srv_name)
            self.grap_planning_srv = rospy.ServiceProxy(srv_name,
                                                        GraspPlanning)
        rospy.loginfo("Calling the grasp planning service")
        gp = GraspPlanningRequest()
        gp.arm_name = which_arm
        gp.target = graspable
        gp.collision_object_name = graspable_name
        gp.collision_support_surface_name = table_name

        res = self.grap_planning_srv(gp)
        isinstance(res, GraspPlanningResponse)
        if res.error_code.value != res.error_code.SUCCESS:
            rospy.logerr("Could not find valid grasps!")
            return None
        else:
            grasps = sorted(res.grasps, key = lambda g:g.success_probability)
            res.grasps = grasps
            return res 
    
    def publish_grasps(self, grasp_planning_response, frame_id="/base_link",
                       sleeping_time = 0.05):
        assert isinstance(grasp_planning_response, GraspPlanningResponse)
        for g in grasp_planning_response.grasps:
            p = PoseStamped()
            p.pose = g.grasp_pose
            p.header.frame_id = frame_id
            self.publish_gripper_pose(p)
            time.sleep(sleeping_time)
            
    def grab_pointcloud(self, whicharm="leftarm",
                                pc = None,
                                pullup = False):
        if pc is None:
            rospy.loginfo("Waiting for BOLT pointcloud")
            pc = rospy.wait_for_message("/bolt/vision/pcl_robot", PointCloud2)
            rospy.loginfo("Ok got it!")
        xyz, rgb = utils.pc2xyzrgb(pc)
        weights = rgb[0,:,0]
        
        sorted_indexes = np.argsort(weights)[::-1]
        weights = weights[sorted_indexes]
        xyz = xyz[0,:,:]
        xyz = xyz[sorted_indexes,:]
        rospy.loginfo("The maximum weight is %f", weights.max())        
        rospy.loginfo("The min weight is %f", weights.min())                
        
        trim_weights = 20
        rospy.loginfo("Taking only the maximum %d weights", trim_weights)
        weights = weights[:trim_weights]
        xyz = xyz[:trim_weights,:] 
        
        rospy.loginfo("XYZ shape %s, W shape: %s", xyz.shape, weights.shape)

        rospy.loginfo("The maximum weight is %f", weights.max())        
        rospy.loginfo("The min weight is %f", weights.min())        
        rospy.loginfo("Calculating standard grasping points")
        graspable = utils.pc2graspable(pc)
        grasps = self.plan_grasp(graspable, whicharm)
        self.publish_grasps(grasps, sleeping_time=0.2)
        return        
        if grasps is None:
            rospy.logerr("No grasping poses found!")
            return None
        
        #self.publish_grasps(grasps,sleeping_time = 1.0)
        
        rospy.loginfo("reweighting grasps")        
        

        grasp_xyz = np.array([[g.grasp_pose.position.x,
                               g.grasp_pose.position.y,
                               g.grasp_pose.position.z,]
                              for g in grasps.grasps])
        
        #grasping_poses = []
        #for i, (x,y,z) in enumerate(xyz):
            #closest = np.argmin(np.abs(grasp_xyz-(x,y,z)).sum(1))            
            #g = grasps.grasps[closest]
            #grasping_poses.append(g)
            #p = PoseStamped()
            #p.pose = g.grasp_pose
            #p.header.frame_id = "/base_link"
            #self.publish_gripper_pose(p)
            #time.sleep(0.2)
        
        grasping_scores_poses = []
        for g in grasps.grasps:
            gx = g.grasp_pose.position.x
            gy = g.grasp_pose.position.y
            gz = g.grasp_pose.position.z
            closest = np.argmin(np.abs(xyz-(gx,gy,gz - 0.18 )).sum(1))
            w = weights[closest]
            if w > 0:
                grasping_scores_poses.append((w, g))
        
        grasping_poses = [g[1] for g in reversed(sorted(grasping_scores_poses))]
        
        rospy.loginfo("We have a total of %d grasps", len(grasping_poses))
        success = False
        for g in grasping_poses:            
            ps = PoseStamped()
            ps.header.frame_id = pc.header.frame_id
            ps.pose = g.grasp_pose
            self.publish_gripper_pose(ps)
            
            approach = copy.deepcopy(ps)
            approach.pose.position.z += 0.1            

            if whicharm == "leftarm":
                self.robot.move_left_arm(approach)
                self.robot.controller.open_left_gripper(True)                
                success = self.robot.move_left_arm(ps)
                self.robot.controller.close_left_gripper(True)
                if pullup:
                    self.robot.move_left_arm(approach)
            else:
                self.robot.move_right_arm(approach)
                self.robot.controller.open_right_gripper()
                success = self.robot.move_right_arm(ps)
                self.robot.controller.close_right_gripper()    
                if pullup:
                    self.robot.move_right_arm(approach)
            
            if success:
                break
            
    def do_the_grasp(self, pc = None, angle=None, pullup=False, side_grasp = False):
        if pc is None:
            bolt_points = rospy.wait_for_message("/bolt/vision/pcl_robot", 
                                             PointCloud2)
            rospy.loginfo("received the points")
        else:
            bolt_points = pc
        xyz, rgb = utils.pc2xyzrgb(bolt_points)
        if angle is None:
            p0 = xyz[0,0,:]
            angle = rgb[0,0,2] /255. * 360.
        #angle = all_points[1,0]*360. / 255.
        #rospy.loginfo("Raw angle: %s", all_points)
        angle = np.deg2rad(angle + 90)
        ps = self.grasp_bowl(p0, angle, side_grasp=side_grasp)
        self.publish_gripper_pose(ps)
        approach = copy.deepcopy(ps)
        approach.pose.position.z += 0.1
        #if ps.pose.position.y > 0:
        if True:
            self.robot.move_left_arm(approach)
            self.robot.controller.open_left_gripper(True)
            
            if self.robot.move_left_arm(ps):
                self.robot.controller.close_left_gripper(True)
                if pullup:
                    self.robot.move_left_arm(approach)
        else:
            self.robot.move_right_arm(approach)
            self.robot.controller.open_right_gripper(True)
            if self.robot.move_right_arm(ps):
                self.robot.controller.close_right_gripper(True)    
                if pullup:
                    self.robot.move_right_arm(approach)
        return ps        
        
    def work_on_bolt_pointcloud(self, pc = None):
        if pc is None:
            rospy.loginfo("Waiting for BOLT pointcloud")
            pc = rospy.wait_for_message("/bolt/vision/pcl_robot", PointCloud2)
            rospy.loginfo("Ok got it!")
        
        length = pc.width * pc.height
        if length == 1:
            rospy.loginfo("Old grasp with a single point")
            self.do_the_grasp(pc, pullup=True)
        else:
            rospy.loginfo("Grasping a pointcloud")
            #self.grab_pointcloud(pc=pc, pullup=True)  
            self.generate_scored_grasps(pc=pc, pullup=True) 
            
            
    def standard_grasping(self, pc = None, whicharm="leftarm",
                          pullup=False
                          ):
        if pc is None:
            rospy.loginfo("Waiting for BOLT pointcloud")
            pc = rospy.wait_for_message("/bolt/vision/pcl_robot", PointCloud2)
            rospy.loginfo("Ok got it!")
        
        rospy.loginfo("Calculating standard grasping points")
        graspable = utils.pc2graspable(pc)
        grasps = self.plan_grasp(graspable, whicharm)
        if grasps is None:
            rospy.logerr("No grasping poses found!")
            return None
        
        grasp_tuples = [(g.success_probability, g) for g in grasps.grasps]
        sorted_grasps = [g[1] for g in reversed(sorted(grasp_tuples))]
        for g in sorted_grasps:
            rospy.loginfo("Testing a grasp")
            ps = PoseStamped()
            ps.header.frame_id = pc.header.frame_id
            ps.pose = g.grasp_pose
            self.publish_gripper_pose(ps)
            
            approach = copy.deepcopy(ps)
            approach.pose.position.z += 0.1            

            if whicharm == "leftarm":
                self.robot.move_left_arm(approach)
                self.robot.controller.open_left_gripper(True)                
                success = self.robot.move_left_arm(ps)
                if success:
                    self.robot.controller.close_left_gripper(True)
                if pullup:
                    self.robot.move_left_arm(approach)
            else:
                self.robot.move_right_arm(approach)
                self.robot.controller.open_right_gripper(True)
                success = self.robot.move_right_arm(ps)
                if success:
                    self.robot.controller.close_right_gripper(True)    
                if pullup:
                    self.robot.move_right_arm(approach)
            
            if success:
                break        
    
    def generate_scored_grasps(self, pc = None, whicharm = "leftarm",
                               pullup = False):
        if pc is None:
            rospy.loginfo("Waiting for BOLT pointcloud")
            pc = rospy.wait_for_message("/bolt/vision/pcl_robot", PointCloud2)
            rospy.loginfo("Ok got it!")
            
        xyz, rgb = utils.pc2xyzrgb(pc)
        weights = rgb[0,:,0]
        
        sorted_indexes = np.argsort(weights)[::-1]
        weights = weights[sorted_indexes]
        xyz = xyz[0,:,:]
        xyz = xyz[sorted_indexes,:]
        rospy.loginfo("Initially The maximum weight is %f", weights.max())        
        rospy.loginfo("Initially The min weight is %f", weights.min())                
        
        trim_weights = 5
        rospy.loginfo("Taking only the maximum %d weights", trim_weights)
        weights = weights[:trim_weights]
        xyz = xyz[:trim_weights,:] 
        
        rospy.loginfo("XYZ shape %s, W shape: %s", xyz.shape, weights.shape)
        rospy.loginfo("After trimming The maximum weight is %f", weights.max())        
        rospy.loginfo("After trimming The min weight is %f", weights.min()) 
        
        rospy.wait_for_service('evaluate_point_cluster_grasps')
        evaluate_grasps = rospy.ServiceProxy('evaluate_point_cluster_grasps', 
                                             GraspPlanning)
        
        grasp_planning = GraspPlanningRequest()
        grasps = []
        for coords in xyz:
            _g = utils.create_spaced_downward_grasps(coords, pc,40)
            grasps.extend(_g)
        rospy.loginfo("Testing a total of %d grasps", len(grasps))
        graspable = utils.pc2graspable(pc)
        grasp_planning.target = graspable
        grasp_planning.grasps_to_evaluate = grasps
        res = evaluate_grasps(grasp_planning)
        grasps = res.grasps
        
        non_zero_grasps = sorted([g for g in grasps 
                                  if g.success_probability > 0],
                                 key = lambda g:g.success_probability,
                                 reverse=True)
        if len(non_zero_grasps) == 0:
            rospy.logerr("Error: no valid grasps found!")
            return False
        
        
        probs = [g.success_probability 
                 for g in non_zero_grasps]
        rospy.loginfo("probabilities are: %s", probs)
        rospy.loginfo("Number of non-zero %d grasps", len(non_zero_grasps))
        #self.publish_grasps(res)
        
        for g in non_zero_grasps:
            rospy.loginfo("Testing a grasp")
            ps = PoseStamped()
            ps.header.frame_id = pc.header.frame_id
            ps.pose = g.grasp_pose
            self.publish_gripper_pose(ps)
            
            approach = copy.deepcopy(ps)
            approach.pose.position.z += 0.1            

            if whicharm == "leftarm":
                self.robot.move_left_arm(approach)
                self.robot.controller.open_left_gripper(True)                
                success = self.robot.move_left_arm(ps)
                if success:
                    self.robot.controller.close_left_gripper(True)
                if pullup:
                    self.robot.move_left_arm(approach)
            else:
                self.robot.move_right_arm(approach)
                self.robot.controller.open_right_gripper(True)
                success = self.robot.move_right_arm(ps)
                if success:
                    self.robot.controller.close_right_gripper(True)    
                if pullup:
                    self.robot.move_right_arm(approach)
            
            if success:
                break                
        
        
        
            
        
         
        
        
        

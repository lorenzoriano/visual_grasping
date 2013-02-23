import copy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose
from sensor_msgs.msg import PointCloud2, PointField
from tf import transformations

def pointcloud2_to_array(cloud_msg):
    ''' 
    Converts a rospy PointCloud2 message to a numpy recordarray 
    
    Assumes all fields 32 bit floats, and there is no padding.
    '''
    dtype_list = [(f.name, np.float32) for f in cloud_msg.fields]
    cloud_arr = np.fromstring(cloud_msg.data, dtype_list)
    return np.reshape(cloud_arr, (cloud_msg.height, cloud_msg.width))

def get_xyz_points(cloud_array, remove_nans=True):
    '''
    Pulls out x, y, and z columns from the cloud recordarray, and returns a 3xN matrix.
    '''
    # remove crap points
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(list(cloud_array.shape) + [3], dtype=np.float)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']

    return points

def pointcloud2_to_xyz_array(cloud_msg, remove_nans=True):
    return get_xyz_points(pointcloud2_to_array(cloud_msg), remove_nans=remove_nans)

def xyz_array_to_pointcloud2(points, stamp=None, frame_id=None):
    '''
    Create a sensor_msgs.PointCloud2 from an array
    of points.
    '''
    msg = PointCloud2()
    if stamp:
        msg.header.stamp = stamp
    if frame_id:
        msg.header.frame_id = frame_id
    if len(points.shape) == 3:
        msg.height = points.shape[1]
        msg.width = points.shape[0]
    else:
        msg.height = 1
        msg.width = len(points)
    msg.fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1)]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = 12*points.shape[0]
    msg.is_dense = int(np.isfinite(points).all())
    msg.data = np.asarray(points, np.float32).tostring()
    return msg

def matrixToPose(matrix):
    """Converts a Matrix  to a Pose message by extracting the translation and
    rotation specified by the Matrix.

    Parameters:
    matrix: a 4x4 transformation matrix
    """
    pos = Pose()
    T = transformations.translation_from_matrix(matrix)
    pos.position.x = T[0]
    pos.position.y = T[1]
    pos.position.z = T[2]
    
    q = transformations.quaternion_from_matrix(matrix)
    pos.orientation.x = q[0]
    pos.orientation.y = q[1]
    pos.orientation.z = q[2]
    pos.orientation.w = q[3]

    return pos

def poseTomatrix(pose):
    """Converts a Pose message to a matrix 4.
    """
    assert isinstance(pose, Pose)
    
    T = ((pose.position.x,
                                              pose.position.y,
                                              pose.position.z, )
                                             )
    R = transformations.euler_from_quaternion((
                            pose.orientation.x,
                            pose.orientation.y,
                            pose.orientation.z,
                            pose.orientation.w)
                 )
    
    return transformations.compose_matrix(translate=T,
                                          angles=R)

def makeGripperMarker(pose, angle=0.0, color=None, scale=1.0):
    """
    Creates an MarkerArray with the PR2 gripper shape. 
    Parameters:
    pose: the position of the marker, a Pose instance
    angle: the aperture angle of the gripper (default=0.541)
    color: (r,g,b,a) tuple or None (default) if using the material colors
    scale: the scale of the gripper, default is 1.0

    Returns:
    The new gripper MarkerArray
    """
    assert isinstance(pose, PoseStamped)
    pose_T = poseTomatrix(pose.pose)

    gripper_marker = MarkerArray()

    
    T1 = transformations.translation_matrix((0.07691, 0.01, 0.))
    T1 = T1.dot(transformations.rotation_matrix(angle, (0, 0, 1)))
    T2 = transformations.translation_matrix((0.09137, 0.00495, 0.))
    T2 = T2.dot(transformations.rotation_matrix(angle, (0, 0, 1)))
    
    #T1 = euclid.Matrix4()
    #T2 = euclid.Matrix4()
    #T1.translate(0.07691, 0.01, 0.)
    #T1.rotate_axis(angle, euclid.Vector3(0,0,1))
    #T2.translate(0.09137, 0.00495, 0.)
    #T1.rotate_axis(-angle, euclid.Vector3(0,0,1))

    T_proximal = T1.copy()
    T_distal = T1.dot(T2)

    mesh = Marker()
    mesh.header.frame_id = pose.header.frame_id
    mesh.type = Marker.MESH_RESOURCE
    mesh.scale.x = scale
    mesh.scale.y = scale
    mesh.scale.z = scale
    
    if color is not None:
        mesh.color.r = color[0]
        mesh.color.g = color[1]
        mesh.color.b = color[2]
        mesh.color.a = color[3]
        mesh.mesh_use_embedded_materials = False
    else:
        mesh.mesh_use_embedded_materials = True

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/gripper_palm.dae"
    mesh.pose = matrixToPose(pose_T)
    gripper_marker.markers.append(copy.deepcopy(mesh))
    
    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger.dae"
    mesh.pose = matrixToPose(pose_T.dot(T_proximal))
    gripper_marker.markers.append(copy.deepcopy(mesh))

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger_tip.dae"
    mesh.pose = matrixToPose(pose_T.dot(T_distal))
    gripper_marker.markers.append(copy.deepcopy(mesh))

    #T1 = euclid.Matrix4()
    #T2 = euclid.Matrix4()
    #T1.translate(0.07691, -0.01, 0.)
    #T1.rotate_axis(np.pi, euclid.Vector3(1,0,0))
    #T1.rotate_axis(angle, euclid.Vector3(0,0,1))
    #T2.translate(0.09137, 0.00495, 0.)
    #T1.rotate_axis(-angle, euclid.Vector3(0,0,1))
    
    T1 = transformations.translation_matrix((0.07691, -0.01, 0.))
    T1 = T1.dot(transformations.rotation_matrix(np.pi, (1, 0, 0)))
    T1 = T1.dot(transformations.rotation_matrix(angle, (0, 0, 1)))
    T2 = transformations.translation_matrix((0.09137, 0.00495, 0.))
    T2 = T2.dot(transformations.rotation_matrix(-angle, (0, 0, 1)))    

    T_proximal = T1.copy()
    T_distal = T1.dot(T2)

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger.dae"
    mesh.pose = matrixToPose(pose_T.dot(T_proximal))
    gripper_marker.markers.append(copy.deepcopy(mesh))

    mesh.mesh_resource = "package://pr2_description/meshes/gripper_v0/l_finger_tip.dae"
    mesh.pose = matrixToPose(pose_T .dot(T_distal))
    gripper_marker.markers.append(copy.deepcopy(mesh))

    for i, marker in enumerate(gripper_marker.markers):
        marker.id = i

    return gripper_marker

def make_orth_basis_z_ax(z_ax):
    """
    orthogonal basis from a given z axis
    John Schulman magic code.
    """
    z_ax = np.asarray(z_ax)
    z_ax = z_ax / np.linalg.norm(z_ax)
    
    if np.allclose(z_ax, [0, 0, 1]) or np.allclose(z_ax, [0, 0, -1]):
        raise Exception("singular values!")
    
    x_ax = np.array([0, 0, -1.])
    x_ax -= z_ax * x_ax.dot(z_ax)
    y_ax = np.cross(z_ax, x_ax)
    
    T = np.c_[x_ax, y_ax, z_ax]
    #building a 4x4 matrix
    M = np.eye(4)
    M[:3, :3] = T
    return M


def make_orth_basis(x_ax):
    """
    John Schulman magic code.
    """
    x_ax = np.asarray(x_ax)

    x_ax = x_ax / np.linalg.norm(x_ax)
    if np.allclose(x_ax, [1,0,0]):
        return np.eye(3)
    elif np.allclose(x_ax, [-1, 0, 0]):
        return np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, 1]])
    else:
        y_ax = np.r_[0, x_ax[2], -x_ax[1]]
        y_ax /= np.linalg.norm(y_ax)
        z_ax = np.cross(x_ax, y_ax)
        return np.c_[x_ax, y_ax, z_ax]
    
    
def valid_sphere_given_one_coord(d, 
                                 x1, min_x2, max_x2,
                                 center = (0, 0, 0), 
                                 n_points=1,                                 
                                 n_attempts = 100,
                                 constrain_x2 = lambda _ :True,
                                 constrain_x3 = lambda _: True):
    
    center_x1, center_x2, center_x3 = center
    dx1 = x1 - center_x1
    
    if d ** 2 - dx1 ** 2 < 0:
        return []
    
    attempt = 0
    all_points = []
    while len(all_points) < n_points and attempt < n_attempts:
        attempt += 1
        x2 = np.random.uniform(min_x2, max_x2)
        dx2 = x2 - center_x2  
        det =  d ** 2 - dx1**2 - dx2 ** 2
        if det < 0:  #invalid coordinates
            continue
        if not constrain_x2(x2):
            continue
        dx3 = np.sqrt(det)
        x3 = dx3 + center_x3
        if not constrain_x3(x3):
            continue
        all_points.append([x1, x2, x3])
    
    return all_points
            
            
        
        
    
    
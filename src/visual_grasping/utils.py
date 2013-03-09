import copy
import numpy as np
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped, Pose, Point32
from sensor_msgs.msg import PointCloud2, PointField, PointCloud
from tf import transformations
from object_manipulation_msgs.msg import GraspableObject

def pc2xyzrgb(pc):
    arr = np.fromstring(pc.data,dtype='float32').reshape(pc.height,
                                                         pc.width,pc.point_step/4)
    xyz = arr[:,:,0:3]
    
    rgb0 = np.ndarray(buffer=arr[:,:,4].copy(),shape=(pc.height,
                                                      pc.width,4),dtype='uint8')
    rgb = rgb0[:,:,0:3]
    
    return xyz,rgb

def pc2xyz(pc):
    arr = np.fromstring(pc.data,dtype='float32').reshape(pc.height,
                                                         pc.width,pc.point_step/4)
    xyz = arr[:,:,0:3]
    return xyz

def xyz2pc(xyz,frame_id):
    bgr = np.zeros_like(xyz)
    return xyzrgb2pc(xyz,bgr,frame_id)

def xyzrgb2pc(xyz,bgr,frame_id):
    xyz = np.asarray(xyz)
    bgr = np.asarray(bgr)
    
    assert xyz.shape == bgr.shape
    if xyz.ndim ==2:
        xyz = xyz[None,:,:]
        bgr = bgr[None,:,:]
    
    height= xyz.shape[0]
    width = xyz.shape[1]
    
    arr = np.empty((height,width,8),dtype='float32')
    arr[:,:,0:3] = xyz
    bgr1 = np.empty((height,width,4),dtype='uint8')
    bgr1[:,:,0:3] = bgr
    arr[:,:,4] = bgr1.view(dtype='float32').reshape(height, width)
    data = arr.tostring()
    msg = sm.PointCloud2()
    msg.data = data
    msg.header.frame_id = frame_id
    msg.fields = [sm.PointField(name='x',offset=0,datatype=7,count=1),
                  sm.PointField(name='y',offset=4,datatype=7,count=1),
                  sm.PointField(name='z',offset=8,datatype=7,count=1),
                  sm.PointField(name='rgb',offset=16,datatype=7,count=1)]
    msg.is_dense = False
    msg.width=width
    msg.height=height
    msg.header.stamp = rospy.Time.now()
    msg.point_step = 32
    msg.row_step = 32 * width
    msg.is_bigendian = False
    
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
            
def pc2graspable(pc, name="", frame_id = "/base_link"):
    msg = GraspableObject()
    msg.reference_frame_id = frame_id
    if type(pc) is PointCloud2:
        pc = PointCloud2_to_PointCloud(pc)
    msg.cluster = pc
    msg.collision_name = name
    return msg
    
def PointCloud_to_PointCloud2(pc):
    isinstance(pc, PointCloud)
    xyz = np.array([[pt.x, pt.y, pt.z] for pt in pc.points])[None,:,:]
    rgb = np.zeros(xyz.shape)
    pc2 = xyzrgb2pc(xyz, rgb, pc.header.frame_id)
    return pc2

def PointCloud2_to_PointCloud(pc):
    assert isinstance(pc, PointCloud2)
    xyz = pc2xyz(pc)
    
    msg = PointCloud()
    msg.header = pc.header
    msg.points = [Point32(x,y,z) for (x,y,z) in xyz[0,:,:]]
    return msg
    
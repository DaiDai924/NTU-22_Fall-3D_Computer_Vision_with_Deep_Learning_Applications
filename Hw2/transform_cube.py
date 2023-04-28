import open3d as o3d
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import pandas as pd

vis = o3d.visualization.VisualizerWithKeyCallback()
cube = o3d.geometry.TriangleMesh.create_box(width=1.0, height=1.0, depth=1.0)
cube_vertices = np.asarray(cube.vertices).copy()
R_euler = np.array([0, 0, 0]).astype(float)
t = np.array([0, 0, 0]).astype(float)
scale = 1.0
shift_pressed = False

cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])    
distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def load_axes():
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    axes.lines  = o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]])          # X, Y, Z
    axes.colors = o3d.utility.Vector3dVector([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # R, G, B

    return axes

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)

    return transform_mat

def update_cube():
    global cube, cube_vertices, R_euler, t, scale
    
    transform_mat = get_transform_mat(R_euler, t, scale)
    
    transform_vertices = (transform_mat @ np.concatenate([
                            cube_vertices.transpose(), 
                            np.ones([1, cube_vertices.shape[0]])
                            ], axis=0)).transpose()

    cube.vertices = o3d.utility.Vector3dVector(transform_vertices)
    cube.compute_vertex_normals()
    cube.paint_uniform_color([1, 0.706, 0])
    vis.update_geometry(cube)

def toggle_key_shift(vis, action, mods):
    global shift_pressed
    if action == 1: # key down
        shift_pressed = True
    elif action == 0: # key up
        shift_pressed = False
    return True

def update_tx(vis):
    global t, shift_pressed
    t[0] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_ty(vis):
    global t, shift_pressed
    t[1] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_tz(vis):
    global t, shift_pressed
    t[2] += -0.01 if shift_pressed else 0.01
    update_cube()

def update_rx(vis):
    global R_euler, shift_pressed
    R_euler[0] += -1 if shift_pressed else 1
    update_cube()

def update_ry(vis):
    global R_euler, shift_pressed
    R_euler[1] += -1 if shift_pressed else 1
    update_cube()

def update_rz(vis):
    global R_euler, shift_pressed
    R_euler[2] += -1 if shift_pressed else 1
    update_cube()

def update_scale(vis):
    global scale, shift_pressed
    scale += -0.05 if shift_pressed else 0.05
    update_cube()

def adjust_cube_position():
    global vis, cube, cube_vertices, R_euler, t, scale, shift_pressed

    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # load axes
    axes = load_axes()
    vis.add_geometry(axes)

    # load cube
    vis.add_geometry(cube)

    update_cube()

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    # set key callback
    vis.register_key_action_callback(340, toggle_key_shift)
    vis.register_key_action_callback(344, toggle_key_shift)
    vis.register_key_callback(ord('A'), update_tx)
    vis.register_key_callback(ord('S'), update_ty)
    vis.register_key_callback(ord('D'), update_tz)
    vis.register_key_callback(ord('Z'), update_rx)
    vis.register_key_callback(ord('X'), update_ry)
    vis.register_key_callback(ord('C'), update_rz)
    vis.register_key_callback(ord('V'), update_scale)

    print('[Keyboard usage]')
    print('Translate along X-axis\tA / Shift+A')
    print('Translate along Y-axis\tS / Shift+S')
    print('Translate along Z-axis\tD / Shift+D')
    print('Rotate    along X-axis\tZ / Shift+Z')
    print('Rotate    along Y-axis\tX / Shift+X')
    print('Rotate    along Z-axis\tC / Shift+C')
    print('Scale                 \tV / Shift+V')

    vis.run()
    vis.destroy_window()

    print('Rotation matrix:\n{}'.format(R.from_euler('xyz', R_euler, degrees=True).as_matrix()))
    print('Translation vector:\n{}'.format(t))
    print('Scale factor: {}'.format(scale))

    np.save('cube_transform_mat.npy', get_transform_mat(R_euler, t, scale))
    np.save('cube_vertices.npy', np.asarray(cube.vertices))

# Build the cube of points in unit length
def build_cube_points(rot_trans_mat: np.ndarray):
    x_values, y_values = np.linspace(0, 1, 10), np.linspace(0, 1, 10)
    xx, yy = np.meshgrid(x_values, y_values)

    points3D = np.empty((0, 3))
    for x, y in zip(np.nditer(xx), np.nditer(yy)):
        # x = 0, x = 1, y = 0, y = 1, z = 0, z = 1
        points3D = np.vstack((points3D, [0, x, y]))
        points3D = np.vstack((points3D, [1, x, y]))
        points3D = np.vstack((points3D, [x, 0, y]))
        points3D = np.vstack((points3D, [x, 1, y]))
        points3D = np.vstack((points3D, [x, y, 0]))
        points3D = np.vstack((points3D, [x, y, 1]))

    _, indices = np.unique(points3D, return_index=True, axis=0)
    points3D = points3D[indices]

    # Assign different color to each surface
    colors = np.empty((len(points3D), 3), dtype=int)
    colors[points3D[:, 0] == 0] = [255, 0, 0]
    colors[points3D[:, 0] == 1] = [128, 128, 0]
    colors[points3D[:, 1] == 0] = [0, 255, 0]
    colors[points3D[:, 1] == 1] = [0, 128, 128]
    colors[points3D[:, 2] == 0] = [0, 0, 255]
    colors[points3D[:, 2] == 1] = [128, 0, 128]

    # Move to the adjusted position
    points3D = np.hstack((points3D, np.ones((points3D.shape[0], 1))))    # size: n x 4
    points3D = np.matmul(rot_trans_mat, points3D.T).T                    # size: n x 3

    return points3D, colors

# Save video from images
def img2video(imgs):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter('video.mp4', fourcc, 5, (imgs[0].shape[1], imgs[0].shape[0]))

    for img in imgs:
        video.write(img)

    cv2.destroyAllWindows()
    video.release()

def main():
    global cameraMatrix, distCoeffs

    adjust_cube_position()

    rot_trans_mat = np.load('cube_transform_mat.npy')
    vertices3D = np.load('cube_vertices.npy')

    points3D, colors = build_cube_points(rot_trans_mat)

    # Directly load estimated rotation and translation matrices
    estimation_df = pd.read_pickle("estimation.pkl")
    file_names = estimation_df["NAME"].values.tolist()

    imgs = []
    for idx_img, fileName in enumerate(file_names):
        estimation = estimation_df.loc[estimation_df["NAME"]==fileName]
        rotq = estimation[["QX","QY","QZ","QW"]].values
        tvec = estimation[["TX","TY","TZ"]].values
        tvec = tvec.reshape(3, 1)
        rotm = R.from_quat(rotq).as_matrix().reshape(3, 3)
        
        # Extrinsic matrix: camera pose R, t (projection from WCS to CCS)
        pose_mat = np.hstack((rotm, tvec))

        homo_points3D = np.hstack((points3D, np.ones((points3D.shape[0], 1))))           # size: n x 4
        homo_points2D = np.matmul(np.matmul(cameraMatrix, pose_mat), homo_points3D.T).T  # size: n x 3
        
        # Painter's algorithm: sort points by depth from the furthest to the nearest (the third value of homogeneous 2D points)
        sort_idx = np.argsort(homo_points2D[:, -1])[::-1]
        sort_homo_points2D = homo_points2D[sort_idx]       # size: n x 3
        sort_colors = colors[sort_idx]                     # size: n x 3

        sort_homo_points2D = sort_homo_points2D / sort_homo_points2D[:, -1].reshape(-1, 1)
        points2D = sort_homo_points2D[:, :-1]

        # Only keep the points inside the image frame, and draw the points on the frame
        img = cv2.imread("data/frames/" + fileName)
        img_h, img_w, _ = img.shape
        for pnt, color in zip(np.round(points2D), sort_colors):
            if pnt[0] >= 0 and pnt[0] < img_w and pnt[1] >= 0 and pnt[1] < img_h:
                pnt = (int(pnt[0]), int(pnt[1]))
                color = (int(color[0]), int(color[1]), int(color[2]))
                img = cv2.circle(img, pnt, 5, color, -1)
        
        # cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
        # cv2.imshow('img', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        imgs.append(img)

    # Conver images to a video
    img2video(imgs)

if __name__ == '__main__':
    main()
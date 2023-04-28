from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import open3d as o3d

def average(x):
    return list(np.mean(x,axis=0))

# Average of the descriptors to describe the same point
def average_desc(train_df, points3D_df):
    """
        Input:
            train_df: the descriptors of keypoints in all images
            points3D_df: the features of 3D points
        Output:
            averaged descriptors (idx: POINT_ID, DESCRIPTORS, XYZ, RGB)
    """
    train_df = train_df[["POINT_ID", "XYZ", "RGB", "DESCRIPTORS"]]    # the same point has several descriptors from different images
    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)    # take average of the descriptors of the same point
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")

    return desc

# Calculate the rotaion and translation matrices, whose projection is from world to the camera
def ICP(CCS_points: np.ndarray, WCS_points: np.ndarray):
    """
        Input:
            CCS_points: 3D points in the camera coordinate system
            WCS_points: 3D points in the world coordinate system

        Output:
            rotm: rotation matrix, size: (3 x 3)
            T: translation matrix, size: (3 x 1)
    """

    CCS_mean = np.mean(CCS_points, axis=0)
    WCS_mean = np.mean(WCS_points, axis=0)
    
    CCS_points_tran = CCS_points - CCS_mean
    WCS_points_tran = WCS_points - WCS_mean

    W = np.matmul(WCS_points_tran.T, CCS_points_tran)
    U, S, V_T = np.linalg.svd(W)

    rotm = np.matmul(V_T.T, U.T)
    t = CCS_mean - np.matmul(rotm, WCS_mean)

    return rotm, t

# Randomly choose 3 points and count inliers for num_iter times
def ransac(points3D, points2D, cameraMatrix, distCoeffs, smallest_points: int, num_iter: int, threshold: float):
    """
        Input:
            points3D: 3D keypoints
            points2D: 2D keypoints
            cameraMatrix: intrinsic matrix
            smallest_points: find at least # of points
            num_iter: at least iterate num_iter times
            threshold: vote for good model if a point is within this threshold
        Output:
            best_rotq: the best rotation matrix (quaterion)
            best_t: the best translation matrix
    """
    max_inlier = 0
    best_rot, best_t = np.array([]), np.array([])

    # At least iterate num_iter times
    for _ in range(num_iter):
        rand_idx = np.random.choice(len(points2D), smallest_points, replace=False)
        sample_points3D, sample_points2D = points3D[rand_idx], points2D[rand_idx]

        # Find the rotation and translation matrices by the three sampled points through solveP3P
        rotms, tvecs = P3P(sample_points3D, sample_points2D, cameraMatrix, distCoeffs)

        # Iterate each rotation and translation matrix
        for rotm, tvec in zip(rotms, tvecs):
            tvec = tvec.reshape(3, 1)
            # Count the inliers of this model for all points
            projection_mat = np.matmul(cameraMatrix, np.hstack((rotm, tvec)))  # size: 3 x 4 from 3x3 * 3x4
            ones = np.ones(len(points3D))
            homo_points3D = np.vstack((points3D.T, ones))                      # size: 4 x n_points
            est_homo_points2D = np.matmul(projection_mat, homo_points3D)       # size: 3 x n_points

            est_points2D = est_homo_points2D / est_homo_points2D[-1, :]        # size: 3 x n_points
            est_points2D = est_points2D[:-1, :].T                              # size: n_points x 2

            errors = np.linalg.norm((points2D - est_points2D), axis=1)
            num_inlier = np.count_nonzero(errors < threshold)
            
            # Save the best model of rotation and translation matrices
            if num_inlier > max_inlier:
                max_inlier = num_inlier
                best_rot = rotm
                best_t = tvec.reshape(3)

    best_rotq = R.from_matrix(best_rot).as_quat()

    return best_rotq, best_t

# Find the three points in the camera coordinate system, and calculate the rotation and translation matrices by ICP
def P3P(points3D: np.ndarray, points2D: np.ndarray, cameraMatrix: np.ndarray, distCoeffs: np.array):
    """
        Input:
            points3D: 3D keypoints
            points2D: 2D keypoints
            cameraMatrix: intrinsic matrix
            distCoeffs: distortion coefficients
        Output:
            retval:
            rvec: rotation vector
            tvec: translation vector
            inliers: 
    """

    # Undistort the points by camera matrix and distort coefficients
    points2D = cv2.undistortPoints(points2D, cameraMatrix, distCoeffs, None, cameraMatrix).reshape(-1, 2)

    # xi: non-homogeneous 3D points (x y z)
    x1, x2, x3 = points3D[0], points3D[1], points3D[2]
    
    # vi: non-homogeneous 2D points (x y), ui: homogeneous 2D points (x y 1)
    # vi = inv(K) * ui
    K_inv = np.linalg.inv(cameraMatrix)
    ones = np.ones((3, 1))
    U = np.append(points2D, ones, axis=1)
    v1, v2, v3 = np.matmul(K_inv, U[0]), np.matmul(K_inv, U[1]), np.matmul(K_inv, U[2])

    # Cij: angle cosine of (i, j)
    # Rij: distance from i to j
    Rab, Rac, Rbc = np.linalg.norm(x1 - x2), np.linalg.norm(x1 - x3), np.linalg.norm(x2 - x3)
    Cab = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    Cac = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    Cbc = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))

    K1, K2 = (Rbc / Rac) ** 2, (Rbc / Rab) ** 2

    # Quartic polynomial: 0 = G4 * x**4 + G3 * x**3 + G2 * x**2 + G1 * x + G0
    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * Cbc**2
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * Cab \
        + 4 * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
    G2 = (2 * K2 * (1 - K1) * Cab) ** 2 \
        + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) \
        + 4 * K1 * ((K1 - K2) * Cbc**2 + K1 * (1 - K2) * Cac**2 - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * Cab \
        + 4 * K1 * ((K1 * K2 - K1 + K2) * Cac * Cbc + 2 * K1 * K2 * Cab * Cac**2)
    G0 = (K1 * K2 + K1 - K2) ** 2 - 4 * K1**2 * K2 * Cac**2

    # Solving roots by companion matrix
    if G4 != 0:
        first_row = np.zeros((1, 3))
        identity_mat = np.identity(3)
        last_col = np.array([-G0 / G4, -G1 / G4, -G2 / G4, -G3 / G4]).reshape(-1, 1)
    elif G3 != 0:
        first_row = np.zeros((1, 2))
        identity_mat = np.identity(2)
        last_col = np.array([-G0 / G3, -G1 / G3, -G2 / G3]).reshape(-1, 1)
    else:
        return [], []

    companion_mat = np.vstack((first_row, identity_mat))
    companion_mat = np.hstack((companion_mat, last_col))
    roots, _ = np.linalg.eig(companion_mat)
    
    # By the real part of root x, compute the corresponding a, y, b, c
    x_list, rotms, tvecs = roots[np.isreal(roots)].real, [], []
    for x in x_list:
        a = np.sqrt( (Rab**2) / (1 + x**2 - 2 * x * Cab))
        
        m = 1 - K1
        p = 2 * (K1 * Cac - x * Cbc)
        q = x**2 - K1
        p_ = -2 * x * Cbc
        q_ = x**2 * (1 - K2) + 2 * x * K2 * Cab - K2

        y = -(q - m * q_) / (p - p_ * m)

        b = x * a
        c = y * a

        # a, b, c are distances, which is impossible to be negative numbers
        if a < 0 or b < 0 or c < 0:
            continue
        
        # 3 points in the camera coordinate system
        CCS_pnt1 = a * v1 / np.linalg.norm(v1)
        CCS_pnt2 = b * v2 / np.linalg.norm(v2)
        CCS_pnt3 = c * v3 / np.linalg.norm(v3)

        # Calculate and save the possible rotation and translation matrices
        rotm, tvec = ICP(np.array([CCS_pnt1, CCS_pnt2, CCS_pnt3]), points3D)

        rotms.append(rotm)
        tvecs.append(tvec)

    return rotms, tvecs

# Find the matches between 2D and 3D, and return the best rotation and translation matrices
def pnpsolver(query, model, cameraMatrix=0, distCoeffs=0):
    kp_query, desc_query = query
    kp_model, desc_model = model

    # Match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query, desc_model, k=2)

    # Ratio test only keeps the good matches of desc_query and desc_model(the average desc)
    gmatches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D, kp_query[query_idx]))
        points3D = np.vstack((points3D, kp_model[model_idx]))

    # Intrinsic Parameters and Distortion Parameters are known
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])    
    distCoeffs = np.array([0.0847023, -0.192929, -0.000201144, -0.000725352])

    # Get rotation and translation matrices by RANSAC
    rotq, tvec = ransac(points3D, points2D, cameraMatrix, distCoeffs, 3, 35, 1.5)

    return rotq, tvec

# Calculate rotation and translation error
def cal_pose_err(est, gt):
    """
        Input: 
            est: estimated rotation vector (in quaternion), estimated translation vector
            gt: ground truth rotation vector (in quaternion), ground truth translation vector
        Output:
            rot_rel_err: relative rotation error (in axis angle representation), translation error
    """

    rotq, tvec = est
    rotq_gt, tvec_gt = gt

    # Rotation relative error (represented in axis angle)
    rotm = R.from_quat(rotq).as_matrix()
    rotm_inv = np.linalg.inv(rotm)
    rotm_gt = R.from_quat(rotq_gt).as_matrix().reshape(3, 3)
    rotm_rel = np.matmul(rotm_gt, rotm_inv)
    rot_angle_err = np.arccos((np.trace(rotm_rel) - 1) / 2)  # axis angle representation

    # Translation Error (L2 norm)
    t_err = np.linalg.norm(tvec_gt - tvec)

    return rot_angle_err, t_err

def load_point_cloud(points3D_df):

    xyz = np.vstack(points3D_df['XYZ'])
    rgb = np.vstack(points3D_df['RGB'])/255

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    
    return pcd

def get_transform_mat(rotation, translation, scale):
    r_mat = R.from_euler('xyz', rotation, degrees=True).as_matrix()
    scale_mat = np.eye(3) * scale
    transform_mat = np.concatenate([scale_mat @ r_mat, translation.reshape(3, 1)], axis=1)

    return transform_mat

# Plot camera position, image plane and trajectory in WCS
def plot_trajectory(pnts, lines):
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(pnts)
    line_set.lines = o3d.utility.Vector2iVector(lines)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # load point cloud
    points3D_df = pd.read_pickle("data/points3D.pkl")
    pcd = load_point_cloud(points3D_df)
    vis.add_geometry(pcd)

    # add camera lines
    vis.add_geometry(line_set)

    o3d.visualization.ViewControl.set_zoom(vis.get_view_control(), 0.8)

    # just set a proper initial camera view
    vc = vis.get_view_control()
    vc_cam = vc.convert_to_pinhole_camera_parameters()
    initial_cam = get_transform_mat(np.array([7.227, -16.950, -14.868]), np.array([-0.351, 1.036, 5.132]), 1)
    initial_cam = np.concatenate([initial_cam, np.zeros([1, 4])], 0)
    initial_cam[-1, -1] = 1.
    setattr(vc_cam, 'extrinsic', initial_cam)
    vc.convert_from_pinhole_camera_parameters(vc_cam)

    vis.run()
    vis.destroy_window()

def main():
    # Read data
    images_df = pd.read_pickle("data/images.pkl")
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")
    point_desc_df = pd.read_pickle("data/point_desc.pkl")

    # Process model descriptors (an average descriptor for each point)
    desc_df = average_desc(train_df, points3D_df)
    kp_model = np.array(desc_df["XYZ"].to_list())
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    # Sort file names based on acsending number of validation images 
    file_names = images_df["NAME"].values.tolist()
    file_names = list(filter(lambda img_name: "valid_img" in img_name, file_names))
    file_names.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    rotq_list, tvec_list = np.empty((0, 4)), np.empty((0, 3))
    imgName_list, imgID_list = file_names, np.empty((0), dtype=int)
    t_errs, rot_errs = np.empty((0)), np.empty((0))
    pnts, lines = [], []
    for idx, fileName in enumerate(file_names):
        # print(fileName)

        # Find the index of the image
        idx_img = int(((images_df.loc[images_df["NAME"] == fileName])["IMAGE_ID"].values)[0])
        imgID_list = np.hstack((imgID_list, idx_img))

        # Load query keypoints and descriptors
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"]==idx_img]
        kp_query = np.array(points["XY"].to_list())
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        # Find correspondence and solve pnp
        rotq, tvec = pnpsolver((kp_query, desc_query),(kp_model, desc_model))


        rotq_list = np.vstack((rotq_list, rotq))
        tvec_list = np.vstack((tvec_list, tvec))

        # Get camera pose groudtruth 
        ground_truth = images_df.loc[images_df["IMAGE_ID"]==idx_img]
        rotq_gt = ground_truth[["QX","QY","QZ","QW"]].values
        tvec_gt = ground_truth[["TX","TY","TZ"]].values

        # Calculate pose error
        rot_err, t_err = cal_pose_err((rotq, tvec), (rotq_gt, tvec_gt))
        # print("Rotation relative angle:", rot_err)
        # print("Translation error:", t_err)

        # Rotation error is the sum of absolute value of euler angle representation
        rot_errs, t_errs = np.append(rot_errs, rot_err), np.append(t_errs, t_err)


        # Save information of camera position and image plane in WCS for plotting trajectory
        rotm = R.from_quat(rotq).as_matrix().reshape(3, 3)
        rotm_inv = np.linalg.inv(rotm)

        # apex: camera position in WCS, imgPlane_corners: 4 corners of image plane
        apex = np.matmul(rotm_inv, -tvec)
        imgPlane_corner1 = np.matmul(rotm_inv, np.array([0.05, 0.05, 0.1]) - tvec)
        imgPlane_corner2 = np.matmul(rotm_inv, np.array([0.05, -0.05, 0.1]) - tvec)
        imgPlane_corner3 = np.matmul(rotm_inv, np.array([-0.05, 0.05, 0.1]) - tvec)
        imgPlane_corner4 = np.matmul(rotm_inv, np.array([-0.05, -0.05, 0.1]) - tvec)

        # Point set
        pnts.append(apex)
        pnts.append(imgPlane_corner1)
        pnts.append(imgPlane_corner2)
        pnts.append(imgPlane_corner3)
        pnts.append(imgPlane_corner4)

        # Line set
        idx_line = idx * 5
        # Lines between apex and the 4 image plane corners
        lines.append([idx_line, idx_line + 1])
        lines.append([idx_line, idx_line + 2])
        lines.append([idx_line, idx_line + 3])
        lines.append([idx_line, idx_line + 4])
        # Lines between the image plane corners
        lines.append([idx_line + 1, idx_line + 2])
        lines.append([idx_line + 1, idx_line + 3])
        lines.append([idx_line + 2, idx_line + 4])
        lines.append([idx_line + 3, idx_line + 4])
        # Camera trajectory
        if idx != 0:
            lines.append([idx_line - 5, idx_line])

    # Write pkl file of estimated rotation and translation matrices
    estimation = {"IMAGE_ID": imgID_list, "NAME": imgName_list, "QW": rotq_list[:, 3], "QX": rotq_list[:, 0], "QY": rotq_list[:, 1], "QZ": rotq_list[:, 2], "TX": tvec_list[:, 0], "TY": tvec_list[:, 1], "TZ": tvec_list[:, 2]}
    estimation_df = pd.DataFrame(estimation)
    with open("estimation.pkl", 'wb+') as file:
        estimation_df.to_pickle(file) 

    # The median of pose error
    rot_errs, t_errs = np.sort(rot_errs), np.sort(t_errs)
    rot_err_median, t_err_median = np.median(rot_errs), np.median(t_errs)
    print("Median of rotation error   :", rot_err_median)
    print("Median of translation error:", t_err_median)

    # Plot camera trajectory
    plot_trajectory(pnts, lines)

if __name__ == '__main__':
    main()
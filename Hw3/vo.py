import open3d as o3d
import numpy as np
import cv2 as cv
import os, argparse, glob
import multiprocessing as mp

class Frame:
    def __init__(self, R, t, scale, keypoints, descriptors) -> None:
        """ 
            Parameters:
                R: rotation matrix in WCS
                t: translation matrix in WCS
                scale: scale compared to the first two frames
                kp: keypoints of this frame
                des: descriptors of this frame
        """

        self.R = R
        self.t = t
        self.scale = scale
        self.kp = keypoints
        self.des = descriptors

class SimpleVO:
    def __init__(self, args):
        # Camera intrinsic parameters
        camera_params = np.load(args.camera_parameters, allow_pickle=True)[()]
        self.K = camera_params['K']
        self.dist = camera_params['dist']

        self.frame_paths = sorted(list(glob.glob(os.path.join(args.frame_dir, '*.png'))))

    # Preprcess the first two frames as the basis
    def preprocess(self):
        # Initiate two relative frames and rotation / translation matrices in WCS
        self.pre_frame: Frame = None
        self.cur_frame: Frame = None
        self.R_WCS = None
        self.t_wcs = None

        # Initiate ORB detector
        self.orb = cv.ORB_create()
        # Create BFMatcher object
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Process the first and the second frames
        img_0 = cv.imread(self.frame_paths[0])
        kp_0, des_0 = self.orb.detectAndCompute(img_0, None)
        self.pre_frame = Frame(np.eye(3, dtype=np.float64), np.zeros((3, 1), dtype=np.float64), 1, kp_0, des_0)

        img_1 = cv.imread(self.frame_paths[1])
        kp_1, des_1 = self.orb.detectAndCompute(img_1, None)
        R, t = self.get_rel_pose(kp_0, des_0, kp_1, des_1)
        t = -t  # for the correct direction
        self.cur_frame = Frame(R, t, 1, kp_1, des_1)
        
        self.R_WCS, self.t_WCS = R, t
        
    # Calculate the relative pose between two frames
    def get_rel_pose(self, kp1: np.array, des1: np.array, kp2: np.array, des2: np.array):
        """
            Parameters:
                kp1, des1: the keypoints and descriptors of the formal frame
                kp2, des2: the keypoints and descriptors of the latter frame

            Return:
                rel_R: relative rotation matrix between the two frames
                rel_t: relative translation matrix between the two frames
        """

        # Match descriptors
        matches = self.bf.match(des1, des2)

        pnts1 = np.array([kp1[m.queryIdx].pt for m in matches])
        pnts2 = np.array([kp2[m.trainIdx].pt for m in matches])

        # Undistort points for Essential matrix calculation
        pnts1 = cv.undistortPoints(pnts1, self.K, self.dist, None, self.K)
        pnts2 = cv.undistortPoints(pnts2, self.K, self.dist, None, self.K)

        # Find Essential matrix and recover rotation and translation matrices from E
        E, _ = cv.findEssentialMat(pnts1, pnts2, self.K)
        _, rel_R, rel_t, _ = cv.recoverPose(E, pnts1, pnts2, self.K)
        
        return rel_R, rel_t

    # Calculate the relative scale between three frames
    def get_rel_scale(self, pre_frame: Frame, cur_frame: Frame, post_frame: Frame):
        """
            Parameters:
                pre_frame: the previous frame
                cur_frame: the current frame
                post_frame: the next frame

            Return:
                scale: the median of the relative scales
        """
        
        # Match between the three frames by the current frame
        matches_1, matches_2= self.bf.match(cur_frame.des, pre_frame.des), self.bf.match(cur_frame.des, post_frame.des)
        queryIdx_1, queryIdx_2 = [m.queryIdx for m in matches_1], [m.queryIdx for m in matches_2]

        # Find the matched index of the current frame
        cur_matchIdx, pre_ind, post_ind = np.intersect1d(queryIdx_1, queryIdx_2, return_indices=True)

        # Correspond the index to the matched points of all three frames
        cur_pnts = np.array([cur_frame.kp[idx].pt for idx in cur_matchIdx])
        pre_pnts = np.array([pre_frame.kp[matches_1[pre_i].trainIdx].pt for pre_i in pre_ind])
        post_pnts = np.array([post_frame.kp[matches_2[post_i].trainIdx].pt for post_i in post_ind])

        # Triangulation
        # Projection matrices: cur_proj_mat and post_proj_mat are the relative projection matrices of pre_proj_mat
        pre_proj_mat = self.K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        cur_proj_mat = self.K @ np.hstack((cur_frame.R, cur_frame.t))

        rel_post_R = post_frame.R @ cur_frame.R
        rel_post_t = cur_frame.t + cur_frame.R @ post_frame.t
        post_proj_mat = self.K @ np.hstack((rel_post_R, rel_post_t))

        # Iterate many times in order to get stable result
        num_pts = len(cur_pnts)
        scales = []
        for _ in range(num_pts // 2):

            # Randomly choose two indices for two points
            rand_idxs = np.random.choice(num_pts, size=2, replace=False)
            rand_pre_pnts = pre_pnts[rand_idxs].T
            rand_cur_pnts = cur_pnts[rand_idxs].T
            rand_post_pnts = post_pnts[rand_idxs].T

            # Reconstructed points in 3D homogeneous coordinates
            homo_pnts_1 = cv.triangulatePoints(pre_proj_mat, cur_proj_mat, rand_pre_pnts, rand_cur_pnts)
            homo_pnts_2 = cv.triangulatePoints(cur_proj_mat, post_proj_mat, rand_cur_pnts, rand_post_pnts)

            homo_pnts_1 /= homo_pnts_1[-1, :]
            homo_pnts_2 /= homo_pnts_2[-1, :]

            homo_pnts_1, homo_pnts_2 = homo_pnts_1.T, homo_pnts_2.T

            # Distances between each two reconstructed points
            dist1, dist2 = np.linalg.norm(homo_pnts_1[0] - homo_pnts_1[1]), np.linalg.norm(homo_pnts_2[0] - homo_pnts_2[1])
            
            # Calculate the scale, the ratio of two distances
            scale = dist2 / dist1
            scales.append(scale)

        return np.median(scales)

    # Calculate the pose of all the frames
    def process_frames(self, queue):

        # Preprocess the first two frames
        self.preprocess()

        # Process the other following frames
        for frame_path in self.frame_paths[2:]:
            post_img = cv.imread(frame_path)
            post_kp, post_des = self.orb.detectAndCompute(post_img, None)

            rel_R, rel_t = self.get_rel_pose(self.cur_frame.kp, self.cur_frame.des, post_kp, post_des)
            rel_t = -rel_t
            post_frame = Frame(rel_R, rel_t, 1, post_kp, post_des)

            scale = self.get_rel_scale(self.pre_frame, self.cur_frame, post_frame)
            if scale > 2.5:
                scale = 2.5
            post_frame.scale = scale

            # Rotation and translation in the world cooridinate system
            self.R_WCS = rel_R @ self.R_WCS
            self.t_WCS = self.t_WCS + scale * self.R_WCS @ rel_t

            queue.put((self.R_WCS, self.t_WCS))

            # Update new frame
            self.pre_frame = self.cur_frame
            self.cur_frame = post_frame
            
            # show image
            img = cv.drawKeypoints(post_img, post_kp, None, color=(0, 255, 0))
            cv.imshow('frame', img)

            if cv.waitKey(30) == 27:
                break

    # Get the line set of each camera pose for open3d plotting
    def get_lineset(self, R: np.array, t: np.array):
        """
            Parameters:
                R, t: rotation and translation matrices
            Return:
                line_set: line sets for plotting
        """

        # Point set: apex and four corners of the image plane
        pnts = np.array([[0, 0, 0], [1, 1, 3], [1, -1, 3], [-1, 1, 3], [-1, -1, 3]])
        pnts = R @ pnts.T + t
        pnts = pnts.T

        # Line set
        lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [1, 3], [2, 4], [3, 4]]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(pnts)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # blue
        colors = np.tile([0, 0, 1], (8, 1))
        line_set.colors = o3d.utility.Vector3dVector(colors)

        return line_set

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        queue = mp.Queue()
        p = mp.Process(target=self.process_frames, args=(queue, ))
        p.start()
        
        keep_running = True
        while keep_running:
            try:
                R, t = queue.get(block=False)
                if R is not None:
                    # insert new camera pose
                    vc = vis.get_view_control()
                    line_set = self.get_lineset(R, t)
                    vis.add_geometry(line_set)
            except:
                pass
            
            keep_running = keep_running and vis.poll_events()
        vis.destroy_window()
        p.join()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frame_dir', help='directory of sequential frames')
    parser.add_argument('--camera_parameters', default='camera_parameters.npy', help='npy file of camera parameters')
    args = parser.parse_args()

    vo = SimpleVO(args)
    vo.run()

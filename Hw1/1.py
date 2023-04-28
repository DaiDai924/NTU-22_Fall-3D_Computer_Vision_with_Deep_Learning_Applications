from operator import gt
import sys
import numpy as np
import cv2 as cv # opencv-python==4.5.1.48
import math

def get_sift_correspondences(img1, img2, k_toppairs):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image

    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''

    # sift = cv.xfeatures2d.SIFT_create() # opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()      
    # find keypoints and descriptors       
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matcher = cv.BFMatcher() # BFMatcher: Brute-Force Matcher
    matches = matcher.knnMatch(des1, des2, k=2) # Match descriptors

    # Ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)

    # Sample k pairs of correspondences
    good_matches = good_matches[:k_toppairs]

    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.namedWindow('match', cv.WINDOW_NORMAL)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)

    return points1, points2

# Direct Linear Transform
def DLT(k_points1, k_points2):
    """
    Input:
        points1: numpy array [k, 2], k is the number of top k pairs of correspondences
        points2: numpy array [k, 2], k is the number of top k pairs of correspondences

    Return:
        Homography matrix: 3 x 3 matrix
    """

    # Construct A matrix from point correspondences
    k_toppairs = len(k_points1)
    A = np.zeros([k_toppairs * 2, 9], dtype=float)
    for i in range(k_toppairs):
        A[i * 2, 3] = -k_points1[i, 0]
        A[i * 2, 4] = -k_points1[i, 1]
        A[i * 2, 5] = -1
        A[i * 2, 6] = k_points2[i, 1] * k_points1[i, 0]
        A[i * 2, 7] = k_points2[i, 1] * k_points1[i, 1]
        A[i * 2, 8] = k_points2[i, 1]

        A[i * 2 + 1, 0] = k_points1[i, 0]
        A[i * 2 + 1, 1] = k_points1[i, 1]
        A[i * 2 + 1, 2] = 1
        A[i * 2 + 1, 6] = -k_points2[i, 0] * k_points1[i, 0]
        A[i * 2 + 1, 7] = -k_points2[i, 0] * k_points1[i, 1]
        A[i * 2 + 1, 8] = -k_points2[i, 0]

    # SVD
    _, _, V = np.linalg.svd(A)
    h = V[-1]
    H = h.reshape(3, 3)
    H /= H[2, 2]

    return H

# Normalized Direct Linear Transform
def norm_DLT(k_points1, k_points2):
    """
    Input:
        points1: numpy array [k, 2], k is the number of top k pairs of correspondences
        points2: numpy array [k, 2], k is the number of top k pairs of correspondences
        k_toppairs: k_toppairs

    Return:
        Homography matrix: 3 x 3 matrix
    """

    k_toppairs = len(k_points1)

    # Calculate mean of u, v and u', v'
    mu_u1 = k_points1[:, 0].sum() / k_toppairs
    mu_v1 = k_points1[:, 1].sum() / k_toppairs
    mu_u2 = k_points2[:, 0].sum() / k_toppairs
    mu_v2 = k_points2[:, 1].sum() / k_toppairs

    # Scalar quantities
    s1 = 0
    for pnt in k_points1:
        s1 += np.sqrt(np.square(pnt[0] - mu_u1) + np.square(pnt[1] - mu_v1))
    s1 /= math.sqrt(2) * k_toppairs
    s1 =  np.reciprocal(s1)

    s2 = 0
    for pnt in k_points2:
        s2 += np.sqrt(np.square(pnt[0] - mu_u2) + np.square(pnt[1] - mu_v2))
    s2 /= math.sqrt(2) * k_toppairs
    s2 =  np.reciprocal(s2)

    # Calculate similarity transform T, T'
    T1 = np.zeros([3, 3], dtype=float)
    T1[0, 0] = s1
    T1[0, 2] = -s1 * mu_u1
    T1[1, 1] = s1
    T1[1, 2] = -s1 * mu_v1
    T1[2, 2] = 1

    T2 = np.zeros([3, 3], dtype=float)
    T2[0, 0] = s2
    T2[0, 2] = -s2 * mu_u2
    T2[1, 1] = s2
    T2[1, 2] = -s2 * mu_v2
    T2[2, 2] = 1

    # pi = [ui, vi, 1], norm_p = T dot p
    col_ones = np.ones(k_toppairs)
    p1 = np.column_stack((k_points1, col_ones))
    p2 = np.column_stack((k_points2, col_ones))

    norm_p1 = np.matmul(p1, T1.T)
    norm_p2 = np.matmul(p2, T2.T)

    # DLT of normalized points
    H = DLT(norm_p1, norm_p2)

    # Denormalize to get the normalized Homography matrix
    norm_H = np.matmul(np.linalg.inv(T2), np.matmul(H, T1))
    norm_H /= norm_H[2, 2]

    return norm_H

# Compute the reprojection error with the ground truth matching pairs
def compute_error(gt_correspondences, homography_mat):
    # p_s: 100 source points, p_t: 100 target points
    num_pts = len(gt_correspondences[0]) # 100
    p_s = gt_correspondences[0]
    p_t = gt_correspondences[1]

    col_ones = np.ones(num_pts)
    p_s = np.column_stack((p_s, col_ones))
    p_t = np.column_stack((p_t, col_ones))
    p_t_est = np.matmul(p_s, homography_mat.T)

    error = np.sum(np.linalg.norm((p_t[:, :2] - p_t_est[:, :2]), ord=2, axis=0)) / num_pts

    return error

if __name__ == '__main__':
    img1 = cv.imread(sys.argv[1])
    img2 = cv.imread(sys.argv[2])
    gt_correspondences = np.load(sys.argv[3])
    
    # Choose top k pairs for estimating Homography matrix
    k_toppairs = int(sys.argv[4])

    k_points1, k_points2 = get_sift_correspondences(img1, img2, k_toppairs)

    # Direct Linear Transform
    homography_mat = DLT(k_points1, k_points2)
    error = compute_error(gt_correspondences, homography_mat)
    print("DLT: ")
    print(homography_mat)
    print(error)

    # Normalized DLT
    norm_homography_mat = norm_DLT(k_points1, k_points2)
    norm_error = compute_error(gt_correspondences, norm_homography_mat)
    print("Normalized DLT: ")
    print(norm_homography_mat)
    print(norm_error)
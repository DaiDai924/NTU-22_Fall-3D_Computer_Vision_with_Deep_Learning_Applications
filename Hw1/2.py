import sys
import numpy as np
import cv2 as cv

def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param[0].append([x, y])

# Use mouse to get 4 corner points of the image
def get_corners(img):
    points_add= []
    cv.namedWindow('get_corners', cv.WINDOW_NORMAL)
    cv.setMouseCallback('get_corners', on_mouse, [points_add])

    # Break when it gets 4 points
    while len(points_add) < 4:
        img_ = img.copy()
        for i, p in enumerate(points_add):
            # draw points on img_
            cv.circle(img_, tuple(p), 2, (0, 255, 0), -1)
        cv.imshow('get_corners', img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()
    
    print('{} corner points added'.format(len(points_add)))
    print(points_add)

    return np.array(points_add)

# Direct Linear Transform
def DLT(k_points1, k_points2, k_toppairs = 4):
    """
    Input:
        points1: numpy array [k, 2], k is the number of top k pairs of correspondences
        points2: numpy array [k, 2], k is the number of top k pairs of correspondences

    Return:
        Homography matrix: 3 x 3 matrix
    """

    # Construct A matrix from point correspondences
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

# Inverse/Backward warping
def backward_warping(ori_img, homography_mat):
    """
    Input:
        ori_img: original image
        homography_mat: 3 x 3 matrix

    Return:
        warp_img: the warped image after doing backward warping
    """

    inv_homography_mat = np.linalg.inv(homography_mat)
    img_h, img_w, _ = ori_img.shape
    img_h, img_w = int(img_h/2), int(img_w/2)
    warp_img = np.empty((img_h, img_w), dtype=np.uint8)

    # Construct matrix dst_coordinates [wi hi 1], size = (img_w * img_h, 3)
    # ex: image size = 2 x 3
    #     | 0 0 1 | 
    #     | 1 0 1 | 
    #     | 2 0 1 | 
    #     | 0 1 1 |
    #     | 1 1 1 | 
    #     | 2 1 1 |

    w_arr = np.arange(img_w).reshape((img_w, 1))
    h_arr = np.zeros((img_w, 1))
    dst_coordinates = np.hstack((w_arr, h_arr))

    for i in range(1, img_h):
        h_arr = np.ones((img_w, 1)) * i
        dst_coordinates = np.vstack((dst_coordinates, np.hstack((w_arr, h_arr))))

    one_arr = np.ones((len(dst_coordinates), 1))
    dst_coordinates = np.hstack((dst_coordinates, one_arr))
    
    # Inverse warping to source coordinates
    src_coordinates = np.matmul(dst_coordinates, inv_homography_mat.T)
    src_coordinates = src_coordinates.T / src_coordinates[:, 2]
    src_w_arr, src_h_arr = src_coordinates[0], src_coordinates[1]

    # Calculate values of each pixel
    h_l_arr, h_r_arr = np.floor(src_h_arr).astype(int), np.ceil(src_h_arr).astype(int)
    w_l_arr, w_r_arr = np.floor(src_w_arr).astype(int), np.ceil(src_w_arr).astype(int)

    warp_img = (ori_img[h_l_arr, w_l_arr, :] * (src_h_arr - h_l_arr).reshape(-1, 1) * (src_w_arr - w_l_arr).reshape(-1, 1) 
                + ori_img[h_l_arr, w_r_arr, :] * (src_h_arr - h_l_arr).reshape(-1, 1) * (w_r_arr - src_w_arr).reshape(-1, 1) 
                + ori_img[h_r_arr, w_l_arr, :] * (h_r_arr - src_h_arr).reshape(-1, 1) * (src_w_arr - w_l_arr).reshape(-1, 1) 
                + ori_img[h_r_arr, w_r_arr, :] * (h_r_arr - src_h_arr).reshape(-1, 1) * (w_r_arr - src_w_arr).reshape(-1, 1)).reshape((img_h, img_w, 3)).astype(np.uint8)

    return warp_img

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python 2.py [IMAGE PATH]')
        sys.exit(1)

    # img size: 3024 x 4032 x 3
    img = cv.imread(sys.argv[1])
    img_h, img_w, _ = img.shape
    img_h, img_w = int(img_h/2), int(img_w/2)

    # left-top, right-top, left-bottom, right-bottom
    corners = get_corners(img)
    #corners = np.array([[696, 1270], [2951, 294], [1395, 2747], [3556, 1841]], dtype=float)

    proj_corners = np.array([[0, 0],
                            [img_w - 1, 0],
                            [0, img_h - 1],
                            [img_w - 1, img_h - 1]])

    H = DLT(corners, proj_corners)

    warp_img = backward_warping(img, H)
    
    cv.namedWindow("warped", cv.WINDOW_NORMAL)
    cv.imshow("warped", warp_img)
    if len(sys.argv) == 3:
        cv.imwrite(sys.argv[2], warp_img)
    cv.waitKey(0)
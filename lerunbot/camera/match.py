import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Sequence, Tuple, Union


def relative_position(
    img1: np.ndarray,
    img2: np.ndarray,
    calibration: Optional[dict] = None,
    num_features: int = 5000,  # DEPRECATED
    debug: bool = False,
    ref_point: Optional[Union[Sequence[float], np.ndarray]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate the relative position between two camera poses given two images and camera calibration.
    If a calibration chessboard is visible in both frames, it recovers real translation scale.
    The translation vector is expressed as the displacement of the reference point.

    Args:
        img1, img2: BGR images (np.ndarray).
        calibration: optional dict with keys 'mtx' (3x3) and optionally 'dist' (dist coeffs).
        num_features: ORB features.
        debug: if True, save debug_matches.jpg with inlier matches.
        ref_point: optional 3-vector (x,y,z) expressed in camera1 coordinate system.
                   Must be provided to compute relative translation.

    Returns:
        suc: bool, True if chessboard detected and scale recovered.
        t_vec: (3,) translation vector = ref_point_in_cam2 - ref_point
        quat: (4,) rotation quaternion (x,y,z,w)
    """
    if ref_point is None:
        raise ValueError("You must provide a reference point to get relative translation.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    if calibration is not None:
        K = np.asarray(calibration['mtx'], dtype=np.float64)
        dist = np.asarray(calibration.get('dist', np.zeros(5)), dtype=np.float64)
        gray1 = cv2.undistort(gray1, K, dist)
        gray2 = cv2.undistort(gray2, K, dist)

    # # ORB features [DEPRECATED]
    # orb = cv2.ORB_create(num_features)
    # kp1, des1 = orb.detectAndCompute(gray1, None)
    # kp2, des2 = orb.detectAndCompute(gray2, None)

    # if des1 is None or des2 is None:
    #     raise ValueError("ORB failed to find descriptors.")

    # # Matcher
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    # knn = bf.knnMatch(des1, des2, k=2)
    # good_matches = []
    # for m_n in knn:
    #     if len(m_n) != 2:
    #         continue
    #     m, n = m_n
    #     if m.distance < 0.75 * n.distance:
    #         good_matches.append(m)
    # if len(good_matches) < 10:
    #     raise ValueError("Not enough good matches.")

    # pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # E, mask_E = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # _, R_recov, t_dir, mask_pose = cv2.recoverPose(E, pts1, pts2, K, mask=mask_E)

    # if debug:
    #     inlier_indices = [i for i in range(len(good_matches)) if mask_pose[i, 0] == 1]
    #     inlier_matches = [good_matches[i] for i in inlier_indices]
    #     img_matches = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:200], None,
    #                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #     cv2.imwrite("debug_matches.jpg", img_matches)
    #     print(f"Debug: {len(inlier_matches)} inliers, saved debug_matches.jpg")

    # R_rel = R_recov.copy()
    # t_rel = t_dir.flatten()

    # Chessboard detection for scale recovery
    pattern_size = (8, 5)
    square_size = 0.03
    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size)
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size)

    if debug:
        # Visualize chessboard detection, merge both images side by side
        vis1 = img1.copy()
        vis2 = img2.copy()
        cv2.drawChessboardCorners(vis1, pattern_size, corners1, ret1)
        cv2.drawChessboardCorners(vis2, pattern_size, corners2, ret2)
        vis = np.hstack((vis1, vis2))
        cv2.imwrite("debug_chessboard.jpg", vis)
        print("Debug: saved debug_chessboard.jpg")

    chessboard_flag = False
    if ret1 and ret2:
        if debug:
            print("Chessboard detected in both images, recovering transformation.")

        criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
        corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
        corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size

        ok1, rvec1, tvec1 = cv2.solvePnP(objp, corners1, K, dist)
        ok2, rvec2, tvec2 = cv2.solvePnP(objp, corners2, K, dist)
        if ok1 and ok2:
            if debug:
                print("Successfully solved PnP for both images.")
            R1, _ = cv2.Rodrigues(rvec1)
            R2, _ = cv2.Rodrigues(rvec2)
            R_rel = R2 @ R1.T
            t_rel = (tvec2 - R2 @ R1.T @ tvec1).reshape(3,)
            chessboard_flag = True
    if not chessboard_flag:
        # return zero translation if no chessboard detected
        if debug:
            print("Chessboard detection failed, returning zero translation.")

        t_vec = np.zeros(3, dtype=np.float64)
        quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return False, t_vec, quat

    # Transform reference point
    t_rel[0] = -t_rel[0]
    rp = np.asarray(ref_point, dtype=np.float64).reshape(3,)
    t_vec = (R_rel @ rp + t_rel) - rp

    quat = R.from_matrix(R_rel).as_quat()
    return True, t_vec, quat


def _test():
    calibration = np.load('camera_calibration.npy', allow_pickle=True).item()
    cap = cv2.VideoCapture('/dev/video0')
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    print("Warming up the camera...")
    for _ in range(30):
        ret, frame = cap.read()  # Warm up the camera
    _ = input("Press Enter to capture the first image...")
    for _ in range(5):
        ret, frame1 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    _ = input("Press Enter to capture the second image...")
    for _ in range(5):
        ret, frame2 = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return
    suc, t_vec, quat = relative_position(frame1, frame2, calibration, debug=True, ref_point=[0.10, 0.00, 0.16])
    np.set_printoptions(precision=6, suppress=True)
    print("Relative Translation Vector:", t_vec)
    print("Relative Rotation Quaternion (x, y, z, w):", quat)


if __name__ == "__main__":
    _test()

import cv2
import numpy as np
import glob
import os

# 棋盘格的尺寸（横向和纵向交点的数量）
chessboard_size = (8, 5)

# 世界坐标系中的 3D 点
obj_points = []  # 存储 3D 点
img_points = []  # 存储 2D 图像点

# 创建棋盘格上的 3D 点，假设棋盘的每个方块为 1 单位大小
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# 读取图像文件
images = glob.glob('calibration_images/*.jpg')  # 修改为你的图像路径
os.makedirs('output_corners', exist_ok=True)  # 创建输出目录
print(images)

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 寻找棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
    cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
    cv2.imwrite(f'output_corners/{fname.split("/")[-1]}', img)  # 保存带角点的图像

    if ret:
        obj_points.append(objp)  # 添加 3D 点
        img_points.append(corners)  # 添加 2D 图像点

# 执行标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

print("ret", ret)
print("mtx", mtx)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)

calibration = {
    'ret': ret,
    'mtx': mtx,
    'dist': dist,
    'rvecs': rvecs,
    'tvecs': tvecs,
}
np.save('camera_calibration.npy', calibration)

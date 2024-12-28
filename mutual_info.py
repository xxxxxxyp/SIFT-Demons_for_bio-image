import cv2
import numpy as np

# 定义刚体变换函数，这里使用OpenCV的仿射变换矩阵来实现刚体变换
def rigid_transform_image(image, tx, ty, theta):
    """
    对输入图像应用刚体变换
    :param image: 待变换图像
    :param tx: x方向平移量（像素单位）
    :param ty: y方向平移量（像素单位）
    :param theta: 旋转角度（弧度制）
    :return: 变换后的图像
    """
    h, w = image.shape
    # 计算旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), np.degrees(theta), 1)
    # 平移量添加到旋转矩阵
    rotation_matrix[0, 2] += tx
    rotation_matrix[1, 2] += ty
    # 应用仿射变换（刚体变换是仿射变换的一种特殊形式）
    warped_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    return warped_image


# 计算经过刚体变换后图像的互信息
def mutual_info_loss(params, image1, image2):
    """
    计算互信息损失函数，目标是最小化这个损失，即最大化互信息
    :param params: 包含 [tx, ty, theta] 的参数列表，分别对应平移和旋转参数
    :return: 1 - 互信息，作为损失值（因为优化是求最小值，而我们要最大化互信息）
    """
    tx, ty, theta = params
    warped_image = rigid_transform_image(image2, tx, ty, theta)
    # 使用OpenCV的直方图计算函数来辅助计算互信息（这里是简化的计算思路示例）
    hist_1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
    hist_2 = cv2.calcHist([warped_image], [0], None, [256], [0, 256])
    hist_combined = cv2.calcHist([image1, warped_image], [0, 0], None, [256, 256], [0, 256, 0, 256])
    # 这里对互信息的计算是简单示意，实际中可能需要更精细的计算逻辑和归一化等处理
    mutual_info = 0
    for i in range(256):
        for j in range(256):
            if hist_combined[i, j] > 0 and hist_1[i] > 0 and hist_2[j] > 0:
                mutual_info += hist_combined[i, j] * np.log(
                    (hist_combined[i, j] * np.sum(hist_combined)) / (hist_1[i] * hist_2[j]))
    return 1 - mutual_info


import numpy as np
import cv2

def rigid_registration(src_points, dst_points):
    """
    估计两个点集之间的刚性变换矩阵。

    参数:
    src_points : 源图像中的特征点坐标，格式为 numpy 数组。
    dst_points : 目标图像中的特征点坐标，格式为 numpy 数组。

    返回值:
    rigid_transform : 刚性变换矩阵。
    """
    # 检查点的数量是否一致
    if len(src_points) != len(dst_points):
        raise ValueError("The number of points in src_points and dst_points must be the same.")

    # 将点集转换为 float32 类型
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    # 使用 estimateRigidTransform 函数计算刚性变换矩阵
    # 参数 fullAffine=False 表示我们只考虑平移和旋转，不考虑缩放
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

    return H

# 示例用法
# 假设我们已经有了两组匹配的特征点坐标
src_points = np.array([[50, 50], [150, 50], [50, 150]], dtype=np.float32)
dst_points = np.array([[55, 55], [155, 55], [55, 155]], dtype=np.float32)

# 计算刚性变换矩阵
rigid_transform = rigid_registration(src_points, dst_points)

# 使用计算出的刚性变换矩阵对源图像进行变换
# 假设 img1 是源图像
img1 = cv2.imread('image1.jpg', cv2.IMREAD_GRAYSCALE)
transformed_img = cv2.warpAffine(img1, rigid_transform, (img1.shape[1], img1.shape[0]))

# 显示变换后的图像
cv2.imshow('Transformed Image', transformed_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2
import numpy as np
import logging

# 配置日志记录器，设置日志级别为 DEBUG，并设置日志格式
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_mat_gradient(src):

    logger.debug('Get gradient from float...')
    # 定义 Sobel 梯度算子
    ker_x = np.array([[-1.0, 0.0, 1.0],
                      [-2.0, 0.0, 2.0],
                      [-1.0, 0.0, 1.0]], dtype=np.float32)
    ker_y = np.array([[-1.0, -2.0, -1.0],
                      [0.0, 0.0, 0.0],
                      [1.0, 2.0, 1.0]], dtype=np.float32)

    # 将源图像转换为单精度浮点数
    src_tmp = src.astype(np.float32)

    # 使用 Sobel 算子作卷积运算，得到 x、y 方向的梯度
    Fx_tmp = cv2.filter2D(src_tmp, -1, ker_x)
    Fy_tmp = cv2.filter2D(src_tmp, -1, ker_y)

    return Fx_tmp, Fy_tmp


def demons_one_wang(S, M, Sx, Sy, alpha, win_size, sigma):
    logger.debug('Demons one wang...')
    diff = M - S  # 计算差值图
    Tx_tmp = np.zeros_like(S, dtype=np.float32)
    Ty_tmp = np.zeros_like(S, dtype=np.float32)

    Mx, My = get_mat_gradient(M)  # 求浮动图像的梯度

    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            a1 = (Sx[i, j] ** 2 + Sy[i, j] ** 2 + alpha ** 2 * diff[i, j] ** 2)
            a2 = (Mx[i, j] ** 2 + My[i, j] ** 2 + alpha ** 2 * diff[i, j] ** 2)
            if (abs(a1) < 1e-7) or (abs(a2) < 1e-7):
                Tx_tmp[i, j] = 0.0
                Ty_tmp[i, j] = 0.0
            else:
                ax = Sx[i, j] / a1 + Mx[i, j] / a2
                Tx_tmp[i, j] = (-diff[i, j] * ax)

                ay = Sy[i, j] / a1 + My[i, j] / a2
                Ty_tmp[i, j] = (-diff[i, j] * ay)

    Tx_tmp *= 10  # 实际测试时，发现对计算得到的坐标偏移扩大一定倍数，可加快收敛速度
    Ty_tmp *= 10

    # 对坐标偏移进行高斯平滑，减小毛刺
    Tx = cv2.GaussianBlur(Tx_tmp, (win_size, win_size), sigma)
    Ty = cv2.GaussianBlur(Ty_tmp, (win_size, win_size), sigma)

    return Tx, Ty

def movepixels_2d2(src, Tx, Ty, interpolation):
    logger.debug('Move pixels...')

    src_tmp = src.astype(np.float32)
    dst_tmp = np.zeros_like(src_tmp, dtype=np.float32)

    Tx_map = np.zeros_like(Tx, dtype=np.float32)
    Ty_map = np.zeros_like(Ty, dtype=np.float32)
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            x = j + Tx[i, j]
            y = i + Ty[i, j]
            if 0 <= x < src.shape[1] and 0 <= y < src.shape[0]:
                Tx_map[i, j] = x
                Ty_map[i, j] = y
            else:
                Tx_map[i, j] = 0
                Ty_map[i, j] = 0

    dst = cv2.remap(src_tmp, Tx_map, Ty_map, interpolation)

    return dst

def Inertial_demons(S, M, alpha, win_size, sigma, num):
    Sx, Sy = get_mat_gradient(S)  # 求参考图像的梯度

    Tx = np.zeros_like(S, dtype=np.float32)
    Ty = np.zeros_like(S, dtype=np.float32)

    S_tmp = S.astype(np.float32)
    M_tmp = M.astype(np.float32)

    for i in range(num):  # 迭代过程
        Tx_tmp, Ty_tmp = demons_one_wang(S_tmp, M_tmp, Sx, Sy, alpha, win_size, sigma)
        Tx += Tx_tmp * 0.75  # 把上一层迭代计算得到的偏移量加入到当前层迭代的偏移量计算当中
        Ty += Ty_tmp * 0.75
        M_tmp = movepixels_2d2(M_tmp, Tx, Ty, cv2.INTER_CUBIC)  # 像素重采样

    D = M_tmp.astype(np.uint8)
    return D
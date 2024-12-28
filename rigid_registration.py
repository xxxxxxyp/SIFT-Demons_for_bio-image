import cv2
import numpy as np
import logging

# 配置日志记录器，设置日志级别为 DEBUG，并设置日志格式
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_rigid_transform(src_image, dst_image, src_keypoints, dst_keypoints, good_matches):
    logger.debug('Applying rigid transform...')

    # Matched points
    src_points = np.float32([src_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_points = np.float32([dst_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Instead of cv2.findHomography, because the projection doesn't need to be considered in bio-image matching
    H, _ =  cv2.estimateAffinePartial2D(src_points, dst_points)

    if H is None:
        raise ValueError("Cannot apply rigid transform.")

    if H.shape == (2, 3):
        H = np.vstack([H, [0, 0, 1]])

    h, w = src_image.shape[:2]
    transformed_image = cv2.warpPerspective(dst_image, H, (w, h))

    return transformed_image

# 假设你已经有了两张图片image_mri和image_ct，以及它们的特征点keypoints_mri和keypoints_ct，以及匹配特征点good_matches
# 你可以这样调用函数：
# transformed_image_ct = apply_rigid_transform(image_mri, image_ct, keypoints_mri, keypoints_ct, good_matches)

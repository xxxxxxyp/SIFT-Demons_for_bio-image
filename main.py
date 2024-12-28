from matplotlib import pyplot as plt
import sklearn.metrics as skm

from extract_deatures import computeKeypointsAndDescriptors as extract_features
# from match_features import match_features as match_features
from match_features import flann_match_features as match_features
from rigid_registration import apply_rigid_transform as apply_rigid_transform
from evaluate_registration import mutual_info as mutual_info
from non_rigid_registration import Inertial_demons as inertial_demons
from mutual_info import mutual_info_loss as mutual_info_loss
from mutual_info import rigid_transform_image as rigid_transform_image
import cv2


# 主流程
CD31 = cv2.imread('breast_CD.jpg', cv2.IMREAD_GRAYSCALE)
HE = cv2.imread('breast-HE.jpg', cv2.IMREAD_GRAYSCALE)

CD31 = cv2.GaussianBlur(CD31, (5, 5), 0)
HE = cv2.GaussianBlur(HE, (5, 5), 0)

CD31 = cv2.equalizeHist(CD31)
HE = cv2.equalizeHist(HE)

plt.imshow(CD31, cmap='gray')
plt.show()
plt.imshow(HE, cmap='gray')
plt.show()

score1 = mutual_info(CD31, HE)-1
print(score1)

keypointsCD31, descriptorsCD31 = extract_features(CD31)
keypointsHE, descriptorsHE = extract_features(HE)
matched_points = match_features(descriptorsCD31, descriptorsHE)

matchesGMS = cv2.xfeatures2d.matchGMS(size1=CD31.shape,
                                     size2=HE.shape,
                                     keypoints1=keypointsCD31,
                                     keypoints2=keypointsHE,
                                     matches1to2=matched_points,
                                     withRotation=False,
                                     withScale=False,
                                     thresholdFactor=6.0)

rigid_transformation =apply_rigid_transform(CD31,HE, keypointsCD31, keypointsHE, matched_points)

plt.imshow(rigid_transformation, cmap='gray')
plt.show()

# # 初始参数猜测（根据图像实际情况合理调整）
# initial_guess = [1, 3, 5]
# from scipy.optimize import minimize
# result = minimize(mutual_info_loss, initial_guess, args=(CD31, HE), method='BFGS')
# # 获取优化后的参数
# optimal_tx, optimal_ty, optimal_theta = result.x
# # 应用最优参数进行最终的刚体变换配准
# registered_image = rigid_transform_image(HE, optimal_tx, optimal_ty, optimal_theta)
# plt.imshow(registered_image, cmap='gray')
# plt.show()
# score1 = mutual_info(CD31, registered_image)-1
# print(score1)

non_rigid_registration = inertial_demons(CD31, rigid_transformation, 50.0, 599, 20.0, 75)

plt.imshow(non_rigid_registration, cmap='gray')
plt.show()

score1 = mutual_info(CD31, non_rigid_registration)-1
print(score1)
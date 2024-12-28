import numpy as np
import sklearn.metrics as skm


def mutual_info(img_1, img_2):
    img_1_flat = img_1.flatten()
    img_2_flat = img_2.flatten()

    score = skm.mutual_info_score(img_1_flat, img_2_flat)
    return score


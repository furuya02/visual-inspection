import cv2
import numpy as np
import numpy as np


# 輪郭取得
def get_corner(img):

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 二値化（閾値を150に設定）
    ret, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    # 輪郭取得
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours[0], binary


def move_target_to_center(org_img):
    work_img = np.copy(org_img)
    oh, ow, _ = org_img.shape
    center_img = np.zeros((oh, ow, 3))

    contour, binary = get_corner(work_img)

    # 外接の矩形を求める
    x, y, w, h = cv2.boundingRect(contour)

    # 確認用に矩形を描画（デバッグ用）
    work_img = cv2.rectangle(work_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    work_img = cv2.drawContours(work_img, contour, -1, (0, 80, 255), 2)

    # ターゲットを切り取り
    target_img = org_img[y : y + h, x : x + w]

    th, tw, _ = target_img.shape
    print("oh:{} ow:{} th:{} tw:{}".format(oh, ow, th, tw))
    dx = int((ow - tw) / 2)  # 横方向の移動距離
    dy = int((oh - th) / 2)  # 縦方向の移動距離
    center_img[dy : dy + th, dx : dx + tw] = target_img

    return center_img, work_img


def save_img(img, output_path, basename, name):
    save_file_name = "{}/{}_{}.jpg".format(output_path, name, basename)
    cv2.imwrite(save_file_name, img)
    return save_file_name

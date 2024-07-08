# weights/FastSAM.pt

import os
import cv2
import torch
import numpy as np
import FastSAM.fastsam as fastsam
from sensor import Sensor
from servo import Servo
from img_util import move_target_to_center
from img_util import save_img
from embedding import Embedding
import time

output_path = "./output"


# ガイドの表示
def disp_guide(frame, input_box):
    height, width, _ = frame.shape
    color = (200, 200, 0)
    cv2.rectangle(
        frame,
        pt1=(input_box[0], input_box[1]),
        pt2=(input_box[2], input_box[3]),
        color=color,
        thickness=2,
    )
    cv2.line(
        frame,
        pt1=(0, int(height / 2)),
        pt2=(width, int(height / 2)),
        color=color,
    )
    cv2.line(
        frame,
        pt1=(int(width / 2), 0),
        pt2=(int(width / 2), height),
        color=color,
    )


# 適切な大きさのサイズのマスクを取得する
def get_target_mask(annotations, cropped_img):
    tmp_img = np.copy(cropped_img)
    target_mask = None
    for annotation in annotations:
        mask = annotation.astype(np.uint8)

        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # デバッグ用に一次マスクの輪郭を描画
        tmp_img = cv2.drawContours(tmp_img, contours, -1, (0, 80, 255), 5)
        for contour in contours:
            # 外接の矩形を求める
            x, y, w, h = cv2.boundingRect(contour)
            target_size = w * h
            print("target_size:{}".format(target_size))
            if 260000 < target_size and target_size < 360000:
                print("set mask")
                target_mask = mask
    return target_mask, tmp_img


def display_images(frame, tmp_img, detect_img):
    magnification = 0.4
    # magnification = 1

    # カメラ画像の表示
    cv2.imshow("Frame", cv2.resize(frame, None, fx=magnification, fy=magnification))
    # 検出過程の画像(デバッグ用)
    cv2.imshow("tmp_img", cv2.resize(tmp_img, None, fx=magnification, fy=magnification))
    # # 検出物の表示
    cv2.imshow(
        "detect_img", cv2.resize(detect_img, None, fx=magnification, fy=magnification)
    )


def get_detect_img(cropped_img, target_mask):
    h, w, _ = cropped_img.shape
    detect_img = np.zeros((h, w), np.uint8)
    if target_mask is not None:
        detect_img = np.copy(cropped_img)
        h, w, _ = cropped_img.shape

        detect_img[:] = np.where(
            target_mask[:h, :w, np.newaxis] == True,
            cropped_img,
            (0, 0, 0),
        )
    return detect_img


def get_annotations(model, DEVICE, img):
    tmp_img = np.copy(img)
    # tmp_img = cv2.resize(tmp_img, None, fx=0.1, fy=0.1)
    target_mask = None
    everything_results = model(
        img,
        device=DEVICE,
        retina_masks=True,
        conf=0.8,
        iou=0.9,
    )
    try:
        annotations = everything_results[0].masks.data.cpu().numpy()
        print("shape:{} annotations:{}".format(img.shape, len(annotations)))
    except Exception:
        annotations = []

    if len(annotations) > 0:
        # 適切な大きさのサイズのマスクを取得する
        target_mask, tmp_img = get_target_mask(annotations, tmp_img)

    return target_mask, tmp_img


def loop(cap, detection_area, embedding):

    # FastSAMの初期化
    FAST_SAM_CHECKPOINT = "./weights/FastSAM.pt"
    model = fastsam.FastSAM(FAST_SAM_CHECKPOINT)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("FAST_SAM_CHECKPOINT:{} DEVICE:{}".format(FAST_SAM_CHECKPOINT, DEVICE))

    save_counter = 0
    sensor = Sensor()
    servo = Servo()
    # routeTarget = RouteTarget()
    counter = 0

    while True:
        try:
            # カメラ画像の取得
            ret, frame = cap.read()
            if ret is False:
                raise IOError

            # 検出エリアの切取り
            cropped_img = frame.copy()
            cropped_img = cropped_img[
                detection_area[1] : detection_area[3],
                detection_area[0] : detection_area[2],
            ]
            h, w, _ = cropped_img.shape
            detect_img = np.zeros((h, w), np.uint8)
            tmp_img = cropped_img.copy()
            target_mask = None

            # ガイドライン描画
            disp_guide(frame, detection_area)

            # センサーのチェック
            sensor_status = sensor.check()

            # Frame読み飛ばし (24FPS)
            counter += 1
            if counter % 12 == 0:

                if sensor_status == "on":
                    print("sensor_status:{}".format(sensor_status))

                # FastSAMによる検出
                target_mask, tmp_img = get_annotations(model, DEVICE, cropped_img)

                # 検出結果画像の生成 （マスク以外を黒く塗りつぶす）
                detect_img = get_detect_img(cropped_img, target_mask)

                # センサーのチェック
                if sensor_status == "on":
                    if target_mask is not None:
                        basename = "{:03}".format(save_counter)

                        save_img(detect_img, output_path, basename, "detect")
                        save_img(frame, output_path, basename, "frame")
                        save_img(tmp_img, output_path, basename, "tmp")

                        center_img, work_img = move_target_to_center(detect_img)
                        save_file_name = save_img(
                            center_img, output_path, basename, "center"
                        )
                        save_img(work_img, output_path, basename, "work")

                        cosine = embedding.compare(save_file_name)
                        print("==========================================")
                        if cosine < 0.9:
                            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            print("【{} {}】".format(cosine, basename))
                            print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
                            time.sleep(1)
                            servo.close_gate()
                        else:
                            print("【{} {}】".format(cosine, basename))
                        print("==========================================")

                        save_counter += 1
                        sensor.reset()

            # 画像の表示
            display_images(frame, tmp_img, detect_img)

            # if sendor.check(target_mask):
            #     basename = "{:03}".format(save_counter)

            #     save_img(detect_img, output_path, basename, "detect")
            #     save_img(frame, output_path, basename, "frame")
            #     save_img(tmp_img, output_path, basename, "tmp")

            #     center_img, work_img = move_target_to_center(detect_img)
            #     save_img(center_img, output_path, basename, "center")
            #     save_img(work_img, output_path, basename, "work")

            #     # center_img = cv2.imread("output/center_000.jpg")
            #     print(center_img.dtype)
            #     center_img = np.array(center_img, dtype="uint8")
            #     print(center_img.dtype)

            #     route_img, dst_img = routeTarget.route(center_img)
            #     save_img(route_img, output_path, basename, "route")
            #     save_img(dst_img, output_path, basename, "dst")

            #     save_counter += 1

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except KeyboardInterrupt:
            break


def main():

    os.makedirs(output_path, exist_ok=True)

    droid_cam = "http://192.168.1.41:4747/video/force/1920x1080"
    # droid_cam = "http://192.168.1.7:4747/video/force/640x480"
    cap = cv2.VideoCapture(droid_cam)
    # cap = cv2.VideoCapture(0)
    if cap.isOpened() is False:
        raise IOError
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, frame = cap.read()

    # 検出エリアの取得
    margin_x = 220
    margin_y = 60
    frame_h, frame_w, _ = frame.shape
    detection_area = np.array(
        [margin_x, margin_y, frame_w - margin_x, frame_h - margin_y]
    )

    base_image_path = os.path.join(os.path.dirname(__file__), "./base.jpg")
    embedding = Embedding(base_image_path)

    loop(cap, detection_area, embedding)

    cap.release()


if __name__ == "__main__":
    main()

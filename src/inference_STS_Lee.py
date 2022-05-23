
import os, glob
import argparse
import csv
import math
import cv2

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from skimage.io import imread
from google.protobuf import text_format

from shapely.geometry import Polygon

from builders import model_builder
from protos import pipeline_pb2
from utils.np_rbox_ops import non_max_suppression
import xml.etree.ElementTree as ET
import inference_FN_cal as fn

image_path = os.path.join('.', 'dataset', 'test', 'GIEP_TEST', 'images')
annotation_path = os.path.join('.', 'dataset', 'test', 'GIEP_TEST', 'annotations')
predict_path = './output_predict_image'
prediction_path = os.path.join('.', 'prediction_csv.csv')
prediction_csv = open(prediction_path, 'w', newline='')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def crop_rect(img, rect):
    # 크롭 방법은 cx, cy 센터와 w, h crop size를 가지고 실시한다.

    center = rect[0]
    size = rect[1]
    angle = rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    # get row and col num in img
    height, width = img.shape[0], img.shape[1]
    # print("width: {}, height: {}".format(width, height))

    # cv2.getRotationMatrix2D : cx, cy와 angle이 주어지면 반시계 방향으로 Rotation하는 Transposition Matrix를 생성 1은 Scale factor
    M = cv2.getRotationMatrix2D(center, angle, 1)

    # cv2.warpAffine(image, Transposition Matrix, (w, h)) -> image가 Transposition Matrix에 의해 회전, 이동 변환함.
    img_rot = cv2.warpAffine(img, M, (width, height))

    # cv2.getRectSubPix(image, (w, h), (cx, cy)) -> image를 crop함. angle 만큼 돌려줬으니 그냥 center와 size로 crop하면됨.
    img_crop = cv2.getRectSubPix(img_rot, size, center)

    return img_crop, img_rot

def Adding_undetected_objects(GT_labels, predict_bbox, prediction_path, img_num):
    for cls_GT, point1_x_GT, point1_y_GT, point2_x_GT, point2_y_GT, point3_x_GT, point3_y_GT, point4_x_GT, point4_y_GT, in GT_labels:
        IOU_NO = []
        for cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y in predict_bbox:
            if cls_GT == cls:
                GT_area = Polygon([(point1_x_GT, point1_y_GT), (point2_x_GT, point2_y_GT), (point3_x_GT, point3_y_GT),
                                   (point4_x_GT, point4_y_GT)])  # Ground true area
                bbox = Polygon([(point1_x, point1_y), (point2_x, point2_y), (point3_x, point3_y),
                                (point4_x, point4_y)])  # Prediction area
                # print("GT_area", GT_area)
                # print("bbox", bbox)
                iou = round(GT_area.intersection(bbox).area / GT_area.union(bbox).area, 3)  # IOU
                IOU_NO.append(iou)

            else:
                continue

        try:
            IOU_NO = max(IOU_NO)
            if IOU_NO == 0:
                prediction_write = open(prediction_path, 'a', newline='')
                wr = csv.writer(prediction_write)
                if cls_GT == 1:
                    class_name = "container"
                elif cls_GT == 2:
                    class_name = "oil tanker"
                elif cls_GT == 3:
                    class_name = "aircraft carrier"
                elif cls_GT == 4:
                    class_name = "maritime vessels"
                elif cls_GT == 5:
                    class_name = "war ship"
                wr.writerow([class_name, img_num, IOU_NO, IOU_NO, '0', 'False'])  # 미검출 객체 Flase
                # print("Cls_GT *******************,", cls_GT, class_name)
                prediction_write.close()

        except Exception as e:
            print(str(e))


def IOU_calibration(num, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y, class_name):
    train_file_name = os.path.join(annotation_path, "{0}.xml".format(num))  # IOU 계산하기
    # print("anndtation path :", annotation_path)
    # print("GT_image_name:", train_file_name)
    train_xml = ET.parse(train_file_name)
    IOU = []
    IOU_AP = []
    for object_1 in train_xml.iter("object"):
        object_name = str(object_1.find("name").text)
        for bndbox in object_1.iter("bndbox"):
            point1_x_GT = float(bndbox.find("point1_x").text)
            point1_y_GT = float(bndbox.find('point1_y').text)
            point2_x_GT = float(bndbox.find("point2_x").text)
            point2_y_GT = float(bndbox.find('point2_y').text)
            point3_x_GT = float(bndbox.find("point3_x").text)
            point3_y_GT = float(bndbox.find('point3_y').text)
            point4_x_GT = float(bndbox.find("point4_x").text)
            point4_y_GT = float(bndbox.find('point4_y').text)

            # print("GT:", point1_x_GT, point1_y_GT, point2_x_GT, point2_y_GT, point3_x_GT, point3_y_GT, point4_x_GT, point4_y_GT)
            # print("predict:", point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y)

            GT_area = Polygon([(point1_x_GT, point1_y_GT), (point2_x_GT, point2_y_GT), (point3_x_GT, point3_y_GT),
                               (point4_x_GT, point4_y_GT)])  # Ground true area
            bbox = Polygon([(point1_x, point1_y), (point2_x, point2_y), (point3_x, point3_y),
                            (point4_x, point4_y)])  # Prediction area

            # print("GT_area :", GT_area)
            # print("bbox :", bbox)

            intersection = round(GT_area.intersection(bbox).area, 3)
            union = round(GT_area.union(bbox).area, 3)

            # print(intersection)
            # print(union)

            iou = round(intersection / union, 3)  # IOU

            # print("ioU:", iou)

            # handle case where there is NO overlap
            if iou == 0:
                IOU.append(iou)
                IOU_AP.append(iou)

            else:
                IOU.append(iou)
                ####################################################### 오검출 객체 구분하기 (클래스 불일치, IOU overlap 기준치 미달)

                if iou > 0.1 and object_name == class_name:
                    IOU_AP.append(iou)

                else:
                    iou = 0
                    IOU_AP.append(iou)

    IOU = max(IOU)
    IOU_AP = max(IOU_AP)

    return IOU, IOU_AP


def rectangle_visualization(img, class_name, IOU, score, classes, point1_x, point1_y, point2_x, point2_y, point3_x,
                            point3_y, point4_x, point4_y):
    x1min = point1_x
    y1min = point1_y
    x2min = point2_x
    y2min = point2_y
    x3min = point3_x
    y3min = point3_y
    x4min = point4_x
    y4min = point4_y

    if classes == 1:
        # img2 = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,0,255), 3)          # 회색, 컨테이너선
        cv2.line(img, (x1min, y1min), (x2min, y2min), (0, 0, 255), 5)
        cv2.line(img, (x2min, y2min), (x3min, y3min), (0, 0, 255), 5)
        cv2.line(img, (x3min, y3min), (x4min, y4min), (0, 0, 255), 5)
        cv2.line(img, (x4min, y4min), (x1min, y1min), (0, 0, 255), 5)
        # cv2.imwrite("C:/Users/user/Desktop/add_test/{0}".format(img_num), line4)

    elif classes == 2:
        # img3 = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (0,255,255), 3)        # 노란색, 유조선
        cv2.line(img, (x1min, y1min), (x2min, y2min), (255, 255, 0), 5)
        cv2.line(img, (x2min, y2min), (x3min, y3min), (255, 255, 0), 5)
        cv2.line(img, (x3min, y3min), (x4min, y4min), (255, 255, 0), 5)
        cv2.line(img, (x4min, y4min), (x1min, y1min), (255, 255, 0), 5)
        # cv2.imwrite("C:/Users/user/Desktop/add_test/{0}".format(img_num), line4)

    elif classes == 3:
        # img4 = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,255), 3)
        cv2.line(img, (x1min, y1min), (x2min, y2min), (0, 255, 0), 5)  # 연두색, 항공모함
        cv2.line(img, (x2min, y2min), (x3min, y3min), (0, 255, 0), 5)
        cv2.line(img, (x3min, y3min), (x4min, y4min), (0, 255, 0), 5)
        cv2.line(img, (x4min, y4min), (x1min, y1min), (0, 255, 0), 5)
        # cv2.imwrite("C:/Users/user/Desktop/add_test/{0}".format(img_num), line4)

    elif classes == 4:
        # img5 = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (155,5,255), 3)        # 파랑색, 상선
        cv2.line(img, (x1min, y1min), (x2min, y2min), (255, 0, 0), 5)
        cv2.line(img, (x2min, y2min), (x3min, y3min), (255, 0, 0), 5)
        cv2.line(img, (x3min, y3min), (x4min, y4min), (255, 0, 0), 5)
        cv2.line(img, (x4min, y4min), (x1min, y1min), (255, 0, 0), 5)
        # cv2.imwrite("C:/Users/user/Desktop/add_test/{0}".format(img_num), line4)

    elif classes == 5:
        # img5 = cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (155,5,255), 3)        # 파랑색, 군함
        cv2.line(img, (x1min, y1min), (x2min, y2min), (255, 0, 255), 5)
        cv2.line(img, (x2min, y2min), (x3min, y3min), (255, 0, 255), 5)
        cv2.line(img, (x3min, y3min), (x4min, y4min), (255, 0, 255), 5)
        cv2.line(img, (x4min, y4min), (x1min, y1min), (255, 0, 255), 5)
        # cv2.imwrite("C:/Users/user/Desktop/add_test/{0}".format(img_num), line4)

    score_coordin = int(point1_y + 15)

    img = cv2.putText(img, "{0}".format(class_name), (point1_x, point1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                      1, cv2.LINE_AA)
    img = cv2.putText(img, "IOU:{0}".format(IOU), (point2_x, point2_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1,
                      cv2.LINE_AA)
    img = cv2.putText(img, "Prob:{0}".format(score), (point1_x, score_coordin), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                      (255, 0, 0), 1, cv2.LINE_AA)

    return img


def GT_label(num):
    # GT label class별 box 불러오기
    train_file_name = os.path.join(annotation_path, "{0}.xml".format(num))
    GT_labels = []
    train_xml = ET.parse(train_file_name)
    for object_1 in train_xml.iter("object"):
        object_name = str(object_1.find("name").text)
        for bndbox in object_1.iter("bndbox"):
            point1_x = int(bndbox.find("point1_x").text)
            point1_y = int(bndbox.find("point1_y").text)
            point2_x = int(bndbox.find("point2_x").text)
            point2_y = int(bndbox.find("point2_y").text)
            point3_x = int(bndbox.find("point3_x").text)
            point3_y = int(bndbox.find("point3_y").text)
            point4_x = int(bndbox.find("point4_x").text)
            point4_y = int(bndbox.find("point4_y").text)

            if object_name == 'container':
                cls = int(1)
                GT_area = [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]
                GT_labels.append(GT_area)

            elif object_name == 'oil tanker':
                cls = int(2)
                GT_area = [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]
                GT_labels.append(GT_area)

            elif object_name == 'aircraft carrier':
                cls = int(3)
                GT_area = [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]
                GT_labels.append(GT_area)

            elif object_name == 'maritime vessels':
                cls = int(4)
                GT_area = [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]
                GT_labels.append(GT_area)


            elif object_name == 'war ship':
                cls = int(5)
                GT_area = [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y]
                GT_labels.append(GT_area)

    return GT_labels


def get_detection_graph(pipeline_config_path):
    """build a graph from pipline_config_path

    :param: str pipeline_config_path: path to pipeline config file

    :return: graph
    """

    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, 'r') as f:
        text_format.Merge(f.read(), pipeline_config)

    detection_model = model_builder.build(pipeline_config.model, is_training=False)
    input_tensor = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='image_tensor')
    inputs = tf.to_float(input_tensor)
    preprocessed_inputs = detection_model.preprocess(inputs)
    output_tensors = detection_model.predict(preprocessed_inputs)
    postprocessed_tensors = detection_model.postprocess(output_tensors)

    output_collection_name = 'inference_op'
    boxes = postprocessed_tensors.get('detection_boxes')
    scores = postprocessed_tensors.get('detection_scores')
    classes = postprocessed_tensors.get('detection_classes') + 1
    num_detections = postprocessed_tensors.get('num_detections')
    outputs = dict()
    outputs['detection_boxes'] = tf.identity(boxes, name='detection_boxes')
    outputs['detection_scores'] = tf.identity(scores, name='detection_scores')
    outputs['detection_classes'] = tf.identity(classes, name='detection_classes')
    outputs['num_detections'] = tf.identity(num_detections, name='num_detections')
    for output_key in outputs:
        tf.add_to_collection(output_collection_name, outputs[output_key])

    graph = tf.get_default_graph()

    return graph


def convert_rbox_to_poly(rbox):
    """ Convert RBox to polygon as 4 points

    :param numpy rbox: rotated bounding box as [cy, cx, height, width, angle]
    :return: list of tuple as 4 corner points
    """

    cy, cx = rbox[0], rbox[1]
    height, width = rbox[2], rbox[3]
    angle = rbox[4]

    lt_x, lt_y = -width / 2, -height / 2
    rt_x, rt_y = width / 2, -height / 2
    lb_x, lb_y = -width / 2, height / 2
    rb_x, rb_y = width / 2, height / 2

    lt_x_ = lt_x * math.cos(angle) - lt_y * math.sin(angle)
    lt_y_ = lt_x * math.sin(angle) + lt_y * math.cos(angle)
    rt_x_ = rt_x * math.cos(angle) - rt_y * math.sin(angle)
    rt_y_ = rt_x * math.sin(angle) + rt_y * math.cos(angle)
    lb_x_ = lb_x * math.cos(angle) - lb_y * math.sin(angle)
    lb_y_ = lb_x * math.sin(angle) + lb_y * math.cos(angle)
    rb_x_ = rb_x * math.cos(angle) - rb_y * math.sin(angle)
    rb_y_ = rb_x * math.sin(angle) + rb_y * math.cos(angle)

    lt_x_ = lt_x_ + cx
    lt_y_ = lt_y_ + cy
    rt_x_ = rt_x_ + cx
    rt_y_ = rt_y_ + cy
    lb_x_ = lb_x_ + cx
    lb_y_ = lb_y_ + cy
    rb_x_ = rb_x_ + cx
    rb_y_ = rb_y_ + cy

    return [(lt_x_, lt_y_), (rt_x_, rt_y_), (rb_x_, rb_y_), (lb_x_, lb_y_)]


def save_det_to_csv(dst_path, det_by_file):
    """ Save detected objects to CSV format

    :param str dst_path: Path to save csv
    :param dict det_by_file: detected objects that key is filename
    :return: None (save csv file)
    """
    with open(dst_path, 'w') as f:
        w = csv.DictWriter(f, ['file_name', 'class_id', 'confidence', 'point1_x', 'point1_y', 'point2_x', 'point2_y',
                               'point3_x', 'point3_y', 'point4_x', 'point4_y'])
        w.writeheader()

        # new_ff = open('C:/Users/user/Desktop/GIEP_1end_(0.3)_STS.csv', 'w', newline='')

        for file_path, det in det_by_file.items():
            rboxes = det['rboxes']
            classes = det['classes']
            scores = det['scores']

            img_num = file_path.split("\\")[-1]
            img_num = img_num.split(".")[0]
            image_name = '{0}.png'.format(img_num)
            # print("image_name", image_name)                # C:/Users/PC/Desktop/simplified_rbox_cnn-master/simplified_rbox_cnn-master/dataset/test/images_GIEP(3)\0133.png
            image = cv2.imread(os.path.join(image_path, image_name))

            path = os.path.join(predict_path, img_num)
            savename = path + "_result.png"
            # print("savename", savename)

            predict_bbox = []
            for rbox, cls, score in zip(rboxes, classes, scores):
                poly = convert_rbox_to_poly(rbox)

                point1_x = int(poly[0][0])
                point1_y = int(poly[0][1])
                point2_x = int(poly[1][0])
                point2_y = int(poly[1][1])
                point3_x = int(poly[2][0])
                point3_y = int(poly[2][1])
                point4_x = int(poly[3][0])
                point4_y = int(poly[3][1])

                # mask image gray scale로 불러오기.
                img_path = "E:/my_rotetion_detector/SES_SEGMENT_MASK/pred_mask/mask/{0}".format(image_name)
                img = cv2.imread(img_path, flags=0)

                # object detection 결과 출력 순서 [cy, cx, height, width, angle] -> ((cx, cy), (w, h), angle) 의 rect dict를 만들어준다.
                rect = ((rbox[1], rbox[0]), (rbox[3], rbox[2]), math.degrees(rbox[4]))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                # print("bounding box: {}".format(box))

                im_crop, img_rot = crop_rect(img, rect)

                im_crop = im_crop / 255
                a = np.array(im_crop, dtype=np.int64)
                iou = np.mean(a)
                print(iou)

                if iou < 0.6752:
                    det_dict = {'file_name': os.path.basename(file_path),
                                'class_id': cls,
                                'confidence': score,
                                'point1_x': poly[0][0],
                                'point1_y': poly[0][1],
                                'point2_x': poly[1][0],
                                'point2_y': poly[1][1],
                                'point3_x': poly[2][0],
                                'point3_y': poly[2][1],
                                'point4_x': poly[3][0],
                                'point4_y': poly[3][1],
                                }
                    w.writerow(det_dict)

                    #######################################################################

                    if cls == 1:
                        class_name = "container"
                    elif cls == 2:
                        class_name = "oil tanker"
                    elif cls == 3:
                        class_name = "aircraft carrier"
                    elif cls == 4:
                        class_name = "maritime vessels"
                    elif cls == 5:
                        class_name = "war ship"
                    

                    ###############################################################################

                    IOU, IOU_AP = IOU_calibration(img_num, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y,
                                                  point4_x, point4_y, class_name)  # IOU 계산
                    score = format(score, ".3f")
                    im = rectangle_visualization(image, class_name, IOU, score, cls, point1_x, point1_y, point2_x,
                                                 point2_y, point3_x, point3_y, point4_x, point4_y)
                    # print("Class:", cls)
                    # print("IOU:", IOU)
                    # print("Score:", score)
                    prediction_write = open(prediction_path, 'a', newline='')
                    wr = csv.writer(prediction_write)
                    if IOU_AP == 0:
                        wr.writerow([class_name, img_num, IOU_AP, IOU, score,
                                     'False'])  # 오검출(클래스 불균형, IOU overlap 기준치 미달) 객체 False
                    else:
                        wr.writerow([class_name, img_num, IOU_AP, IOU, score,
                                     'True'])  # 오검출(클래스 불균형, IOU overlap 기준치 미달) 아닌 객체 True
                    prediction_write.close()

                    ########################################## 미검출 객체 추가하기

                    predict_bbox.append(
                        [cls, point1_x, point1_y, point2_x, point2_y, point3_x, point3_y, point4_x, point4_y])

                    cv2.imwrite(savename, im)

                else:
                    # print(image_num, class_name, rbox, cls, score)
                    continue

            GT_labels = GT_label(img_num)

            fn.fn_object_cal(GT_labels, predict_bbox, prediction_path, img_num)


def get_patch_generator(image, patch_size, overlay_size):
    """ Patch Generator to split image by grid

    :param numpy image: source image
    :param int patch_size: patch size that width and height of patch is equal
    :param overlay_size: overlay size in patches
    :return: generator for patch image, row and col coordinates
    """
    step = patch_size - overlay_size
    for row in range(0, image.shape[0] - overlay_size, step):
        for col in range(0, image.shape[1] - overlay_size, step):
            # Handling for out of bounds
            patch_image_height = patch_size if image.shape[0] - row > patch_size else image.shape[0] - row
            patch_image_width = patch_size if image.shape[1] - col > patch_size else image.shape[1] - col

            # Set patch image
            patch_image = image[row: row + patch_image_height, col: col + patch_image_width]

            # Zero padding if patch image is smaller than patch size
            if patch_image_height < patch_size or patch_image_width < patch_size:
                pad_height = patch_size - patch_image_height
                pad_width = patch_size - patch_image_width
                patch_image = np.pad(patch_image, ((0, pad_height), (0, pad_width), (0, 0)), 'constant')

            yield patch_image, row, col


def inference(pipeline_config_path, ckpt_path, image_dir, dst_path, patch_size, overlay_size, class_num):
    """ Inference images to detect objects

    :param str pipeline_config_path: path to a pipeline_pb2.TrainEvalPipelineConfig config file
    :param str ckpt_path: path to trained checkpoint
    :param str image_dir: directory to source images
    :param str dst_path: path to save detection output
    :param int patch_size: patch size that width and height of patch is equal
    :param int overlay_size: overlay size in patches
    :return: None (save detection output)

    """
    # Get filenames
    file_paths = [os.path.join(root, name) for root, dirs, files in os.walk(image_dir) for name in files if
                  name.endswith('png')]

    # Create graph
    graph = get_detection_graph(pipeline_config_path)

    # Inference
    with tf.Session(graph=graph) as sess:
        # Load weights from a checkpoint file
        variables_to_restore = tf.global_variables()
        saver = tf.train.Saver(variables_to_restore)
        saver.restore(sess, ckpt_path)

        # Get tensors of detection model
        image_tensor = graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = graph.get_tensor_by_name('detection_scores:0')
        detection_classes = graph.get_tensor_by_name('detection_classes:0')

        # Run detection
        det_by_file = dict()
        for file_path in tqdm(file_paths):
            image = imread(file_path)
            patch_generator = get_patch_generator(image, patch_size=patch_size, overlay_size=overlay_size)

            classes_list, scores_list, rboxes_list = list(), list(), list()
            for patch_image, row, col in patch_generator:
                classes, scores, rboxes = sess.run([detection_classes, detection_scores, detection_boxes],
                                                   feed_dict={image_tensor: [patch_image]})

                # detection 결과 저장 rboxes : cx, cy, w, h, angle 순서는 파악해야함. 여튼 5개, classes : object class, scores : confidence(classfication 점수?)
                rboxes = rboxes[0]
                classes = classes[0]
                scores = scores[0]

                # normalization 되서 나온 좌표 값을 cx, cy, w, h * 768, angle * 1 만 곱해줌 그리고 cx, cy는 행, 렬 까지 고려
                rboxes *= [patch_image.shape[0], patch_image.shape[1], patch_image.shape[0], patch_image.shape[1], 1]
                rboxes[:, 0] = rboxes[:, 0] + row
                rboxes[:, 1] = rboxes[:, 1] + col

                # patch로 나눴으니 1개의 image의 정보로 합치기 위해 list에 순차적으로 저장.
                rboxes_list.append(rboxes)
                classes_list.append(classes)
                scores_list.append(scores)

            # 각 patch별로 100개의 detection 결과를 3600개로 만들기.
            rboxes = np.array(rboxes_list).reshape(-1, 5)
            classes = np.array(classes_list).flatten()
            scores = np.array(scores_list).flatten()

            # confidence 0.2 이상되는 데이터만 저장.
            rboxes = rboxes[scores > 0.2]
            classes = classes[scores > 0.2]
            scores = scores[scores > 0.2]

            rboxes_nms, classes_nms, scores_nms = list(), list(), list()

            # sub_class nms algorithm
            for i in range(1, class_num+1):
                idx = np.where(classes == float(i))[0]

                if len(idx) == 0:
                    continue
                sub_cls_rboxes = rboxes[idx]
                sub_cls_classes = classes[idx]
                sub_cls_scores = scores[idx]

                indices = non_max_suppression(sub_cls_rboxes, sub_cls_scores, iou_threshold=0.3)

                rboxes_nms.extend(np.array(sub_cls_rboxes[indices]))
                classes_nms.extend(np.array(sub_cls_classes[indices]))
                scores_nms.extend(np.array(sub_cls_scores[indices]))

            final_rboxes = np.array(rboxes_nms)
            final_classes = np.array(classes_nms)
            final_scores = np.array(scores_nms)

            # iou 계산, csv file 저장, OES 적용을 위해 det_by_file dict에 예측 결과 저장.
            det_by_file[file_path] = {'rboxes': final_rboxes, 'classes': final_classes, 'scores': final_scores}

        # Save detection output
        save_det_to_csv(dst_path, det_by_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--pipeline_config_path', type=str,
                        help='Path to a pipeline_pb2.TrainEvalPipelineConfig config file.')
    parser.add_argument('--ckpt_path', type=str,
                        help='Path to trained checkpoint, typically of the form path/to/model-%step.ckpt')
    parser.add_argument('--image_dir', default='./dataset/test/GIEP_TEST/images',type=str,
                        help='Path to images to be inferred')
    parser.add_argument('--dst_path', default='./dataset/test/dst_ses/processing_rboxcnn.csv', type=str,
                        help='Path to save detection output')
    parser.add_argument('--patch_size', type=int, default=768,
                        help='Patch size, width and height of patch is equal.')
    parser.add_argument('--overlay_size', type=int, default=256,
                        help='Overlay size for patching.')
    parser.add_argument('--class_num', type=int, default=5,
                        help='Overlay size for patching.')

    args = parser.parse_args()

    inference(**vars(args))

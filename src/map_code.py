import numpy as np
import pandas as pd
import csv
import os
import math
import matplotlib.pyplot as plt


def csv_making(classes, class_name, path_new_csv, image_path, IoU_AP, IoU, prob, T_F):
    if classes == class_name:
        mAP_write = open(path_new_csv, 'a', newline='')
        wr = csv.writer(mAP_write)
        wr.writerow([classes, image_path, IoU_AP, IoU, prob, T_F])
        mAP_write.close()


def each_class_total_detection_num(path):
    f = open(path, 'r')
    csv_file = csv.reader(f)

    container_TP = 0
    container_FN = 0
    oil_tanker_TP = 0
    oil_tanker_FN = 0
    aircraft_carrier_TP = 0
    aircraft_carrier_FN = 0
    maritime_vessels_TP = 0
    maritime_vessels_FN = 0
    war_ship_TP = 0
    war_ship_FN = 0

    for line in csv_file:
        classes_name = line[1]
        TP_FP_FN = line[6]

        if classes_name == "container" and TP_FP_FN == "True":
            container_TP += 1
        elif classes_name == "container" and TP_FP_FN == "FN":
            container_FN += 1
        if classes_name == "oil tanker" and TP_FP_FN == "True":
            oil_tanker_TP += 1
        elif classes_name == "oil tanker" and TP_FP_FN == "FN":
            oil_tanker_FN += 1
        if classes_name == "aircraft carrier" and TP_FP_FN == "True":
            aircraft_carrier_TP += 1
        elif classes_name == "aircraft carrier" and TP_FP_FN == "FN":
            aircraft_carrier_FN += 1
        if classes_name == "maritime vessels" and TP_FP_FN == "True":
            maritime_vessels_TP += 1
        elif classes_name == "maritime vessels" and TP_FP_FN == "FN":
            maritime_vessels_FN += 1
        if classes_name == "war ship" and TP_FP_FN == "True":
            war_ship_TP += 1
        elif classes_name == "war ship" and TP_FP_FN == "FN":
            war_ship_FN += 1
        else:
            pass

    return container_TP + container_FN, oil_tanker_TP + oil_tanker_FN, aircraft_carrier_TP + aircraft_carrier_FN, maritime_vessels_TP + maritime_vessels_FN, war_ship_TP + war_ship_FN


def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def tp_fp_cal(csv_path, recall_total_num):
    f = open(csv_path, 'r')
    rdr = csv.reader(f)

    row_count = sum(1 for row in rdr if rdr)

    tp = np.zeros(row_count)
    fp = np.zeros(row_count)

    idx = 0

    fr = open(csv_path, 'r')
    frdr = csv.reader(fr)

    for line in frdr:
        T_or_F = line[5]

        if T_or_F == 'FN':
            continue

        if T_or_F == 'True':
            tp[idx] = 1.
        else:
            fp[idx] = 1.
        idx += 1

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(recall_total_num)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    ap = voc_ap(rec, prec, use_07_metric=True)

    return ap, rec, prec


if __name__ == "__main__":

    container = 'container'
    oil_tanker = 'oil tanker'
    aircraft_carrier = 'aircraft carrier'
    maritime_vessels = 'maritime vessels'
    war_ship = 'war ship'

    path = os.path.join('.')
    path_result = os.path.join(path, 'test_result_csv')

    path_csv = os.path.join(path_result, 'prediction_csv.csv')
    path_csv_sort = os.path.join(path, 'prediction_csv_sort.csv')

    path_new_csv_Container = os.path.join(path, 'new_mAP_Container.csv')
    path_new_csv_Oil_tanker = os.path.join(path, 'new_mAP_Oil_tanker.csv')
    path_new_csv_Aircraft_carrier = os.path.join(path, 'new_mAP_Aircraft_carrier.csv')
    path_new_csv_Maritime_vessels = os.path.join(path, 'new_mAP_Maritime_vessels.csv')
    path_new_csv_War_ship = os.path.join(path, 'new_mAP_War_ship.csv')

    new_Container_csv = open(path_new_csv_Container, 'w', newline='')
    new_Oil_tanker_csv = open(path_new_csv_Oil_tanker, 'w', newline='')
    new_Aircraft_carrier_csv = open(path_new_csv_Aircraft_carrier, 'w', newline='')
    new_Maritime_vessels_csv = open(path_new_csv_Maritime_vessels, 'w', newline='')
    new_War_ship_csv = open(path_new_csv_War_ship, 'w', newline='')

    dfTitanic = pd.read_csv(path_csv, header=None)
    dfTitanic.sort_values(by=4, ascending=False, inplace=True)
    dfTitanic.to_csv(path_csv_sort, header=False, index=True)

    f = open(path_csv_sort, 'r')
    rdr = csv.reader(f)

    for line in rdr:
        print(line)
        classes = line[1]
        print(classes)
        image_path = line[2]
        IoU_AP = line[3]
        IoU = line[4]
        prob = line[5]
        T_F_FN = line[6]

        # Inference된 result를 각 Class별로 csv 파일 생성
        if T_F_FN == 'FN':
            continue
        csv_making(classes, container, path_new_csv_Container, image_path, IoU_AP, IoU, prob, T_F_FN)
        csv_making(classes, oil_tanker, path_new_csv_Oil_tanker, image_path, IoU_AP, IoU, prob, T_F_FN)
        csv_making(classes, aircraft_carrier, path_new_csv_Aircraft_carrier, image_path, IoU_AP, IoU, prob, T_F_FN)
        csv_making(classes, maritime_vessels, path_new_csv_Maritime_vessels, image_path, IoU_AP, IoU, prob, T_F_FN)
        csv_making(classes, war_ship, path_new_csv_War_ship, image_path, IoU_AP, IoU, prob, T_F_FN)

    Container_total_num, Oil_tanker_total_num, Aircraft_carrier_total_num, \
    Maritime_vessels_total_num, war_ship_total_num = each_class_total_detection_num(path_csv_sort)

    ap_container, recall_container, precision_container = tp_fp_cal(path_new_csv_Container, Container_total_num)
    ap_oil_tanker, recall_oil_tanker, precision_oil_tanker = tp_fp_cal(path_new_csv_Oil_tanker, Oil_tanker_total_num)
    ap_aircraft_carrier, recall_aircraft_carrier, precision_aircraft_carrier = tp_fp_cal(path_new_csv_Aircraft_carrier, Aircraft_carrier_total_num)
    ap_maritime_vessels, recall_maritime_vessels, precision_maritime_vessels = tp_fp_cal(path_new_csv_Maritime_vessels, Maritime_vessels_total_num)
    ap_war_ship, recall_war_ship, precision_war_ship = tp_fp_cal(path_new_csv_War_ship, war_ship_total_num)

    map = (ap_container + ap_oil_tanker + ap_aircraft_carrier + ap_maritime_vessels + ap_war_ship) / 5.0

    print(ap_container)
    print(ap_oil_tanker)
    print(ap_aircraft_carrier)
    print(ap_maritime_vessels)
    print(ap_war_ship)

    print(f'map : {map}')

    f = open('result.txt', 'w')
    f.write(f"AP_container: {ap_container}\n")
    f.write(f"AP_oil_tanker: {ap_oil_tanker}\n")
    f.write(f"AP_aircraft_carrier: {ap_aircraft_carrier}\n")
    f.write(f"AP_maritime_vessels: {ap_maritime_vessels}\n")
    f.write(f"AP_war_ship: {ap_war_ship}\n")
    f.write(f"mAP: {map}")
    f.close()

    # fig, ax = plt.subplots()
    # ax.plot(recall_container, precision_container, color='purple')
    #
    # # add axis labels to plot
    # ax.set_title('Precision-Recall Curve')
    # ax.set_ylabel('Precision')
    # ax.set_xlabel('Recall')
    #
    # # display plot
    # plt.savefig('./test_result_csv/container_pr_curve.png')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:57:47 2023

@author: weiwang
"""
import numpy as np
import copy
import os
import re
from xml.etree.ElementTree import ElementTree
import pickle
import argparse
import csv

def make_dir_recursively(input_path):
    # Create the directory recursively if it doesn't exist
    os.makedirs(os.path.dirname(input_path), exist_ok=True)

def save_one_layer_dict(res_dict, out_dir, out_name):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = os.path.join(out_dir, out_name)
    with open(out_file, 'w', newline='') as csvfile:
        fieldnames = list(res_dict.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(res_dict[fieldnames[0]])):
            row = {}
            for key in fieldnames:
                row[key] = res_dict[key][i]
            writer.writerow(row)

def save_two_layer_dict(my_dict, csv_folder, csv_file):
    outer_keys = sorted(my_dict.keys())
    inner_keys = sorted(my_dict[outer_keys[0]].keys())
    data = []
    data.append([""] + outer_keys)
    for inner_key in inner_keys:
        row_data = [inner_key]
        for outer_key in outer_keys:
            row_data.append(my_dict[outer_key][inner_key])
        data.append(row_data)
    csv_path = os.path.join(csv_folder, csv_file)
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data)

def save_pickle_file(pickle_folder, pickle_file, to_save):
    with open(pickle_folder + pickle_file, 'wb') as handle:
        pickle.dump(to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle_file(pickle_folder, pickle_file):
    with open(pickle_folder + pickle_file, 'rb') as handle:
        conts = pickle.load(handle)
    return conts
    
def matching_vals(string):
    num_pattern = r"\d+(?:\.\d+)?"
    non_num_pattern = r"[a-zA-Z_]+"
    
    num_matches = re.findall(num_pattern, string)
    non_num_matches = re.findall(non_num_pattern, string)
    
    num_list = [float(match) for match in num_matches]
    non_num_list = [str(match) for match in non_num_matches]
    
    return num_list, non_num_list

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree

def get_files_with_suffix(folder_path, suffix):
    files = os.listdir(folder_path)
    return [file for file in files if file.endswith(suffix)]

def read_vals_within_txt_file(file_folder, file_path, N):
    with open(file_folder + file_path, 'r') as f:
        lines = f.readlines()
    f.close()
    res = []
    for l in lines:
        if not l:
            continue
        else:
            numbers, non_numbers = matching_vals(l)
            if len(numbers) == N:
                res.append({non_numbers[0]: numbers})
            else:
                obj_len = len(numbers) // N
                for j in range(obj_len):
                    res.append({non_numbers[j] : numbers[j * N : (j + 1) * N]})
    return res

def read_vals_within_xml_file(file_folder, file_path):
    tree = read_xml(file_folder + file_path)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    res = []
    for child in root.findall('object'):
        bndbox = child.find('bndbox')
        spe = child.find('name').text
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)
        res.append({spe:[xmin,ymin,xmax,ymax]})
    return img_width, img_height, res

def construct_detect_dict(file_folder):
    all_files = get_files_with_suffix(file_folder, 'txt')
    res_dict = {}
    for f in all_files:
        temp_res = read_vals_within_txt_file(file_folder, f, 5)
        file_front = f.split('.')[0]
        res_dict[file_front] = temp_res
    return res_dict

def construct_GT_dict(file_folder):
    all_files = get_files_with_suffix(file_folder, 'xml')
    res_dict, size_dict = {}, {}
    for f in all_files:
        img_width, img_height, temp_res = read_vals_within_xml_file(file_folder, f)
        file_front = f.split('.')[0]
        res_dict[file_front] = temp_res
        size_dict[file_front] = [img_width, img_height]
    return res_dict, size_dict

def compute_iou(box_1, box_2):
    x1, y1, x2, y2 = box_1
    x3, y3, x4, y4 = box_2
    xA = max(x1, x3)
    yA = max(y1, y3)
    xB = min(x2, x4)
    yB = min(y2, y4)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (x2 - x1 + 1) * (y2 - y1 + 1)
    boxBArea = (x4 - x3 + 1) * (y4 - y3 + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def if_bound(box_1, box_2, thresh):
    tl_x1, tl_y1, br_x1, br_y1 = box_1
    tl_x2, tl_y2, br_x2, br_y2 = box_2
    area_box_2 = (br_x2 - tl_x2) * (br_y2 - tl_y2)
    overlap_x = max(0, min(br_x1, br_x2) - max(tl_x1, tl_x2))
    overlap_y = max(0, min(br_y1, br_y2) - max(tl_y1, tl_y2))
    overlap_area = overlap_x * overlap_y
    if area_box_2 == 0:
        percentage_within = 0
    else:
        percentage_within = overlap_area / area_box_2
    return percentage_within > thresh

def remove_bound_obj_for_single_img(img_txt_res_list, thresh):
    to_delete, box_num = [], len(img_txt_res_list)
    for i in range(box_num):
        if i in to_delete:
            continue
        else:
            box_1 = list(img_txt_res_list[i].values())[0][1:]
            for j in range(box_num):
                if i == j or j in to_delete:
                    continue
                else:
                    box_2 = list(img_txt_res_list[j].values())[0][1:]
                    if if_bound(box_1, box_2, thresh):
                        to_delete.append(j)
    if not to_delete:
        return img_txt_res_list
    else:
        output = []
        for k in range(box_num):
            if k not in to_delete:
                output.append(img_txt_res_list[k])
        return output

def remove_nonspe_obj_for_single_img(img_txt_res_list, spe_res_list, spe, thresh):
    to_delete, box_num = [], len(img_txt_res_list) if img_txt_res_list else 0
    spe_num = len(spe_res_list) if spe_res_list else 0
    if spe_num == 0:
        output = []
    else:
        for i in range(box_num):
            temp_box = list(img_txt_res_list[i].values())[0][1:]
            is_found = False
            for j in range(spe_num):
                temp_spe_box = list(spe_res_list[j].values())[0][1:]
                temp_spe = list(spe_res_list[j].keys())[0]
                if spe == temp_spe:
                    if if_bound(temp_spe_box, temp_box, thresh):
                        is_found = True
                        continue
            if not is_found:
                to_delete.append(i)        
    if not to_delete:
        return img_txt_res_list
    else:
        output = []
        for k in range(box_num):
            if k not in to_delete:
                output.append(img_txt_res_list[k])
        return output

def remove_overlap_obj_for_single_img(img_txt_res_list, thresh):
    to_delete, box_num = [], len(img_txt_res_list)
    for i in range(box_num):
        if i in to_delete:
            continue
        else:
            box_1 = list(img_txt_res_list[i].values())[0][1:]
            p_1 = list(img_txt_res_list[i].values())[0][0]
            for j in range(box_num):
                if i == j or j in to_delete:
                    continue
                else:
                    box_2 = list(img_txt_res_list[j].values())[0][1:]
                    p_2 = list(img_txt_res_list[j].values())[0][0]
                    temp_iou = compute_iou(box_1, box_2)
                    if temp_iou >= thresh:
                        if p_1 >= p_2:
                            to_delete.append(j)
                        else:
                            to_delete.append(i)
                            break
            if not to_delete:
                return img_txt_res_list
            else:
                output = []
                for i in range(box_num):
                    if i not in to_delete:
                        output.append(img_txt_res_list[i])
                return output

def remove_low_pvals(res_dict, p_thresh):
    refined_dict = {}
    for k in res_dict.keys():
        temp_res = res_dict.get(k)
        if not temp_res:
            continue
        res = []
        for t_r in temp_res:
            temp_list = list(t_r.values())[0]
            if temp_list[0] >= p_thresh:
                res.append(t_r)
        refined_dict[k] = res
    return refined_dict

def remove_bound_objs(res_dict, thresh):
    for k in res_dict.keys():
        temp_list = res_dict.get(k)
        new_temp_list = remove_bound_obj_for_single_img(temp_list, thresh)
        res_dict[k] = new_temp_list
    return res_dict

def remove_overlapped_objs(res_dict, thresh):
    for k in res_dict.keys():
        temp_list = res_dict.get(k)
        new_temp_list = remove_overlap_obj_for_single_img(temp_list, thresh)
        res_dict[k] = new_temp_list
    return res_dict

def remove_nonspe_objs(res_dict, spe_det_dict, spe, thresh):
    for k in res_dict.keys():
        temp_spe_list = spe_det_dict.get(k)
        temp_list = res_dict.get(k)
        new_temp_list = remove_nonspe_obj_for_single_img(temp_list, temp_spe_list, spe, thresh)
        res_dict[k] = new_temp_list
    return res_dict

def generate_empty_dict(xml_dict, GT_dict):
    ks, res_dict, iou_dict = [], {}, {}
    for k in GT_dict.keys():
        temp_res = GT_dict.get(k)
        for t_r in temp_res:
            temp_key = list(t_r.keys())[0]
            if not ks or temp_key not in ks:
                ks.append(temp_key)
    for k in xml_dict.keys():
        temp_res = xml_dict.get(k)
        for t_r in temp_res:
            temp_key = list(t_r.keys())[0]
            if not ks or temp_key not in ks:
                ks.append(temp_key)
    ks.append('NA')
    for k1 in ks:
        temp_dict = {}
        for k2 in ks:
            temp_dict[k2] = 0
        res_dict[k1] = temp_dict
        if k1 != 'NA':
            iou_dict[k1] = []
    return res_dict, iou_dict

def compute_matching_for_single_img(txt_list, xml_list, iou_thresh):
    GT_num = len(xml_list)
    if not txt_list:
        detect_num = 0
    else:
        detect_num = len(txt_list)
    matched = [False] * GT_num
    res, iou_res, to_del_detect_inds, to_del_GT_inds = [], [], [], []
    for i in range(detect_num):
        curr_detect_box = list(txt_list[i].values())[0][1:]
        curr_detect_spe = list(txt_list[i].keys())[0]
        max_ind, max_iou = 0, -1
        for j in range(GT_num):
            if not matched[j]:
                temp_GT_box = list(xml_list[j].values())[0]
                temp_iou = compute_iou(curr_detect_box, temp_GT_box)
                if temp_iou > max_iou:
                    max_iou = temp_iou
                    max_ind = j
        if max_iou >= iou_thresh:
            matched[max_ind] = True
            curr_GT_spe = list(xml_list[max_ind].keys())[0]
            res.append((curr_detect_spe, curr_GT_spe))
            to_del_detect_inds.append(i)
            to_del_GT_inds.append(max_ind)
            iou_res.append(max_iou)
    for i0 in range(detect_num):
        if i0 not in to_del_detect_inds:
            curr_spe = list(txt_list[i0].keys())[0]
            res.append((curr_spe, 'NA'))
            iou_res.append(0)
    for j0 in range(GT_num):
        if not matched[j0]:
            curr_spe = list(xml_list[j0].keys())[0]
            res.append(('NA', curr_spe))
            iou_res.append(0)
    return res, iou_res

def add_single_img_res(curr_dict_0, iou_dict_0, img_res, iou_res):
    curr_dict, iou_dict = copy.deepcopy(curr_dict_0), copy.deepcopy(iou_dict_0)
    for num in range(len(img_res)):
        i_r = img_res[num]
        spe_1, spe_2 = i_r[0], i_r[1]
        sub_dict_to_update = curr_dict.get(spe_1)

        sub_dict_to_update[spe_2] = sub_dict_to_update.get(spe_2) + 1

        curr_dict[spe_1] = sub_dict_to_update
        if spe_2 != 'NA':
            sub_iou_to_update = iou_dict.get(spe_2)
            sub_iou_to_update.append(iou_res[num])
            iou_dict[spe_2] = sub_iou_to_update
        else:
            if spe_1 != 'NA':
                sub_iou_to_update = iou_dict.get(spe_1)
                sub_iou_to_update.append(iou_res[num])
                iou_dict[spe_1] = sub_iou_to_update
    return curr_dict, iou_dict

def compute_pos_neg_res(res_dict):
    pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT = {}, {}, {}, {}
    for k in res_dict.keys():
        if k not in pos_det_pos_GT:
            pos_det_pos_GT[k] = 0
            pos_det_neg_GT[k] = 0
            neg_det_pos_GT[k] = 0
            neg_det_neg_GT[k] = 0
        temp_dict = res_dict.get(k)
        for t_k in temp_dict.keys():
            if t_k not in pos_det_pos_GT:
                pos_det_pos_GT[t_k] = 0
                pos_det_neg_GT[t_k] = 0
                neg_det_pos_GT[t_k] = 0
                neg_det_neg_GT[t_k] = 0
            if t_k == k:
                if str(t_k) != 'NA':
                    pos_det_pos_GT[t_k] = pos_det_pos_GT.get(t_k) + temp_dict.get(t_k)
                else:
                    neg_det_neg_GT[t_k] = neg_det_neg_GT.get(t_k) + temp_dict.get(t_k)
            else:
                pos_det_neg_GT[k] = pos_det_neg_GT.get(k) + temp_dict.get(t_k)
                neg_det_pos_GT[t_k] = neg_det_pos_GT.get(t_k) + temp_dict.get(t_k)
    return pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT
    
def compute_mean_iou(iou_dict):
    res_dict = {}
    for k in iou_dict.keys():
        res_dict[k] = np.mean(iou_dict.get(k))
    return res_dict

def compute_praf(pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT):
    precision_res, recall_res, accuracy_res, f1_res = {}, {}, {}, {}
    for k in pos_det_pos_GT.keys():
        pdpg, pdng, ndpg, ndng = pos_det_pos_GT.get(k), pos_det_neg_GT.get(k), neg_det_pos_GT.get(k), neg_det_neg_GT.get(k)
        prc_val = 0 if pdpg + pdng == 0 else pdpg / (pdpg + pdng)
        rec_val = 0 if pdpg + ndpg == 0 else pdpg / (pdpg + ndpg)
        acc_val = 0 if pdpg + ndng + pdng + ndpg == 0 else (pdpg + ndng) / (pdpg + ndng + pdng + ndpg)
        
        if prc_val + rec_val != 0:
            f1_val = 2 * prc_val * rec_val / (prc_val + rec_val)
        else:
            f1_val = 0
        
        precision_res[k] = prc_val
        recall_res[k] = rec_val
        accuracy_res[k] = acc_val
        f1_res[k] = f1_val
    return precision_res, recall_res, accuracy_res, f1_res

def build_empty_list_dict(spe_keys):
    res_dict = {}
    for k in spe_keys:
        res_dict[k] = []
    return res_dict

def add_new_epoch_res(curr_dict, add_on_dict):
    for k in curr_dict.keys():
        temp_list = curr_dict.get(k)
        temp_list.append(add_on_dict.get(k))
        curr_dict[k] = temp_list
    return curr_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--boxfolder', dest='boxfolder',
                          help='the folder that saves detection boxes',
                          default='./Accuracy_Test/Orig_boxes/Imgs/', type=str)
    parser.add_argument('--speboxfolder', dest='speboxfolder',
                          help='the folder that saves spe detection boxes',
                          default='./Accuracy_Test/spe_boxes/img/50/', type=str)
    parser.add_argument('--gtfolder', dest='gtfolder',
                          help='the folder that saves ',
                          default="Accuracy_Test/GT_Marking/")
    parser.add_argument('--picklesavedir', dest='picklesavedir',
                          help='directory to save pickle files',
                          default="Accuracy_Test/Orig/pickle/")
    parser.add_argument('--csvsavedir', dest='csvsavedir',
                          help='directory to save csv files',
                          default="./Accuracy_Test/Orig/tables/")
    parser.add_argument('--startepoch', dest='startepoch',
                        help='start epoch to load network',
                        default=1, type=int)
    parser.add_argument('--endepoch', dest='endepoch',
                        help='end epoch to load network',
                        default=50, type=int)
    parser.add_argument('--iouthresh', dest='iouthresh',
                        help='threshold for iou values',
                        default=0.60, type=float)
    parser.add_argument('--pthresh', dest='pthresh',
                        help='threshold for p values',
                        default=0.60, type=float)
    parser.add_argument('--boundthresh', dest='boundthresh',
                        help='threshold for bound',
                        default=0.96, type=float)
    parser.add_argument('--overlapthresh', dest='overlapthresh',
                        help='threshold for overlap',
                        default=0.75, type=float)
    parser.add_argument('--spe', dest='spe',
                          help='which spe to compute',
                          default='deer', type=str)
    parser.add_argument('--specoverthresh', dest='specoverthresh',
                        help='threshold for the percent of v/nv boxes located within spe boxes',
                        default=0.8, type=float)
    parser.add_argument('--spepthresh', dest='spepthresh',
                        help='spe threshold for p values',
                        default=0.5, type=float)
    parser.add_argument('--speboundthresh', dest='speboundthresh',
                        help='spe threshold for bound',
                        default=0.96, type=float)
    parser.add_argument('--speoverlapthresh', dest='speoverlapthresh',
                        help='spe threshold for overlap',
                        default=0.75, type=float)
    args = parser.parse_args()
    return args
            
if __name__ == "__main__":
    args = parse_args()
    spe = args.spe
    start_epoch, end_epoch = args.startepoch, args.endepoch
    spe_box_folder = args.speboxfolder
    all_txt_folder, xml_folder, pickle_folder, csv_folder = args.boxfolder, args.gtfolder, args.picklesavedir, args.csvsavedir
    iou_thresh, p_thresh, bound_thresh, overlap_thresh = args.iouthresh, args.pthresh, args.boundthresh, args.overlapthresh
    spe_cover_thresh, spe_p_thresh, spe_bound_thresh, spe_overlap_thresh = \
        args.specoverthresh, args.spepthresh, args.speboundthresh, args.speoverlapthresh
    make_dir_recursively(pickle_folder)
    make_dir_recursively(csv_folder)
    
    pickle_folder_w_setting = pickle_folder + '%02d_%02d_%02d_%02d'%(p_thresh * 100, bound_thresh * 100, overlap_thresh * 100, iou_thresh * 100) + '/'
    csv_folder_w_setting = csv_folder + '%02d_%02d_%02d_%02d'%(p_thresh * 100, bound_thresh * 100, overlap_thresh * 100, iou_thresh * 100) + '/'
    
    make_dir_recursively(pickle_folder_w_setting)
    make_dir_recursively(csv_folder_w_setting)
    
    final_pickle_folder, final_csv_folder = pickle_folder_w_setting + 'all/', csv_folder_w_setting + 'all/'
    
    make_dir_recursively(final_pickle_folder)
    make_dir_recursively(final_csv_folder)
    
    spe_det_dict = construct_detect_dict(spe_box_folder)
    spe_det_dict_wo_bound = remove_bound_objs(copy.deepcopy(spe_det_dict), spe_bound_thresh)
    spe_det_dict_wo_lowpval = remove_low_pvals(copy.deepcopy(spe_det_dict), spe_p_thresh)
    spe_det_dict_wo_overlap = remove_overlapped_objs(copy.deepcopy(spe_det_dict), spe_overlap_thresh)
        
    spe_det_dict_wo_all = remove_bound_objs(copy.deepcopy(spe_det_dict_wo_lowpval), spe_bound_thresh)
    spe_det_dict_wo_all = remove_overlapped_objs(spe_det_dict_wo_all, spe_overlap_thresh)
    
    for epoch in range(1, end_epoch + 1):
        epoch_pickle_folder, epoch_csv_folder = pickle_folder_w_setting + '%d/'%epoch, csv_folder_w_setting + '%d/'%epoch
        
        make_dir_recursively(epoch_pickle_folder)
        make_dir_recursively(epoch_csv_folder)
        
        txt_folder = all_txt_folder + str(epoch) + '/'
        
        det_dict = construct_detect_dict(txt_folder)
        GT_dict, size_dict = construct_GT_dict(xml_folder)
        det_dict_wo_bound = remove_bound_objs(copy.deepcopy(det_dict), bound_thresh)
        det_dict_wo_lowpval = remove_low_pvals(copy.deepcopy(det_dict), p_thresh)
        det_dict_wo_overlap = remove_overlapped_objs(copy.deepcopy(det_dict), overlap_thresh)

        det_dict_wo_spe = remove_nonspe_objs(copy.deepcopy(det_dict), spe_det_dict_wo_all, spe, spe_cover_thresh)
        
        det_dict_wo_all = remove_bound_objs(copy.deepcopy(det_dict_wo_lowpval), bound_thresh)
        det_dict_wo_all = remove_overlapped_objs(det_dict_wo_all, overlap_thresh)
        det_dict_wo_all = remove_nonspe_objs(det_dict_wo_all, spe_det_dict_wo_all, spe, spe_cover_thresh)
        
        count_dict, iou_dict = generate_empty_dict(det_dict, GT_dict)
        count_dict_wo_bound, iou_dict_wo_bound = generate_empty_dict(det_dict, GT_dict)
        count_dict_wo_lowpval, iou_dict_wo_lowpval = generate_empty_dict(det_dict, GT_dict)
        count_dict_wo_overlap, iou_dict_wo_overlap = generate_empty_dict(det_dict, GT_dict)
        count_dict_wo_spe, iou_dict_wo_spe = generate_empty_dict(det_dict, GT_dict)
        count_dict_wo_all, iou_dict_wo_all = generate_empty_dict(det_dict, GT_dict)
        
        for k in det_dict.keys():
            xml_list = GT_dict.get(k)
            txt_list = det_dict.get(k)
            txt_list_wo_bound = det_dict_wo_bound.get(k)
            txt_list_wo_lowpval = det_dict_wo_lowpval.get(k)
            txt_list_wo_overlap = det_dict_wo_overlap.get(k)
            
            txt_list_wo_spe = det_dict_wo_spe.get(k)
            
            txt_list_wo_all = det_dict_wo_all.get(k)
            
            tuple_res, iou_res = compute_matching_for_single_img(txt_list, xml_list, iou_thresh)
            count_dict, iou_dict = add_single_img_res(count_dict, iou_dict, tuple_res, iou_res)
            
            tuple_res_wo_bound, iou_res_wo_bound = compute_matching_for_single_img(txt_list_wo_bound, xml_list, iou_thresh)
            count_dict_wo_bound, iou_dict_wo_bound = add_single_img_res(count_dict_wo_bound, iou_dict_wo_bound, tuple_res_wo_bound, iou_res_wo_bound)
            
            tuple_res_wo_lowpval, iou_res_wo_lowpval = compute_matching_for_single_img(txt_list_wo_lowpval, xml_list, iou_thresh)
            count_dict_wo_lowpval, iou_dict_wo_lowpval = add_single_img_res(count_dict_wo_lowpval, iou_dict_wo_lowpval, tuple_res_wo_lowpval, iou_res_wo_lowpval)
            
            tuple_res_wo_overlap, iou_res_wo_overlap = compute_matching_for_single_img(txt_list_wo_overlap, xml_list, iou_thresh)
            count_dict_wo_overlap, iou_dict_wo_overlap = add_single_img_res(count_dict_wo_overlap, iou_dict_wo_overlap, tuple_res_wo_overlap, iou_res_wo_overlap)
            
            tuple_res_wo_spe, iou_res_wo_spe = compute_matching_for_single_img(txt_list_wo_spe, xml_list, iou_thresh)
            count_dict_wo_spe, iou_dict_wo_spe = add_single_img_res(count_dict_wo_spe, iou_dict_wo_spe, tuple_res_wo_spe, iou_res_wo_spe)
            
            tuple_res_wo_all, iou_res_wo_all = compute_matching_for_single_img(txt_list_wo_all, xml_list, iou_thresh)
            count_dict_wo_all, iou_dict_wo_all = add_single_img_res(count_dict_wo_all, iou_dict_wo_all, tuple_res_wo_all, iou_res_wo_all)
            
        pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT = compute_pos_neg_res(count_dict)
        mean_ious = compute_mean_iou(iou_dict)
        
        pos_det_pos_GT_wo_bound, pos_det_neg_GT_wo_bound, neg_det_pos_GT_wo_bound, neg_det_neg_GT_wo_bound = compute_pos_neg_res(count_dict_wo_bound)
        mean_ious_wo_bound = compute_mean_iou(iou_dict_wo_bound)
        
        pos_det_pos_GT_wo_lowpval, pos_det_neg_GT_wo_lowpval, neg_det_pos_GT_wo_lowpval, neg_det_neg_GT_wo_lowpval = compute_pos_neg_res(count_dict_wo_lowpval)
        mean_ious_wo_lowpval = compute_mean_iou(iou_dict_wo_lowpval)
        
        pos_det_pos_GT_wo_overlap, pos_det_neg_GT_wo_overlap, neg_det_pos_GT_wo_overlap, neg_det_neg_GT_wo_overlap = compute_pos_neg_res(count_dict_wo_overlap)
        mean_ious_wo_overlap = compute_mean_iou(iou_dict_wo_overlap)
        
        pos_det_pos_GT_wo_spe, pos_det_neg_GT_wo_spe, neg_det_pos_GT_wo_spe, neg_det_neg_GT_wo_spe = compute_pos_neg_res(count_dict_wo_spe)
        mean_ious_wo_spe = compute_mean_iou(iou_dict_wo_spe)
        
        pos_det_pos_GT_wo_all, pos_det_neg_GT_wo_all, neg_det_pos_GT_wo_all, neg_det_neg_GT_wo_all = compute_pos_neg_res(count_dict_wo_all)
        mean_ious_wo_all = compute_mean_iou(iou_dict_wo_all)
        
        precision_res, recall_res, accuracy_res, f1_res = \
                compute_praf(pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT)
                
        precision_res_wo_bound, recall_res_wo_bound, accuracy_res_wo_bound, f1_res_wo_bound = \
                compute_praf(pos_det_pos_GT_wo_bound, pos_det_neg_GT_wo_bound, neg_det_pos_GT_wo_bound, neg_det_neg_GT_wo_bound)
                
        precision_res_wo_lowpval, recall_res_wo_lowpval, accuracy_res_wo_lowpval, f1_res_wo_lowpval= \
                compute_praf(pos_det_pos_GT_wo_lowpval, pos_det_neg_GT_wo_lowpval, neg_det_pos_GT_wo_lowpval, neg_det_neg_GT_wo_lowpval)
                
        precision_res_wo_overlap, recall_res_wo_overlap, accuracy_res_wo_overlap, f1_res_wo_overlap = \
                compute_praf(pos_det_pos_GT_wo_overlap, pos_det_neg_GT_wo_overlap, neg_det_pos_GT_wo_overlap, neg_det_neg_GT_wo_overlap)
        
        precision_res_wo_spe, recall_res_wo_spe, accuracy_res_wo_spe, f1_res_wo_spe = \
                compute_praf(pos_det_pos_GT_wo_spe, pos_det_neg_GT_wo_spe, neg_det_pos_GT_wo_spe, neg_det_neg_GT_wo_spe)
                
        precision_res_wo_all, recall_res_wo_all, accuracy_res_wo_all, f1_res_wo_all = \
                compute_praf(pos_det_pos_GT_wo_all, pos_det_neg_GT_wo_all, neg_det_pos_GT_wo_all, neg_det_neg_GT_wo_all)
        
        if epoch == start_epoch:
            spe_keys = pos_det_pos_GT.keys()
            iou_keys = mean_ious.keys()
            pdpg_dict, pdng_dict, ndpg_dict, ndng_dict, prc_dict, rec_dict, acc_dict, f1_dict, miou_dict = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
                
            pdpg_dict_wo_bound, pdng_dict_wo_bound, ndpg_dict_wo_bound, ndng_dict_wo_bound, prc_dict_wo_bound, rec_dict_wo_bound, acc_dict_wo_bound, f1_dict_wo_bound, miou_dict_wo_bound = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
            
            pdpg_dict_wo_lowpval, pdng_dict_wo_lowpval, ndpg_dict_wo_lowpval, ndng_dict_wo_lowpval, prc_dict_wo_lowpval, rec_dict_wo_lowpval, acc_dict_wo_lowpval, f1_dict_wo_lowpval, miou_dict_wo_lowpval = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
            
            pdpg_dict_wo_overlap, pdng_dict_wo_overlap, ndpg_dict_wo_overlap, ndng_dict_wo_overlap, prc_dict_wo_overlap, rec_dict_wo_overlap, acc_dict_wo_overlap, f1_dict_wo_overlap, miou_dict_wo_overlap = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
                
            pdpg_dict_wo_spe, pdng_dict_wo_spe, ndpg_dict_wo_spe, ndng_dict_wo_spe, prc_dict_wo_spe, rec_dict_wo_spe, acc_dict_wo_spe, f1_dict_wo_spe, miou_dict_wo_spe = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
            
            pdpg_dict_wo_all, pdng_dict_wo_all, ndpg_dict_wo_all, ndng_dict_wo_all, prc_dict_wo_all, rec_dict_wo_all, acc_dict_wo_all, f1_dict_wo_all, miou_dict_wo_all = \
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys), build_empty_list_dict(spe_keys),\
                build_empty_list_dict(iou_keys)
        
        save_pickle_file(epoch_pickle_folder, 'count_dict.pkl', count_dict)
        save_pickle_file(epoch_pickle_folder, 'count_dict_wo_bound.pkl', count_dict_wo_bound)
        save_pickle_file(epoch_pickle_folder, 'count_dict_wo_lowpval.pkl', count_dict_wo_lowpval)
        save_pickle_file(epoch_pickle_folder, 'count_dict_wo_overlap.pkl', count_dict_wo_overlap)
        save_pickle_file(epoch_pickle_folder, 'count_dict_wo_spe.pkl', count_dict_wo_spe)
        save_pickle_file(epoch_pickle_folder, 'count_dict_wo_all.pkl', count_dict_wo_all)
        
        save_two_layer_dict(count_dict, epoch_csv_folder, 'count.csv')
        save_two_layer_dict(count_dict_wo_bound, epoch_csv_folder, 'count_wo_bound.csv')
        save_two_layer_dict(count_dict_wo_lowpval, epoch_csv_folder, 'count_wo_lowpval.csv')
        save_two_layer_dict(count_dict_wo_overlap, epoch_csv_folder, 'count_wo_overlap.csv')
        save_two_layer_dict(count_dict_wo_spe, epoch_csv_folder, 'count_wo_spe.csv')
        save_two_layer_dict(count_dict_wo_all, epoch_csv_folder, 'count_wo_all.csv')
        
        save_pickle_file(epoch_pickle_folder, 'iou_dict.pkl', iou_dict)
        save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_bound.pkl', iou_dict_wo_bound)
        save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_lowpval.pkl', iou_dict_wo_lowpval)
        save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_overlap.pkl', iou_dict_wo_overlap)
        save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_spe.pkl', iou_dict_wo_spe)
        save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_all.pkl', iou_dict_wo_all)
        
        pdpg_dict, pdng_dict, ndpg_dict, ndng_dict, prc_dict, rec_dict, acc_dict, f1_dict, miou_dict = \
            add_new_epoch_res(pdpg_dict, pos_det_pos_GT), add_new_epoch_res(pdng_dict, pos_det_neg_GT), \
            add_new_epoch_res(ndpg_dict, neg_det_pos_GT), add_new_epoch_res(ndng_dict, pos_det_neg_GT), \
            add_new_epoch_res(prc_dict, precision_res), add_new_epoch_res(rec_dict, recall_res),\
            add_new_epoch_res(acc_dict, accuracy_res), add_new_epoch_res(f1_dict, f1_res),\
            add_new_epoch_res(miou_dict, mean_ious)
            
        pdpg_dict_wo_bound, pdng_dict_wo_bound, ndpg_dict_wo_bound, ndng_dict_wo_bound, prc_dict_wo_bound, rec_dict_wo_bound, acc_dict_wo_bound, f1_dict_wo_bound, miou_dict_wo_bound = \
            add_new_epoch_res(pdpg_dict_wo_bound, pos_det_pos_GT_wo_bound), add_new_epoch_res(pdng_dict_wo_bound, pos_det_neg_GT_wo_bound), \
            add_new_epoch_res(ndpg_dict_wo_bound, neg_det_pos_GT_wo_bound), add_new_epoch_res(ndng_dict_wo_bound, pos_det_neg_GT_wo_bound), \
            add_new_epoch_res(prc_dict_wo_bound, precision_res_wo_bound), add_new_epoch_res(rec_dict_wo_bound, recall_res_wo_bound),\
            add_new_epoch_res(acc_dict_wo_bound, accuracy_res_wo_bound), add_new_epoch_res(f1_dict_wo_bound, f1_res_wo_bound),\
            add_new_epoch_res(miou_dict_wo_bound, mean_ious_wo_bound)
            
        pdpg_dict_wo_lowpval, pdng_dict_wo_lowpval, ndpg_dict_wo_lowpval, ndng_dict_wo_lowpval, prc_dict_wo_lowpval, rec_dict_wo_lowpval, acc_dict_wo_lowpval, f1_dict_wo_lowpval, miou_dict_wo_lowpval = \
            add_new_epoch_res(pdpg_dict_wo_lowpval, pos_det_pos_GT_wo_lowpval), add_new_epoch_res(pdng_dict_wo_lowpval, pos_det_neg_GT_wo_lowpval), \
            add_new_epoch_res(ndpg_dict_wo_lowpval, neg_det_pos_GT_wo_lowpval), add_new_epoch_res(ndng_dict_wo_lowpval, pos_det_neg_GT_wo_lowpval), \
            add_new_epoch_res(prc_dict_wo_lowpval, precision_res_wo_lowpval), add_new_epoch_res(rec_dict_wo_lowpval, recall_res_wo_lowpval),\
            add_new_epoch_res(acc_dict_wo_lowpval, accuracy_res_wo_lowpval), add_new_epoch_res(f1_dict_wo_lowpval, f1_res_wo_lowpval),\
            add_new_epoch_res(miou_dict_wo_lowpval, mean_ious_wo_lowpval)
            
        pdpg_dict_wo_overlap, pdng_dict_wo_overlap, ndpg_dict_wo_overlap, ndng_dict_wo_overlap, prc_dict_wo_overlap, rec_dict_wo_overlap, acc_dict_wo_overlap, f1_dict_wo_overlap, miou_dict_wo_overlap = \
            add_new_epoch_res(pdpg_dict_wo_overlap, pos_det_pos_GT_wo_overlap), add_new_epoch_res(pdng_dict_wo_overlap, pos_det_neg_GT_wo_overlap), \
            add_new_epoch_res(ndpg_dict_wo_overlap, neg_det_pos_GT_wo_overlap), add_new_epoch_res(ndng_dict_wo_overlap, pos_det_neg_GT_wo_overlap), \
            add_new_epoch_res(prc_dict_wo_overlap, precision_res_wo_overlap), add_new_epoch_res(rec_dict_wo_overlap, recall_res_wo_overlap),\
            add_new_epoch_res(acc_dict_wo_overlap, accuracy_res_wo_overlap), add_new_epoch_res(f1_dict_wo_overlap, f1_res_wo_overlap),\
            add_new_epoch_res(miou_dict_wo_overlap, mean_ious_wo_overlap)
        
        pdpg_dict_wo_spe, pdng_dict_wo_spe, ndpg_dict_wo_spe, ndng_dict_wo_spe, prc_dict_wo_spe, rec_dict_wo_spe, acc_dict_wo_spe, f1_dict_wo_spe, miou_dict_wo_spe = \
            add_new_epoch_res(pdpg_dict_wo_spe, pos_det_pos_GT_wo_spe), add_new_epoch_res(pdng_dict_wo_spe, pos_det_neg_GT_wo_spe), \
            add_new_epoch_res(ndpg_dict_wo_spe, neg_det_pos_GT_wo_spe), add_new_epoch_res(ndng_dict_wo_spe, pos_det_neg_GT_wo_spe), \
            add_new_epoch_res(prc_dict_wo_spe, precision_res_wo_spe), add_new_epoch_res(rec_dict_wo_spe, recall_res_wo_spe),\
            add_new_epoch_res(acc_dict_wo_spe, accuracy_res_wo_spe), add_new_epoch_res(f1_dict_wo_spe, f1_res_wo_spe),\
            add_new_epoch_res(miou_dict_wo_spe, mean_ious_wo_spe)
        
        pdpg_dict_wo_all, pdng_dict_wo_all, ndpg_dict_wo_all, ndng_dict_wo_all, prc_dict_wo_all, rec_dict_wo_all, acc_dict_wo_all, f1_dict_wo_all, miou_dict_wo_all = \
            add_new_epoch_res(pdpg_dict_wo_all, pos_det_pos_GT_wo_all), add_new_epoch_res(pdng_dict_wo_all, pos_det_neg_GT_wo_all), \
            add_new_epoch_res(ndpg_dict_wo_all, neg_det_pos_GT_wo_all), add_new_epoch_res(ndng_dict_wo_all, pos_det_neg_GT_wo_all), \
            add_new_epoch_res(prc_dict_wo_all, precision_res_wo_all), add_new_epoch_res(rec_dict_wo_all, recall_res_wo_all),\
            add_new_epoch_res(acc_dict_wo_all, accuracy_res_wo_all), add_new_epoch_res(f1_dict_wo_all, f1_res_wo_all),\
            add_new_epoch_res(miou_dict_wo_all, mean_ious_wo_all)
    
    save_one_layer_dict(pdpg_dict, final_csv_folder, 'pdpg.csv')
    save_one_layer_dict(pdng_dict, final_csv_folder, 'pdng.csv')
    save_one_layer_dict(ndpg_dict, final_csv_folder, 'ndpg.csv')
    save_one_layer_dict(ndng_dict, final_csv_folder, 'ndng.csv')
    save_one_layer_dict(prc_dict, final_csv_folder, 'prc.csv')
    save_one_layer_dict(rec_dict, final_csv_folder, 'rec.csv')
    save_one_layer_dict(acc_dict, final_csv_folder, 'acc.csv')
    save_one_layer_dict(f1_dict, final_csv_folder, 'f1.csv')
    save_one_layer_dict(miou_dict, final_csv_folder, 'miou.csv')
    
    save_one_layer_dict(pdpg_dict_wo_bound, final_csv_folder, 'pdpg_wo_bound.csv')
    save_one_layer_dict(pdng_dict_wo_bound, final_csv_folder, 'pdng_wo_bound.csv')
    save_one_layer_dict(ndpg_dict_wo_bound, final_csv_folder, 'ndpg_wo_bound.csv')
    save_one_layer_dict(ndng_dict_wo_bound, final_csv_folder, 'ndng_wo_bound.csv')
    save_one_layer_dict(prc_dict_wo_bound, final_csv_folder, 'prc_wo_bound.csv')
    save_one_layer_dict(rec_dict_wo_bound, final_csv_folder, 'rec_wo_bound.csv')
    save_one_layer_dict(acc_dict_wo_bound, final_csv_folder, 'acc_wo_bound.csv')
    save_one_layer_dict(f1_dict_wo_bound, final_csv_folder, 'f1_wo_bound.csv')
    save_one_layer_dict(miou_dict_wo_bound, final_csv_folder, 'miou_wo_bound.csv')
    
    save_one_layer_dict(pdpg_dict_wo_lowpval, final_csv_folder, 'pdpg_wo_lowpval.csv')
    save_one_layer_dict(pdng_dict_wo_lowpval, final_csv_folder, 'pdng_wo_lowpval.csv')
    save_one_layer_dict(ndpg_dict_wo_lowpval, final_csv_folder, 'ndpg_wo_lowpval.csv')
    save_one_layer_dict(ndng_dict_wo_lowpval, final_csv_folder, 'ndng_wo_lowpval.csv')
    save_one_layer_dict(prc_dict_wo_lowpval, final_csv_folder, 'prc_wo_lowpval.csv')
    save_one_layer_dict(rec_dict_wo_lowpval, final_csv_folder, 'rec_wo_lowpval.csv')
    save_one_layer_dict(acc_dict_wo_lowpval, final_csv_folder, 'acc_wo_lowpval.csv')
    save_one_layer_dict(f1_dict_wo_lowpval, final_csv_folder, 'f1_wo_lowpval.csv')
    save_one_layer_dict(miou_dict_wo_lowpval, final_csv_folder, 'miou_wo_lowpval.csv')
    
    save_one_layer_dict(pdpg_dict_wo_overlap, final_csv_folder, 'pdpg_wo_overlap.csv')
    save_one_layer_dict(pdng_dict_wo_overlap, final_csv_folder, 'pdng_wo_overlap.csv')
    save_one_layer_dict(ndpg_dict_wo_overlap, final_csv_folder, 'ndpg_wo_overlap.csv')
    save_one_layer_dict(ndng_dict_wo_overlap, final_csv_folder, 'ndng_wo_overlap.csv')
    save_one_layer_dict(prc_dict_wo_overlap, final_csv_folder, 'prc_wo_overlap.csv')
    save_one_layer_dict(rec_dict_wo_overlap, final_csv_folder, 'rec_wo_overlap.csv')
    save_one_layer_dict(acc_dict_wo_overlap, final_csv_folder, 'acc_wo_overlap.csv')
    save_one_layer_dict(f1_dict_wo_overlap, final_csv_folder, 'f1_wo_overlap.csv')
    save_one_layer_dict(miou_dict_wo_overlap, final_csv_folder, 'miou_wo_overlap.csv')

    save_one_layer_dict(pdpg_dict_wo_spe, final_csv_folder, 'pdpg_wo_spe.csv')
    save_one_layer_dict(pdng_dict_wo_spe, final_csv_folder, 'pdng_wo_spe.csv')
    save_one_layer_dict(ndpg_dict_wo_spe, final_csv_folder, 'ndpg_wo_spe.csv')
    save_one_layer_dict(ndng_dict_wo_spe, final_csv_folder, 'ndng_wo_spe.csv')
    save_one_layer_dict(prc_dict_wo_spe, final_csv_folder, 'prc_wo_spe.csv')
    save_one_layer_dict(rec_dict_wo_spe, final_csv_folder, 'rec_wo_spe.csv')
    save_one_layer_dict(acc_dict_wo_spe, final_csv_folder, 'acc_wo_spe.csv')
    save_one_layer_dict(f1_dict_wo_spe, final_csv_folder, 'f1_wo_spe.csv')
    save_one_layer_dict(miou_dict_wo_spe, final_csv_folder, 'miou_wo_spe.csv')
    
    save_one_layer_dict(pdpg_dict_wo_all, final_csv_folder, 'pdpg_wo_all.csv')
    save_one_layer_dict(pdng_dict_wo_all, final_csv_folder, 'pdng_wo_all.csv')
    save_one_layer_dict(ndpg_dict_wo_all, final_csv_folder, 'ndpg_wo_all.csv')
    save_one_layer_dict(ndng_dict_wo_all, final_csv_folder, 'ndng_wo_all.csv')
    save_one_layer_dict(prc_dict_wo_all, final_csv_folder, 'prc_wo_all.csv')
    save_one_layer_dict(rec_dict_wo_all, final_csv_folder, 'rec_wo_all.csv')
    save_one_layer_dict(acc_dict_wo_all, final_csv_folder, 'acc_wo_all.csv')
    save_one_layer_dict(f1_dict_wo_all, final_csv_folder, 'f1_wo_all.csv')
    save_one_layer_dict(miou_dict_wo_all, final_csv_folder, 'miou_wo_all.csv')

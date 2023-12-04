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
import NewTestBetweenDetectionAndGT

def make_dir_recursively(input_path):
    # Create the directory recursively if it doesn't exist
    os.makedirs(os.path.dirname(input_path), exist_ok=True)

def create_group_dicts(group_folder, img_suffix_1, img_other_suffix, replace = True):
    sub_folders = [f.path for f in os.scandir(group_folder) if f.is_dir()]
    res = {}
    for sf in sub_folders:
        curr_cat = sf.split('/')[-1]
        if replace:
            for io_suf in img_other_suffix:
                imgs_2_replace = NewTestBetweenDetectionAndGT.get_files_with_suffix(sf + '/', io_suf)
                for img in imgs_2_replace:
                    os.rename(sf + '/' + img, sf + '/' + img.split('.')[0] + '.' + img_suffix_1)
        imgs = NewTestBetweenDetectionAndGT.get_files_with_suffix(sf + '/', img_suffix_1)
        res[curr_cat] = imgs
    return res

def construct_detect_dict_w_selected_imgs(file_folder, img_suffix, img_list):
    all_files = NewTestBetweenDetectionAndGT.get_files_with_suffix(file_folder, 'txt')
    res_dict = {}
    for f in all_files:
        file_front = f.split('.')[0]
        if file_front + '.' + img_suffix in img_list:
            temp_res = NewTestBetweenDetectionAndGT.read_vals_within_txt_file(file_folder, f, 5)
            res_dict[file_front] = temp_res
    return res_dict

def construct_GT_dict_w_selected_imgs(file_folder, img_suffix, img_list):
    all_files = NewTestBetweenDetectionAndGT.get_files_with_suffix(file_folder, 'xml')
    res_dict, size_dict = {}, {}
    for f in all_files:
        file_front = f.split('.')[0]
        if file_front + '.' + img_suffix in img_list:
            img_width, img_height, temp_res = NewTestBetweenDetectionAndGT.read_vals_within_xml_file(file_folder, f)
            res_dict[file_front] = temp_res
            size_dict[file_front] = [img_width, img_height]
    return res_dict, size_dict        

def group_parse_args():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--parentfolder', dest='parentfolder',
                          help='the parent folder of all subdirectories',
                          default='./Accuracy_Test/Old/', type=str)
    parser.add_argument('--boxfolder', dest='boxfolder',
                          help='the folder that saves detection boxes',
                          default='boxes/Imgs/', type=str)
    parser.add_argument('--speboxfolder', dest='speboxfolder',
                          help='the folder that saves spe detection boxes',
                          default='./Accuracy_Test/spe_boxes/img/50/', type=str)
    parser.add_argument('--gtfolder', dest='gtfolder',
                          help='the folder that saves ground true marking',
                          default="GT_Marking/")
    parser.add_argument('--classfolder', dest = 'classfolder', 
                          help = 'the folder that saves classification results', type=str,
                          default = '/mnt/ssd1/faster-rcnn/photo_for_category/')
    parser.add_argument('--picklesavedir', dest='picklesavedir',
                          help='directory to save pickle files',
                          default="pickle/")
    parser.add_argument('--csvsavedir', dest='csvsavedir',
                          help='directory to save csv files',
                          default="tables/")
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
    args = group_parse_args()
    spe = args.spe
    spe_box_folder = args.speboxfolder
    start_epoch, end_epoch = args.startepoch, args.endepoch
    parent_folder, class_folder = args.parentfolder, args.classfolder
    iou_thresh, bound_thresh, overlap_thresh, p_thresh = args.iouthresh, args.boundthresh, args.overlapthresh, args.pthresh
    
    all_txt_folder, xml_folder =  parent_folder + 'Orig/' + args.boxfolder, parent_folder + 'Orig/' + args.gtfolder
    spe_cover_thresh, spe_p_thresh, spe_bound_thresh, spe_overlap_thresh = \
        args.specoverthresh, args.spepthresh, args.speboundthresh, args.speoverlapthresh
    
    cat_dict = create_group_dicts(class_folder, 'jpg', ['jpg', 'jpeg', 'png'])
    
    for c_k in cat_dict.keys():
        
        classes = str(c_k)
        temp_img_list = cat_dict.get(c_k)
        pickle_folder, csv_folder = parent_folder + '/' + classes + '/' + args.picklesavedir, \
                parent_folder + '/' + classes + '/' + args.csvsavedir
        
        make_dir_recursively(pickle_folder)
        make_dir_recursively(csv_folder)
        
        pickle_folder_w_setting = pickle_folder + '%02d_%02d_%02d_%02d'%(p_thresh * 100, bound_thresh * 100, overlap_thresh * 100, iou_thresh * 100) + '/'
        csv_folder_w_setting = csv_folder + '%02d_%02d_%02d_%02d'%(p_thresh * 100, bound_thresh * 100, overlap_thresh * 100, iou_thresh * 100) + '/'
        
        make_dir_recursively(pickle_folder_w_setting)
        make_dir_recursively(csv_folder_w_setting)
        final_pickle_folder, final_csv_folder = pickle_folder_w_setting + 'all/', csv_folder_w_setting + 'all/'
        
        make_dir_recursively(final_pickle_folder)
        make_dir_recursively(final_csv_folder)
        
        spe_det_dict = NewTestBetweenDetectionAndGT.construct_detect_dict(spe_box_folder)
        spe_det_dict_wo_bound = NewTestBetweenDetectionAndGT.remove_bound_objs(copy.deepcopy(spe_det_dict), spe_bound_thresh)
        spe_det_dict_wo_lowpval = NewTestBetweenDetectionAndGT.remove_low_pvals(copy.deepcopy(spe_det_dict), spe_p_thresh)
        spe_det_dict_wo_overlap = NewTestBetweenDetectionAndGT.remove_overlapped_objs(copy.deepcopy(spe_det_dict), spe_overlap_thresh)
            
        spe_det_dict_wo_all = NewTestBetweenDetectionAndGT.remove_bound_objs(copy.deepcopy(spe_det_dict_wo_lowpval), spe_bound_thresh)
        spe_det_dict_wo_all = NewTestBetweenDetectionAndGT.remove_overlapped_objs(spe_det_dict_wo_all, spe_overlap_thresh)
        
        for epoch in range(start_epoch, end_epoch + 1):
            epoch_pickle_folder, epoch_csv_folder = pickle_folder_w_setting + '%d/'%epoch, csv_folder_w_setting + '%d/'%epoch
            
            make_dir_recursively(epoch_pickle_folder)
            make_dir_recursively(epoch_csv_folder)
            
            txt_folder = all_txt_folder + str(epoch) + '/'
            
            det_dict = construct_detect_dict_w_selected_imgs(txt_folder, 'jpg', temp_img_list)
            GT_dict, size_dict = construct_GT_dict_w_selected_imgs(xml_folder, 'jpg', temp_img_list)
            det_dict_wo_bound = NewTestBetweenDetectionAndGT.remove_bound_objs(copy.deepcopy(det_dict), bound_thresh)
            det_dict_wo_lowpval = NewTestBetweenDetectionAndGT.remove_low_pvals(copy.deepcopy(det_dict), p_thresh)
            det_dict_wo_overlap = NewTestBetweenDetectionAndGT.remove_overlapped_objs(copy.deepcopy(det_dict), overlap_thresh)
            
            det_dict_wo_spe = NewTestBetweenDetectionAndGT.remove_nonspe_objs(copy.deepcopy(det_dict), spe_det_dict_wo_all, spe, spe_cover_thresh)
            
            det_dict_wo_all = NewTestBetweenDetectionAndGT.remove_bound_objs(copy.deepcopy(det_dict_wo_lowpval), bound_thresh)
            det_dict_wo_all = NewTestBetweenDetectionAndGT.remove_overlapped_objs(det_dict_wo_all, overlap_thresh)
            
            det_dict_wo_all = NewTestBetweenDetectionAndGT.remove_nonspe_objs(det_dict_wo_all, spe_det_dict_wo_all, spe, spe_cover_thresh)
            
            count_dict, iou_dict = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            count_dict_wo_bound, iou_dict_wo_bound = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            count_dict_wo_lowpval, iou_dict_wo_lowpval = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            count_dict_wo_overlap, iou_dict_wo_overlap = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            count_dict_wo_spe, iou_dict_wo_spe = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            count_dict_wo_all, iou_dict_wo_all = NewTestBetweenDetectionAndGT.generate_empty_dict(det_dict, GT_dict)
            

            for k in det_dict.keys():
                xml_list = GT_dict.get(k)
                txt_list = det_dict.get(k)
                txt_list_wo_bound = det_dict_wo_bound.get(k)
                txt_list_wo_lowpval = det_dict_wo_lowpval.get(k)
                txt_list_wo_overlap = det_dict_wo_overlap.get(k)
                txt_list_wo_spe = det_dict_wo_spe.get(k)
                txt_list_wo_all = det_dict_wo_all.get(k)

                tuple_res, iou_res = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list, xml_list, iou_thresh)
                count_dict, iou_dict = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict, iou_dict, tuple_res, iou_res)

                tuple_res_wo_bound, iou_res_wo_bound = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list_wo_bound, xml_list, iou_thresh)
                count_dict_wo_bound, iou_dict_wo_bound = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict_wo_bound, iou_dict_wo_bound, tuple_res_wo_bound, iou_res_wo_bound)

                tuple_res_wo_lowpval, iou_res_wo_lowpval = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list_wo_lowpval, xml_list, iou_thresh)
                count_dict_wo_lowpval, iou_dict_wo_lowpval = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict_wo_lowpval, iou_dict_wo_lowpval, tuple_res_wo_lowpval, iou_res_wo_lowpval)

                tuple_res_wo_overlap, iou_res_wo_overlap = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list_wo_overlap, xml_list, iou_thresh)
                count_dict_wo_overlap, iou_dict_wo_overlap = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict_wo_overlap, iou_dict_wo_overlap, tuple_res_wo_overlap, iou_res_wo_overlap)

                tuple_res_wo_spe, iou_res_wo_spe = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list_wo_spe, xml_list, iou_thresh)
                count_dict_wo_spe, iou_dict_wo_spe = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict_wo_spe, iou_dict_wo_spe, tuple_res_wo_spe, iou_res_wo_spe)

                tuple_res_wo_all, iou_res_wo_all = NewTestBetweenDetectionAndGT.compute_matching_for_single_img(txt_list_wo_all, xml_list, iou_thresh)
                count_dict_wo_all, iou_dict_wo_all = NewTestBetweenDetectionAndGT.add_single_img_res(count_dict_wo_all, iou_dict_wo_all, tuple_res_wo_all, iou_res_wo_all)

            pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT = NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict)
            mean_ious = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict)

            pos_det_pos_GT_wo_bound, pos_det_neg_GT_wo_bound, neg_det_pos_GT_wo_bound, neg_det_neg_GT_wo_bound = \
                NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict_wo_bound)
            mean_ious_wo_bound = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict_wo_bound)

            pos_det_pos_GT_wo_lowpval, pos_det_neg_GT_wo_lowpval, neg_det_pos_GT_wo_lowpval, neg_det_neg_GT_wo_lowpval = \
                NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict_wo_lowpval)
            mean_ious_wo_lowpval = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict_wo_lowpval)

            pos_det_pos_GT_wo_overlap, pos_det_neg_GT_wo_overlap, neg_det_pos_GT_wo_overlap, neg_det_neg_GT_wo_overlap = \
                NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict_wo_overlap)
            mean_ious_wo_overlap = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict_wo_overlap)
            
            pos_det_pos_GT_wo_spe, pos_det_neg_GT_wo_spe, neg_det_pos_GT_wo_spe, neg_det_neg_GT_wo_spe = \
                NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict_wo_spe)
            mean_ious_wo_spe = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict_wo_spe)

            pos_det_pos_GT_wo_all, pos_det_neg_GT_wo_all, neg_det_pos_GT_wo_all, neg_det_neg_GT_wo_all = NewTestBetweenDetectionAndGT.compute_pos_neg_res(count_dict_wo_all)
            mean_ious_wo_all = NewTestBetweenDetectionAndGT.compute_mean_iou(iou_dict_wo_all)

            

            precision_res, recall_res, accuracy_res, f1_res = \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT, pos_det_neg_GT, neg_det_pos_GT, neg_det_neg_GT)

            precision_res_wo_bound, recall_res_wo_bound, accuracy_res_wo_bound, f1_res_wo_bound = \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT_wo_bound, pos_det_neg_GT_wo_bound, neg_det_pos_GT_wo_bound, neg_det_neg_GT_wo_bound)

            precision_res_wo_lowpval, recall_res_wo_lowpval, accuracy_res_wo_lowpval, f1_res_wo_lowpval= \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT_wo_lowpval, pos_det_neg_GT_wo_lowpval, neg_det_pos_GT_wo_lowpval, neg_det_neg_GT_wo_lowpval)

            precision_res_wo_overlap, recall_res_wo_overlap, accuracy_res_wo_overlap, f1_res_wo_overlap = \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT_wo_overlap, pos_det_neg_GT_wo_overlap, neg_det_pos_GT_wo_overlap, neg_det_neg_GT_wo_overlap)
            
            precision_res_wo_spe, recall_res_wo_spe, accuracy_res_wo_spe, f1_res_wo_spe = \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT_wo_spe, pos_det_neg_GT_wo_spe, neg_det_pos_GT_wo_spe, neg_det_neg_GT_wo_spe)

            
            precision_res_wo_all, recall_res_wo_all, accuracy_res_wo_all, f1_res_wo_all = \
                    NewTestBetweenDetectionAndGT.compute_praf(pos_det_pos_GT_wo_all, pos_det_neg_GT_wo_all, neg_det_pos_GT_wo_all, neg_det_neg_GT_wo_all)

            if epoch == start_epoch:
                spe_keys = pos_det_pos_GT.keys()
                iou_keys = mean_ious.keys()
                pdpg_dict, pdng_dict, ndpg_dict, ndng_dict, prc_dict, rec_dict, acc_dict, f1_dict, miou_dict = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)

                pdpg_dict_wo_bound, pdng_dict_wo_bound, ndpg_dict_wo_bound, ndng_dict_wo_bound, prc_dict_wo_bound, rec_dict_wo_bound, acc_dict_wo_bound, f1_dict_wo_bound, miou_dict_wo_bound = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)

                pdpg_dict_wo_lowpval, pdng_dict_wo_lowpval, ndpg_dict_wo_lowpval, ndng_dict_wo_lowpval, prc_dict_wo_lowpval, rec_dict_wo_lowpval, acc_dict_wo_lowpval, f1_dict_wo_lowpval, miou_dict_wo_lowpval = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)

                pdpg_dict_wo_overlap, pdng_dict_wo_overlap, ndpg_dict_wo_overlap, ndng_dict_wo_overlap, prc_dict_wo_overlap, rec_dict_wo_overlap, acc_dict_wo_overlap, f1_dict_wo_overlap, miou_dict_wo_overlap = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)
                
                pdpg_dict_wo_spe, pdng_dict_wo_spe, ndpg_dict_wo_spe, ndng_dict_wo_spe, prc_dict_wo_spe, rec_dict_wo_spe, acc_dict_wo_spe, f1_dict_wo_spe, miou_dict_wo_spe = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)
                
                pdpg_dict_wo_all, pdng_dict_wo_all, ndpg_dict_wo_all, ndng_dict_wo_all, prc_dict_wo_all, rec_dict_wo_all, acc_dict_wo_all, f1_dict_wo_all, miou_dict_wo_all = \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), \
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys), NewTestBetweenDetectionAndGT.build_empty_list_dict(spe_keys),\
                    NewTestBetweenDetectionAndGT.build_empty_list_dict(iou_keys)

            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict.pkl', count_dict)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict_wo_bound.pkl', count_dict_wo_bound)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict_wo_lowpval.pkl', count_dict_wo_lowpval)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict_wo_overlap.pkl', count_dict_wo_overlap)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict_wo_spe.pkl', count_dict_wo_spe)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'count_dict_wo_all.pkl', count_dict_wo_all)

            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict, epoch_csv_folder, 'count.csv')
            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict_wo_bound, epoch_csv_folder, 'count_wo_bound.csv')
            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict_wo_lowpval, epoch_csv_folder, 'count_wo_lowpval.csv')
            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict_wo_overlap, epoch_csv_folder, 'count_wo_overlap.csv')
            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict_wo_spe, epoch_csv_folder, 'count_wo_spe.csv')
            NewTestBetweenDetectionAndGT.save_two_layer_dict(count_dict_wo_all, epoch_csv_folder, 'count_wo_all.csv')

            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict.pkl', iou_dict)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_bound.pkl', iou_dict_wo_bound)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_lowpval.pkl', iou_dict_wo_lowpval)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_overlap.pkl', iou_dict_wo_overlap)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_spe.pkl', iou_dict_wo_spe)
            NewTestBetweenDetectionAndGT.save_pickle_file(epoch_pickle_folder, 'iou_dict_wo_all.pkl', iou_dict_wo_all)

            pdpg_dict, pdng_dict, ndpg_dict, ndng_dict, prc_dict, rec_dict, acc_dict, f1_dict, miou_dict = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict, pos_det_pos_GT), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict, pos_det_neg_GT), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict, neg_det_pos_GT), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict, pos_det_neg_GT), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict, precision_res), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict, recall_res),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict, accuracy_res), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict, f1_res),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict, mean_ious)

            pdpg_dict_wo_bound, pdng_dict_wo_bound, ndpg_dict_wo_bound, ndng_dict_wo_bound, prc_dict_wo_bound, rec_dict_wo_bound, acc_dict_wo_bound, f1_dict_wo_bound, miou_dict_wo_bound = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict_wo_bound, pos_det_pos_GT_wo_bound), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict_wo_bound, pos_det_neg_GT_wo_bound), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict_wo_bound, neg_det_pos_GT_wo_bound), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict_wo_bound, pos_det_neg_GT_wo_bound), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict_wo_bound, precision_res_wo_bound), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict_wo_bound, recall_res_wo_bound),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict_wo_bound, accuracy_res_wo_bound), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict_wo_bound, f1_res_wo_bound),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict_wo_bound, mean_ious_wo_bound)

            pdpg_dict_wo_lowpval, pdng_dict_wo_lowpval, ndpg_dict_wo_lowpval, ndng_dict_wo_lowpval, prc_dict_wo_lowpval, rec_dict_wo_lowpval, acc_dict_wo_lowpval, f1_dict_wo_lowpval, miou_dict_wo_lowpval = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict_wo_lowpval, pos_det_pos_GT_wo_lowpval), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict_wo_lowpval, pos_det_neg_GT_wo_lowpval), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict_wo_lowpval, neg_det_pos_GT_wo_lowpval), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict_wo_lowpval, pos_det_neg_GT_wo_lowpval), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict_wo_lowpval, precision_res_wo_lowpval), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict_wo_lowpval, recall_res_wo_lowpval),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict_wo_lowpval, accuracy_res_wo_lowpval), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict_wo_lowpval, f1_res_wo_lowpval),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict_wo_lowpval, mean_ious_wo_lowpval)

            pdpg_dict_wo_overlap, pdng_dict_wo_overlap, ndpg_dict_wo_overlap, ndng_dict_wo_overlap, prc_dict_wo_overlap, rec_dict_wo_overlap, acc_dict_wo_overlap, f1_dict_wo_overlap, miou_dict_wo_overlap = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict_wo_overlap, pos_det_pos_GT_wo_overlap), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict_wo_overlap, pos_det_neg_GT_wo_overlap), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict_wo_overlap, neg_det_pos_GT_wo_overlap), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict_wo_overlap, pos_det_neg_GT_wo_overlap), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict_wo_overlap, precision_res_wo_overlap), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict_wo_overlap, recall_res_wo_overlap),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict_wo_overlap, accuracy_res_wo_overlap), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict_wo_overlap, f1_res_wo_overlap),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict_wo_overlap, mean_ious_wo_overlap)
                
            pdpg_dict_wo_spe, pdng_dict_wo_spe, ndpg_dict_wo_spe, ndng_dict_wo_spe, prc_dict_wo_spe, rec_dict_wo_spe, acc_dict_wo_spe, f1_dict_wo_spe, miou_dict_wo_spe = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict_wo_spe, pos_det_pos_GT_wo_spe), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict_wo_spe, pos_det_neg_GT_wo_spe), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict_wo_spe, neg_det_pos_GT_wo_spe), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict_wo_spe, pos_det_neg_GT_wo_spe), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict_wo_spe, precision_res_wo_spe), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict_wo_spe, recall_res_wo_spe),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict_wo_spe, accuracy_res_wo_spe), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict_wo_spe, f1_res_wo_spe),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict_wo_spe, mean_ious_wo_spe)

            pdpg_dict_wo_all, pdng_dict_wo_all, ndpg_dict_wo_all, ndng_dict_wo_all, prc_dict_wo_all, rec_dict_wo_all, acc_dict_wo_all, f1_dict_wo_all, miou_dict_wo_all = \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(pdpg_dict_wo_all, pos_det_pos_GT_wo_all), NewTestBetweenDetectionAndGT.add_new_epoch_res(pdng_dict_wo_all, pos_det_neg_GT_wo_all), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(ndpg_dict_wo_all, neg_det_pos_GT_wo_all), NewTestBetweenDetectionAndGT.add_new_epoch_res(ndng_dict_wo_all, pos_det_neg_GT_wo_all), \
                NewTestBetweenDetectionAndGT.add_new_epoch_res(prc_dict_wo_all, precision_res_wo_all), NewTestBetweenDetectionAndGT.add_new_epoch_res(rec_dict_wo_all, recall_res_wo_all),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(acc_dict_wo_all, accuracy_res_wo_all), NewTestBetweenDetectionAndGT.add_new_epoch_res(f1_dict_wo_all, f1_res_wo_all),\
                NewTestBetweenDetectionAndGT.add_new_epoch_res(miou_dict_wo_all, mean_ious_wo_all)

        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict, final_csv_folder, 'pdpg.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict, final_csv_folder, 'pdng.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict, final_csv_folder, 'ndpg.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict, final_csv_folder, 'ndng.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict, final_csv_folder, 'prc.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict, final_csv_folder, 'rec.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict, final_csv_folder, 'acc.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict, final_csv_folder, 'f1.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict, final_csv_folder, 'miou.csv')

        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict_wo_bound, final_csv_folder, 'pdpg_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict_wo_bound, final_csv_folder, 'pdng_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict_wo_bound, final_csv_folder, 'ndpg_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict_wo_bound, final_csv_folder, 'ndng_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict_wo_bound, final_csv_folder, 'prc_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict_wo_bound, final_csv_folder, 'rec_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict_wo_bound, final_csv_folder, 'acc_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict_wo_bound, final_csv_folder, 'f1_wo_bound.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict_wo_bound, final_csv_folder, 'miou_wo_bound.csv')

        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict_wo_lowpval, final_csv_folder, 'pdpg_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict_wo_lowpval, final_csv_folder, 'pdng_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict_wo_lowpval, final_csv_folder, 'ndpg_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict_wo_lowpval, final_csv_folder, 'ndng_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict_wo_lowpval, final_csv_folder, 'prc_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict_wo_lowpval, final_csv_folder, 'rec_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict_wo_lowpval, final_csv_folder, 'acc_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict_wo_lowpval, final_csv_folder, 'f1_wo_lowpval.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict_wo_lowpval, final_csv_folder, 'miou_wo_lowpval.csv')

        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict_wo_overlap, final_csv_folder, 'pdpg_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict_wo_overlap, final_csv_folder, 'pdng_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict_wo_overlap, final_csv_folder, 'ndpg_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict_wo_overlap, final_csv_folder, 'ndng_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict_wo_overlap, final_csv_folder, 'prc_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict_wo_overlap, final_csv_folder, 'rec_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict_wo_overlap, final_csv_folder, 'acc_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict_wo_overlap, final_csv_folder, 'f1_wo_overlap.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict_wo_overlap, final_csv_folder, 'miou_wo_overlap.csv')
        
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict_wo_spe, final_csv_folder, 'pdpg_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict_wo_spe, final_csv_folder, 'pdng_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict_wo_spe, final_csv_folder, 'ndpg_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict_wo_spe, final_csv_folder, 'ndng_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict_wo_spe, final_csv_folder, 'prc_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict_wo_spe, final_csv_folder, 'rec_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict_wo_spe, final_csv_folder, 'acc_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict_wo_spe, final_csv_folder, 'f1_wo_spe.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict_wo_spe, final_csv_folder, 'miou_wo_spe.csv')
        
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdpg_dict_wo_all, final_csv_folder, 'pdpg_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(pdng_dict_wo_all, final_csv_folder, 'pdng_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndpg_dict_wo_all, final_csv_folder, 'ndpg_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(ndng_dict_wo_all, final_csv_folder, 'ndng_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(prc_dict_wo_all, final_csv_folder, 'prc_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(rec_dict_wo_all, final_csv_folder, 'rec_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(acc_dict_wo_all, final_csv_folder, 'acc_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(f1_dict_wo_all, final_csv_folder, 'f1_wo_all.csv')
        NewTestBetweenDetectionAndGT.save_one_layer_dict(miou_dict_wo_all, final_csv_folder, 'miou_wo_all.csv')

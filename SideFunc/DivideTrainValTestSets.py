#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:52:31 2021

@author: weiwang
"""
import os
import argparse
import pandas as pd
import random
from shutil import copyfile
import json as JS
import xml.etree.ElementTree as ET

def ConverJSON2XML(json_folder, json_file, xml_folder, xml_file):
    with open(json_folder + json_file, "r") as j:
        data = JS.load(json_file)

def CheckIfDirExist(paths):
    os.makedirs(paths, exist_ok = True)

def getListWithCorrectMarker(mark_folder, mark_suffix):
    all_imgs = []
    for file in os.listdir(mark_folder):
        if (file[-3:] == mark_suffix):
            all_imgs.append(file.split('.')[0])
    all_imgs = pd.DataFrame(all_imgs, columns=["file"])
    return all_imgs

def DivideImgsIntoDifferentSets(all_imgs):
    img_names = list(all_imgs.file)
    img_num = len(img_names)
    
    the_list = list(range(0, img_num))
    
    random.shuffle(the_list)
    trainval = []
    train = []
    val = []
    test = []
    
    
    for i in range(img_num):
        if i <= int(img_num * 0.7):
            trainval.append(img_names[i])
            if i <= int(img_num * 0.35):
                train.append(img_names[i])
            else:
                val.append(img_names[i])
        else:
            test.append(img_names[i])
    return trainval, train, val, test

def copyFileInListToANewFolder(tar_list, old_folder, new_folder):
    CheckIfDirExist(new_folder)
    for i in tar_list:
        old_path = old_folder + i
        new_path = new_folder + i
        copyfile(old_path, new_path)

def copyXMLFileInListToANewFolder(tar_list, old_folder, new_folder):
    CheckIfDirExist(new_folder)
    for i in tar_list:
        old_path = old_folder + i.split('.')[0] + '.xml'
        new_path = new_folder + i.split('.')[0] + '.xml'
        copyfile(old_path, new_path)
        
def copyJPGFileInListToANewFolder(tar_list, old_folder, new_folder):
    CheckIfDirExist(new_folder)
    for i in tar_list:
        old_path = old_folder + i.split('.')[0] + '.jpg'
        new_path = new_folder + i.split('.')[0] + '.jpg'
        copyfile(old_path, new_path)
        
def writeTrainValAndTestFile(cand_list, save_folder, save_name):
    CheckIfDirExist(save_folder)
    with open(save_folder + save_name, 'w') as f1:
        trainval_len = len(cand_list)
        for i in range(trainval_len):
            if i == trainval_len - 1:
                temp_img = cand_list[i]
                f1.write(temp_img.split('.')[0])
            else:
                temp_img = cand_list[i]
                f1.write(temp_img.split('.')[0])
                f1.write('\n')
        f1.close()
        

parser = argparse.ArgumentParser(description='Divide data sets to train, test and val')

parser.add_argument('--orig_mark_folder', dest='orig_mark_folder', help='original marked dataset', default='Your_path_to_annotation_file', type=str)

parser.add_argument('--orig_img_folder', dest='orig_img_folder', help='original img dataset', default='Your_path_to_original_image_file', type=str)

parser.add_argument('--save_folder', dest='save_folder', help='save dataset', default='Your_path_to_save_split_res', type=str)

parser.add_argument('--mark_suffix', dest='mark_suffix', help='suffix of marked data', default='xml', type=str)

args = parser.parse_args()

orig_mark_folder = args.orig_mark_folder

orig_img_folder = args.orig_img_folder

save_div_folder = args.save_folder + 'ImageSets/Main/'
CheckIfDirExist(save_div_folder)

save_anno_folder = args.save_folder + 'Annotations/'
CheckIfDirExist(save_anno_folder)

save_img_folder = args.save_folder + 'JPEGImages/'
CheckIfDirExist(save_img_folder)

mark_suffix = args.mark_suffix

all_imgs = getListWithCorrectMarker(orig_mark_folder, mark_suffix)

# copyXMLFileInListToANewFolder(list(all_imgs.file), orig_mark_folder, save_anno_folder)
# copyJPGFileInListToANewFolder(list(all_imgs.file), orig_img_folder, save_img_folder)

trainval, train, val, test = DivideImgsIntoDifferentSets(all_imgs)

trainval_name = 'trainval.txt'
train_name = 'train.txt'
val_name = 'val.txt'
test_name =  'test.txt'

writeTrainValAndTestFile(trainval, save_div_folder, trainval_name)
writeTrainValAndTestFile(train, save_div_folder, train_name)
writeTrainValAndTestFile(val, save_div_folder, val_name)
writeTrainValAndTestFile(test, save_div_folder, test_name)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 15:15:12 2023

@author: weiwang
"""

import os
import shutil
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Modify xml files')
    parser.add_argument('--input_file_folder', dest='input_file_folder', help = "Folder to save input test document", default="")
    parser.add_argument('--input_file_name', dest='input_file_name', help = "Name of input test document", default="test.txt")
    parser.add_argument('--input_img_folder', dest='input_img_folder', help = "Folder to save input image dataset", default="")
    parser.add_argument('--input_img_suffix', dest='input_img_suffix', help = "image suffix", default="jpg")
    parser.add_argument('--input_xml_folder', dest='input_xml_folder', help = "Folder to save input marking dataset", default="")
    parser.add_argument('--input_xml_suffix', dest='input_xml_suffix', help = "marking suffix", default="xml")
    parser.add_argument('--output_parent_folder', dest='output_parent_folder', help = "Parent folder to save output images and marking", default="")
    parser.add_argument('--output_img_folder', dest='output_img_folder', help = "Folder to save output images under output parent folder", default="Imgs/")
    parser.add_argument('--output_xml_folder', dest='output_xml_folder', help = "Folder to save output markings under output parent folder", default="GT_Marking/")

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    output_parent_folder = args.output_parent_folder
    input_file_folder = args.input_file_folder
    input_file_name = args.input_file_name
    input_img_folder = args.input_img_folder
    input_img_suffix = args.input_img_suffix
    output_img_folder = args.output_img_folder

    input_xml_folder = args.input_xml_folder
    input_xml_suffix = args.input_xml_suffix
    output_xml_folder = args.output_xml_folder
    
    #output_parent_folder = './Accuracy_Test/'
    #input_file_folder = './data/VOCdevkit2007/VOC2007/ImageSets/Main/'
    #input_file_name = 'test.txt'
    #input_img_folder = './data/VOCdevkit2007/VOC2007/JPEGImages/'
    #input_img_suffix = 'jpg'
    #output_img_folder = 'Imgs/'

    #input_xml_folder = './data/VOCdevkit2007/VOC2007/Annotations/'
    #input_xml_suffix = 'xml'
    #output_xml_folder = 'GT_Marking/'


    if not os.path.exists(output_parent_folder):
        os.mkdir(output_parent_folder)

    if not os.path.exists(output_parent_folder + output_img_folder):
        os.mkdir(output_parent_folder + output_img_folder)

    if not os.path.exists(output_parent_folder + output_xml_folder):
        os.mkdir(output_parent_folder + output_xml_folder)

    with open(input_file_folder + input_file_name, 'r') as f:
        lines = f.readlines()
        f.close()

    for l0 in lines:
        if l0:
            l = l0.split('\n')[0]
            img_name = l + '.' + input_img_suffix
            input_img_path = input_img_folder + img_name
            output_img_path = output_parent_folder + output_img_folder + img_name
            shutil.copy(input_img_path, output_img_path)

            xml_name = l + '.' + input_xml_suffix
            input_xml_path = input_xml_folder + xml_name
            output_xml_path = output_parent_folder + output_xml_folder + xml_name
            shutil.copy(input_xml_path, output_xml_path)
        
        

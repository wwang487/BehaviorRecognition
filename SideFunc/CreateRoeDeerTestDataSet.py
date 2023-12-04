import os
import argparse
import shutil
import xml.dom.minidom
import random

def getListWithCorrectMarker(folder, suffix):
    all_imgs = []
    for f in os.listdir(folder):
        if (f.split('.')[-1] == suffix):
            all_imgs.append(f)
    return all_imgs
    
def extract_strings(strings, N):
    if N > len(strings):
        raise ValueError("N cannot be greater than the length of the input list.")

    extracted_strings = random.sample(strings, N)
    return extracted_strings

def copy_roedeer_xml_data(orig_xml_file, orig_xml_path, new_xml_path, new_spe_tag, name_choice):
    dom = xml.dom.minidom.parse(orig_xml_path + orig_xml_file)
    root = dom.documentElement
    
    size = root.getElementsByTagName('size')
    temp_height = int(size[0].getElementsByTagName('height')[0].firstChild.data)
    temp_width = int(size[0].getElementsByTagName('width')[0].firstChild.data)
    objects = root.getElementsByTagName('object')
    for o in range(len(objects)):
        temp_bndbox = objects[o].getElementsByTagName('bndbox')[0]
        if name_choice:
            objects[o].getElementsByTagName('name')[0].firstChild.data = new_spe_tag
   
    folder=root.getElementsByTagName('folder')
    file = root.getElementsByTagName('filename')
    path = root.getElementsByTagName('path')

    for i in range(len(folder)): 
    	folder[i].firstChild.data = 'VOC2007'
    
    output_file = new_xml_path + orig_xml_file
    with open(output_file,'w') as fh:
        dom.writexml(fh)

def check_if_dir_exist(out_path):
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok = True)

def parse_args():
    parser = argparse.ArgumentParser(description='Modify xml files')
    parser.add_argument('--roedeer_spe_xml_path', dest='roedeer_spe_xml_path',
                      help='directory to roedeer spe xml files',
                      default= "/media/wlp/My Book/checked_roedeer_data/xml_spe/")
    parser.add_argument('--roedeer_head_xml_path', dest='roedeer_head_xml_path',
                      help='directory to roedeer head xml files',
                      default= "/media/wlp/My Book/checked_roedeer_data/xml_head/")
    parser.add_argument('--roedeer_img_path', dest='roedeer_img_path',
                      help='directory to roedeer img files',
                      default= "/media/wlp/My Book/checked_roedeer_data/img/")

    parser.add_argument('--out_path', dest='out_path',
                      help='directory to save merged dataset',
                      default= "/media/wlp/My Book/NewDataSet/test/")
    parser.add_argument('--N', dest='N',
                      help='number of files to extract',
                      default= 300)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    roedeer_spe_xml_path = args.roedeer_spe_xml_path
    roedeer_head_xml_path = args.roedeer_head_xml_path
    roedeer_img_path = args.roedeer_img_path
    
    out_path = args.out_path
    out_img_path, out_spe_path, out_head_path = out_path + 'img/', out_path + 'spe/', out_path + 'head/'
    check_if_dir_exist(out_img_path)
    check_if_dir_exist(out_spe_path)
    check_if_dir_exist(out_head_path)
    N = args.N
    img_suffix = 'jpg'
    all_roedeer_imgs = getListWithCorrectMarker(roedeer_img_path, 'jpg')
    roedeer_to_test = extract_strings(all_roedeer_imgs, N)
    for r in roedeer_to_test:
        temp_jpg = r
        temp_xml = r.split('.')[0] + '.xml'
        shutil.copy(roedeer_img_path + r, out_img_path + r)
        copy_roedeer_xml_data(temp_xml, roedeer_spe_xml_path, out_spe_path, 'deer', True)
        copy_roedeer_xml_data(temp_xml, roedeer_head_xml_path, out_head_path, '', False)
        

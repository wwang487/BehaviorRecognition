import random
import argparse
import os
import shutil
import xml.dom.minidom
import cv2

def extract_strings(strings, N):
    if N > len(strings):
        raise ValueError("N cannot be greater than the length of the input list.")

    extracted_strings = random.sample(strings, N)
    return extracted_strings
    
def getListWithCorrectMarker(folder, suffix):
    all_imgs = []
    for f in os.listdir(folder):
        if (f.split('.')[-1] == suffix):
            all_imgs.append(f)
    return all_imgs

def find_string_index(strings, search_string):
    try:
        index = strings.index(search_string)
        return index
    except ValueError:
        return -1

def create_main_file(main_folder, main_file, to_extract_orig, to_extract_new, img_suffix, save_folder, save_file):
    f = open(main_folder + main_file, 'r')
    f_lines = f.readlines()
    for i in range(len(f_lines)):
        if not f_lines[i] or f_lines[i] == '\n':
            continue
        if f_lines[i] in to_extract_orig:
            temp_ind = find_string_index(f_lines[i] + '.' + img_suffix, to_extract_orig)
            f_lines[i] = to_extract_new[temp_ind].split('.')[0]
    f.close()
    to_print = ''
    for j in range(len(f_lines)):
        to_print = to_print + f_lines[j] 
        
    with open(save_folder + save_file, 'w') as f1:
        print(to_print, file = f1)
    f1.close()

def modify_xml(old_xml_path, old_xml_file, ref_xml_path, ref_xml_file, new_xml_path, new_spe_tag, new_file_tag, new_file_suffix, name_choice):
    dom = xml.dom.minidom.parse(old_xml_path + old_xml_file)
    root = dom.documentElement
    
    dom_old = xml.dom.minidom.parse(ref_xml_path + ref_xml_file)
    root_old = dom_old.documentElement
    
    size = root.getElementsByTagName('size')
    size_old = root_old.getElementsByTagName('size')
    temp_height = int(size[0].getElementsByTagName('height')[0].firstChild.data)
    old_height = int(size_old[0].getElementsByTagName('height')[0].firstChild.data)
    temp_width = int(size[0].getElementsByTagName('width')[0].firstChild.data)
    old_width = int(size_old[0].getElementsByTagName('width')[0].firstChild.data)
    height_scaling = old_height / temp_height
    width_scaling = old_width / temp_width
    objects = root.getElementsByTagName('object')
    for o in range(len(objects)):
        temp_bndbox = objects[o].getElementsByTagName('bndbox')[0]
        temp_xmin = int(temp_bndbox.getElementsByTagName('xmin')[0].firstChild.data)
        temp_ymin = int(temp_bndbox.getElementsByTagName('ymin')[0].firstChild.data)
        temp_xmax = int(temp_bndbox.getElementsByTagName('xmax')[0].firstChild.data)
        temp_ymax = int(temp_bndbox.getElementsByTagName('ymax')[0].firstChild.data)
        if name_choice:
            objects[o].getElementsByTagName('name')[0].firstChild.data = new_spe_tag
        objects[o].getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data = \
		str(round(min(temp_xmin * width_scaling, old_width)))
        objects[o].getElementsByTagName('bndbox')[0].getElementsByTagName('ymin')[0].firstChild.data = \
		str(round(min(temp_ymin * height_scaling, old_height)))
        objects[o].getElementsByTagName('bndbox')[0].getElementsByTagName('xmax')[0].firstChild.data = \
		str(round(min(temp_xmax * width_scaling, old_width)))
        objects[o].getElementsByTagName('bndbox')[0].getElementsByTagName('ymax')[0].firstChild.data = \
		str(round(min(temp_ymax * height_scaling, old_height)))
    size[0].getElementsByTagName('height')[0].firstChild.data = str(old_height)
    size[0].getElementsByTagName('width')[0].firstChild.data = str(old_width)
    folder=root.getElementsByTagName('folder')
    file = root.getElementsByTagName('filename')
    path = root.getElementsByTagName('path')

    for i in range(len(folder)): 
    	folder[i].firstChild.data = 'VOC2007'
    for j in range(len(file)):
    	file[j].firstChild.data = '.'.join([new_file_tag, new_file_suffix])
    for k in range(len(path)):
    	temp_path = path[k].firstChild.data
    	new_path = '/'.join(temp_path.split('/')[:-1]) + '/' + '.'.join([new_file_tag, new_file_suffix])
    	path[k].firstChild.data = new_path
    output_file = new_xml_path + new_file_tag + '.xml'
    with open(output_file,'w') as fh:
        dom.writexml(fh)

def parse_args():
    parser = argparse.ArgumentParser(description='Modify xml files')
    parser.add_argument('--roedeer_xml_path', dest='roedeer_xml_path',
                      help='directory to roedeer xml_files',
                      default= "/media/wlp/My Book/checked_roedeer_data/xml_head/")
    parser.add_argument('--roedeer_img_path', dest='roedeer_img_path',
                      help='directory to roedeer img files',
                      default= "/media/wlp/My Book/checked_roedeer_data/img/")
    parser.add_argument('--orig_xml_path', dest='orig_xml_path',
                      help='directory to original xml_files',
                      default= "/media/wlp/My Book/VOC2007_head/")
    parser.add_argument('--save_path', dest='save_path',
                      help='directory to save merged dataset',
                      default= "/media/wlp/My Book/NewDataSet/head/")
    parser.add_argument('--N', dest='N',
                      help='number of files to extract',
                      default= 500)
    parser.add_argument('--name_choice', dest='name_choice',
                      help='replace the object name or not', type = bool,
                      default= False)
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = parse_args()
    roedeer_xml_path = args.roedeer_xml_path
    roedeer_img_path = args.roedeer_img_path
    
    orig_path = args.orig_xml_path
    orig_annotation_path = orig_path + 'Annotations/'
    orig_main_path = orig_path + 'ImageSets/Main/'
    orig_img_path = orig_path + 'JPEGImages/'
    name_choice = args.name_choice
    out_path = args.save_path
    img_suffix = 'jpg'
    
    if os.path.exists(out_path):
        shutil.rmtree(out_path)
        
    out_img_path = out_path + 'JPEGImages/'
    out_xml_path = out_path + 'Annotations/'
    out_main_path = out_path + 'ImageSets/Main/'
    os.makedirs(out_img_path, exist_ok = True)
    os.makedirs(out_xml_path, exist_ok = True)
    os.makedirs(out_main_path, exist_ok = True)
    
    N = args.N
    roedeer_xmls = getListWithCorrectMarker(roedeer_xml_path, 'xml')
    to_extract_roedeer = extract_strings(roedeer_xmls, N)
    
    orig_xmls = getListWithCorrectMarker(orig_annotation_path, 'xml')
    to_extract_orig = extract_strings(orig_xmls, N)
    
    for i in range(N):
        temp_roedeer = to_extract_roedeer[i]
        temp_orig = to_extract_orig[i]
        temp_tag = temp_orig.split('.')[0]
        temp_suffix = img_suffix
        modify_xml(roedeer_xml_path, temp_roedeer, orig_annotation_path, temp_orig, out_xml_path, 'deer', temp_tag, temp_suffix, name_choice)
        temp_img_file = temp_roedeer.split('.')[0] + '.jpg'
        temp_img = cv2.imread(roedeer_img_path + temp_img_file)
        ref_img_file = temp_orig.split('.')[0] + '.jpg'
        ref_img = cv2.imread(orig_img_path + ref_img_file)
        ref_height, ref_width = ref_img.shape[0], ref_img.shape[1]
        resized_img = cv2.resize(temp_img, (ref_width, ref_height), interpolation = cv2.INTER_AREA)
        cv2.imwrite(out_img_path + temp_tag + '.' + img_suffix, resized_img)
            
    for i in orig_xmls:
        if i not in to_extract_orig:
            shutil.copy(orig_annotation_path + i, out_xml_path + i)
            img_file = i.split('.')[0] + '.jpg'
            shutil.copy(orig_img_path + img_file, out_img_path + img_file)
    
    create_main_file(orig_main_path, 'train.txt', to_extract_orig, to_extract_roedeer, 'jpg', out_main_path, 'train.txt')
    create_main_file(orig_main_path, 'trainval.txt', to_extract_orig, to_extract_roedeer, 'jpg', out_main_path, 'trainval.txt')
    create_main_file(orig_main_path, 'val.txt', to_extract_orig, to_extract_roedeer, 'jpg', out_main_path, 'val.txt')
    create_main_file(orig_main_path, 'test.txt', to_extract_orig, to_extract_roedeer, 'jpg', out_main_path, 'test.txt')
    

import os
import copy
import xml.dom.minidom
from xml.dom.minidom import Document
import cv2
import shutil
from xml.etree.ElementTree import ElementTree,Element
import argparse

def read_xml(in_path):
    tree = ElementTree()
    tree.parse(in_path)
    return tree
 
def write_xml(tree, out_path):
    tree.write(out_path, encoding="utf-8",xml_declaration=True)
    
 
def if_match(node, kv_map):
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True
 
#---------------search -----
 
def find_nodes(tree, path):

    return tree.findall(path)
 
 
def get_node_by_keyvalue(nodelist, kv_map):
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes

def get_node_by_textvalue(nodelist, spe):
    result_nodes = []
    for node in nodelist:
        if node.text == spe:
            result_nodes.append(node)
    return result_nodes
 
#---------------change -----
 
def change_node_properties(nodelist, kv_map, is_delete=False):
    for node in nodelist:
        for key in kv_map:
            if is_delete: 
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))
            
def change_node_text(nodelist, text, is_add=False, is_delete=False):
    for node in nodelist:
        if is_add:
            node.text += text
        elif is_delete:
            node.text = ""
        else:
            node.text = text
            
def create_node(tag, property_map, content):
    element = Element(tag, property_map)
    element.text = content
    return element
        
def add_child_node(node, element):
    node.appendChild(element)
        
def del_node_by_tagkeyvalue(nodelist, tag, kv_map):
    for parent_node in nodelist:
        children = parent_node.getchildren()
        for child in children:
            if child.tag == tag and if_match(child, kv_map):
                parent_node.remove(child)

def del_node_by_textvalue(nodelist, text):
    for parent_node in nodelist:
        if parent_node.text == text:
            children = parent_node.getchildren()
            for child in children:
                parent_node.remove(child)

def getListWithCorrectMarker(folder, suffix):
    all_imgs = []
    for file in os.listdir(folder):
        if (file[-3:] == suffix):
            all_imgs.append(file)
    return all_imgs

def getOverlappedFileList(file_list_1, file_list_2):
    set1 = set(file_list_1)
    set2 = set(file_list_2)
    iset = set1.intersection(set2)
    return list(iset)

def DetectCandSpeFiles(folder1, list1, spe_to_select):
    res = []
    for l in list1:
        root = xml.dom.minidom.parse(folder1 + l)
        temp_object = root.getElementsByTagName('object')
        
        try:
            spe_name = temp_object[0].getElementsByTagName('name')[0].firstChild.data
            if spe_name == spe_to_select:
                res.append(l)
        except:
            print(folder1 + l)
            continue
    return res
        # temp_object[0].getElementsByTagName('bndbox')[0].getElementsByTagName('xmin')[0].firstChild.data =   

def CheckIfDirExist(path):
    if os.path.exists(f'{path}'):
        pass
    else:
        os.mkdir(f'{path}')

def CopyDeerImgsIntoNewFolder(old_folder, new_folder, deer_list):
    CheckIfDirExist(new_folder)
    for dl in deer_list:
        new_img_name = dl.split('.')[0] + '.' + 'jpg'
        temp_img = cv2.imread(old_folder + new_img_name)
        cv2.imwrite(new_folder + new_img_name, temp_img)

# def ExtractNewBoundaryInfo(xml_file):
#     get_node_by_keyvalue(find_nodes(tree, "object/name"), {}) 
    
def ReadAllData(single_box, p_val, spe_list):
    res = []
    file_name = single_box
    f = open(file_name)
    # print(b)
    l = f.readline()
    while l:
        l = l.replace(' ', '')
        if l.find(',') != -1:
            for spe in spe_list:
                new_l = l.replace(spe, ' ' + spe)
            all_l_boxes = new_l.split(' ')
            for al in all_l_boxes:
                if al:
                    if al.find(',') != -1:                        
                        temp_list = al.split(',')
                        print(temp_list)
                        temp_spe = temp_list[0]
                        temp_pval = temp_list[1]
                        temp_xlt = temp_list[2]
                        temp_ylt = temp_list[3]
                        temp_xrb = temp_list[4]
                        temp_yrb = temp_list[5]
                        if float(temp_pval) > p_val:
                            res.append([temp_spe, temp_pval, temp_xlt, temp_ylt, temp_xrb, temp_yrb])
                                    
        l = f.readline()
    f.close()
    return res

def ChangeDeerXMLBasedOnVNVMarking(xml_1, xml_2):
    tree_1 = read_xml(xml_1)
    root_1 = tree_1.getroot()
    tree_2 = read_xml(xml_2)
    root_2 = tree_2.getroot()
    res_tree = copy.deepcopy(tree_1)
    for child in root.findall('object'):
        pass

def ChangeDeerXMLForASingleFile(xml_file, single_box, spe_list, p_val, spe_to_del):
    res_list = ReadAllData(single_box, p_val, spe_list)
    tree = read_xml(xml_file)
    root = tree.getroot()
    doc = xml.dom.minidom.Document() 
    new_root = doc.createElement('annotation') 
    for child in root.findall('object'):
        name = child.find('name').text
        if name == spe_to_del:
            root.remove(child)
            
    nodeFolder = doc.createElement('folder')
    nodeFilename = doc.createElement('filename')
    nodePath = doc.createElement('path')
    nodeSource = doc.createElement('source')
    nodeDatabase = doc.createElement('database')
    nodeSize = doc.createElement('size')
    nodeWidth = doc.createElement('width')
    nodeHeight = doc.createElement('height')    
    nodeDepth = doc.createElement('depth')
    nodeSegmented = doc.createElement('segmented')
    
    nodeFolder.appendChild(doc.createTextNode(root.find('folder').text))
    nodeFilename.appendChild(doc.createTextNode(root.find('filename').text))
    nodePath.appendChild(doc.createTextNode(root.find('path').text))  
    nodeDatabase.appendChild(doc.createTextNode(root.find('source/database').text))
    nodeWidth.appendChild(doc.createTextNode(root.find('size/width').text))
    nodeHeight.appendChild(doc.createTextNode(root.find('size/height').text))
    nodeDepth.appendChild(doc.createTextNode(root.find('size/depth').text))
    nodeSegmented.appendChild(doc.createTextNode(root.find('segmented').text))
    
    new_root.appendChild(nodeFolder)
    new_root.appendChild(nodeFilename)
    new_root.appendChild(nodePath)
    
    nodeSource.appendChild(nodeDatabase)
    new_root.appendChild(nodeSource)
    
    nodeSize.appendChild(nodeWidth)
    nodeSize.appendChild(nodeHeight)
    nodeSize.appendChild(nodeDepth)
    
    new_root.appendChild(nodeSize)
    new_root.appendChild(nodeSegmented)
    
    
    for child in root.findall('object'):
        try:
            new_root.appendChild(child)
        except:
            continue
    
    if res_list:
        for rl in res_list:
            temp_spe, temp_pval, temp_xlt, temp_ylt, temp_xrb, temp_yrb = rl[0], rl[1], rl[2], rl[3], rl[4], rl[5]
            nodeObject = doc.createElement('object')
            nodeName = doc.createElement('name')
            nodePose = doc.createElement('pose')
            nodeTruncated = doc.createElement('truncated')
            nodeDifficult = doc.createElement('difficult')
            nodeBndbox = doc.createElement('bndbox')
            nodeXmin = doc.createElement('xmin')
            nodeYmin = doc.createElement('ymin')
            nodeXmax = doc.createElement('xmax')
            nodeYmax = doc.createElement('ymax')
            nodeName.appendChild(doc.createTextNode(temp_spe.lower()))
            nodePose.appendChild(doc.createTextNode('Unspecified'))
            nodeTruncated.appendChild(doc.createTextNode('0'))
            nodeDifficult.appendChild(doc.createTextNode('0'))
            nodeXmin.appendChild(doc.createTextNode(temp_xlt.split('.')[0]))
            nodeYmin.appendChild(doc.createTextNode(temp_ylt.split('.')[0]))
            nodeXmax.appendChild(doc.createTextNode(temp_xrb.split('.')[0]))
            nodeYmax.appendChild(doc.createTextNode(temp_yrb.split('.')[0]))
            
            nodeObject.appendChild(nodeName)
            nodeObject.appendChild(nodePose)
            nodeObject.appendChild(nodeTruncated)
            nodeObject.appendChild(nodeDifficult)
            
            nodeBndbox.appendChild(nodeXmin)
            nodeBndbox.appendChild(nodeYmin)
            nodeBndbox.appendChild(nodeXmax)
            nodeBndbox.appendChild(nodeYmax)
            
            nodeObject.appendChild(nodeBndbox)
            new_root.appendChild(nodeObject)
            
    doc.appendChild(new_root)

    return doc

def MirrorImageMark(xml_file, save_path, save_name):    
    
    tree = read_xml(xml_file)
    root = tree.getroot()
    img_width = int(root.find('size/width').text)
    img_height = int(root.find('size/height').text)
    

    doc = xml.dom.minidom.Document() 
    new_root = doc.createElement('annotation') 
            
    nodeFolder = doc.createElement('folder')
    nodeFilename = doc.createElement('filename')
    nodePath = doc.createElement('path')
    nodeSource = doc.createElement('source')
    nodeDatabase = doc.createElement('database')
    nodeSize = doc.createElement('size')
    nodeWidth = doc.createElement('width')
    nodeHeight = doc.createElement('height')    
    nodeDepth = doc.createElement('depth')
    nodeSegmented = doc.createElement('segmented')
    
    nodeFolder.appendChild(doc.createTextNode(root.find('folder').text))
    nodeFilename.appendChild(doc.createTextNode(root.find('filename').text.split('.')[0] + '_M' + '.' + root.find('filename').text.split('.')[1]))
    nodePath.appendChild(doc.createTextNode(root.find('path').text))  
    nodeDatabase.appendChild(doc.createTextNode(root.find('source/database').text))
    nodeWidth.appendChild(doc.createTextNode(root.find('size/width').text))
    nodeHeight.appendChild(doc.createTextNode(root.find('size/height').text))
    nodeDepth.appendChild(doc.createTextNode(root.find('size/depth').text))
    nodeSegmented.appendChild(doc.createTextNode(root.find('segmented').text))
    
    new_root.appendChild(nodeFolder)
    new_root.appendChild(nodeFilename)
    new_root.appendChild(nodePath)
    
    nodeSource.appendChild(nodeDatabase)
    new_root.appendChild(nodeSource)
    
    nodeSize.appendChild(nodeWidth)
    nodeSize.appendChild(nodeHeight)
    nodeSize.appendChild(nodeDepth)
    
    new_root.appendChild(nodeSize)
    new_root.appendChild(nodeSegmented)
    
    
    for child in root.findall('object'):
        bndbox = child.find('bndbox')
        spe = child.find('name').text
        trun = child.find('truncated').text
        diff = child.find('difficult').text
        
        xmin = int(bndbox[0].text)
        ymin = int(bndbox[1].text)
        xmax = int(bndbox[2].text)
        ymax = int(bndbox[3].text)
        new_xmin = img_width - xmin - 1
        new_ymin = ymin
        new_xmax = img_width - xmax - 1
        new_ymax = ymax
        
        nodeObject = doc.createElement('object')
        nodeName = doc.createElement('name')
        nodePose = doc.createElement('pose')
        nodeTruncated = doc.createElement('truncated')
        nodeDifficult = doc.createElement('difficult')
        nodeBndbox = doc.createElement('bndbox')
        nodeXmin = doc.createElement('xmin')
        nodeYmin = doc.createElement('ymin')
        nodeXmax = doc.createElement('xmax')
        nodeYmax = doc.createElement('ymax')
        nodeName.appendChild(doc.createTextNode(spe.lower()))
        nodePose.appendChild(doc.createTextNode('Unspecified'))
        nodeTruncated.appendChild(doc.createTextNode(trun))
        nodeDifficult.appendChild(doc.createTextNode(diff))
        nodeXmin.appendChild(doc.createTextNode(str(new_xmin)))
        nodeYmin.appendChild(doc.createTextNode(str(new_ymin)))
        nodeXmax.appendChild(doc.createTextNode(str(new_xmax)))
        nodeYmax.appendChild(doc.createTextNode(str(new_ymax)))
            
        nodeObject.appendChild(nodeName)
        nodeObject.appendChild(nodePose)
        nodeObject.appendChild(nodeTruncated)
        nodeObject.appendChild(nodeDifficult)
            
        nodeBndbox.appendChild(nodeXmin)
        nodeBndbox.appendChild(nodeYmin)
        nodeBndbox.appendChild(nodeXmax)
        nodeBndbox.appendChild(nodeYmax)
            
        nodeObject.appendChild(nodeBndbox)
        new_root.appendChild(nodeObject)
    
    doc.appendChild(new_root)
    with open(save_path + save_name, 'w') as nf:
         doc.writexml(nf, addindent='  ') 
         nf.close()
    # with open(os.path.join(save_path, save_name),'w') as fh:
    #     dom.writexml(fh)

def FindWrongMarkings(xml_folder, xml_files, spe_2_check):
    res = []
    for f in xml_files:
        tree = read_xml(xml_folder + f)
        root = tree.getroot()
        for child in root.findall('object'):
            spe = child.find('name').text
            if spe == spe_2_check:
                res.append(f)
                break
    return res

def parse_args():
    parser = argparse.ArgumentParser(description='Modify xml files')
    parser.add_argument('--xml_folder', dest='xml_folder', help = "Folder to save input xml documents", default="")
    
    parser.add_argument('--xml_suffix', dest='xml_suffix', help = "marking suffix", default="xml")
    
    parser.add_argument('--spe_wrong', dest='spe_wrong', help = "Wrong spe to detect", default="v")
    parser.add_argument('--spe_correct', dest='spe_correct', help = "correct spe to replace", default="deer_v")

    args = parser.parse_args()
    return args
        
if __name__ == '__main__':
    args = parse_args()
    xml_folder = args.xml_folder
    xml_suffix = args.xml_suffix
    spe_wrong = args.spe_wrong
    spe_correct = args.spe_correct
    xmls = getListWithCorrectMarker(xml_folder, xml_suffix)
    xmls_new = FindWrongMarkings(xml_folder, xmls, spe_wrong)
    
    for i in range(len(xmls_new)):

        tree_1 = read_xml(xml_folder + xmls_new[i])
        root_1 = tree_1.getroot()
    
        for name in root_1.iter('name'):
            if name.text == spe_wrong:
                name.text = spe_correct
        tree_1.root = root_1    
        tree_1.write(xml_folder + xmls_new[i])

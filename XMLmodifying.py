#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:19:28 2021

@author: weiwang
"""

#coding=utf-8
import os
import os.path
import xml.dom.minidom
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Modify xml files')
    parser.add_argument('--xml_path', dest='xml_path',
                      help='directory to xml_files',
                      default="")
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    path = args.xml_path
    files = os.listdir(path)  #get all files within a folder

    for xmlFile in files: #Traverse the folder

        if not os.path.isdir(xmlFile) and xmlFile.split('.')[1] != 'txt': #Only open if the dir is not a folder
            print(xmlFile)


        #dom for xml

            dom=xml.dom.minidom.parse(os.path.join(path,xmlFile)) 
            root=dom.documentElement
            name=root.getElementsByTagName('name')
            folder=root.getElementsByTagName('folder')


            # for i in range(len(name)):	
            #     print (name[i].firstChild.data)
            #     name[i].firstChild.data='plane'
            #     print (name[i].firstChild.data)

            for i in range(len(folder)):  
                #print (folder[i].firstChild.data)
                folder[i].firstChild.data='VOC2007'
                #print (folder[i].firstChild.data)

            with open(os.path.join(path,xmlFile),'w') as fh:
                dom.writexml(fh)
                #print('Done')


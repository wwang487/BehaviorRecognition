import glob

def getListWithCorrectMarker(folder, suffix):
    all_imgs = []
    for f in glob.glob(folder + '*.' + suffix):
        all_imgs.append(f.split('/')[-1])
    return all_imgs

folder_1 = './caltech_pic/caltech_label_photo/'
folder_2 = './caltech_pic/caltech_species_xml/'
folder_3 = './caltech_pic/caltech_v_nv_xml/'
all_imgs = getListWithCorrectMarker(folder_1, 'jpg')
xmls_1 = getListWithCorrectMarker(folder_2, 'xml')
xmls_2 = getListWithCorrectMarker(folder_3, 'xml')

for a_i in all_imgs:
    #if len(a_i.split('.')) == 2:
    #    check_a_i = a_i.split('.')[0] + '.' + 'xml'
    #else:
    #    check_a_i = a_i.split('.')[1][1:] + '.' + 'xml'
    check_a_i = a_i.split('.')[0] + '.' + 'xml'
    if check_a_i not in xmls_1 or check_a_i not in xmls_2:
        print('1')
        print(a_i)

for a_i in xmls_1:
    check_a_i = a_i.split('.')[0] + '.' + 'jpg'
    if check_a_i not in all_imgs or a_i not in xmls_2:
        print('2')
        print(a_i)

for a_i in xmls_2:
    check_a_i = a_i.split('.')[0] + '.' + 'jpg'
    if check_a_i not in all_imgs or a_i not in xmls_1:
        print('3')
        print(a_i)

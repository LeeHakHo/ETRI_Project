import os
import shutil
import glob
import re

datasets_path = '/home/ohh/dataset/ICDAR'

file_path = datasets_path + '/ICDAR15'
file_names = sorted(os.listdir(file_path))
image_path = datasets_path + '/ICDAR15'

new_file_path = file_path + '_OEpos'
new_image_path = image_path + '_OEpos'
if not os.path.isdir(new_file_path):
    os.makedirs(new_file_path)
if not os.path.isdir(new_image_path):
    os.makedirs(new_image_path)

# 라틴 확장문자 변환  #Ĳ ĳ ĸ Œ œ ſ ŉ
trans_char_dict = {
    'À':'A', 'Á': 'A', 'Â': 'A', 'Ã': 'A', 'Ä': 'A',
    'Ç':'C', 'È':'E', 'É':'E', 'Ê':'E', 'Ë':'E',
    'Ì':'I', 'Î':'I', 'Ñ':'N', 'Ò':'O', 'Ô':'O', 'Ö':'O', 'Ù':'U', 'Ü':'U',
    'à':'a', 'á':'a', 'â':'a', 'ä':'a', 'ç':'c', 'è':'e', 'é':'e', 'ê':'e', 'ì':'i', 'î':'i', 'ò':'o', 'ó':'o', 'ô':'o', 'ö':'o',
    'ù':'u', 'ú':'u', 'û':'u', 'ü':'u',
    'Ā':'A', 'ō':'o', 'Œ':'OE', 'œ':'oe', 'Š':'S','Ṡ':'S', 'Ÿ':'Y',
    '、':',', '‘':'\'', '《':'<', '》':'>','–':'-', '—':'-', '’':'\'', '“':'\"', '”':'\"',
    '²':'2', '×':'x', '™':'TM', '▪':'·', '●':'·', '・':'·', 'ـ':'_', '´':'\''
}
# transTable = txt.maketrans(trans_char_dict)
# txt = txt.translate(transTable)
regular_char = re.compile(r"[ 0-9a-zA-Z`=;,./~!@#$%^&*()_+|:<>?°·£¥₩€\-\[\]\'\"\{\}\\]")

def check_possible_word(word) -> bool:
    for char in word:
        if not regular_char.match(char):
            return False
    return True

count_pos_img, count_neg_img = 0, 0
count_pos_word, count_total_word = 0, 0

remain_latin_set = set([])

for file_name in file_names:
    src = os.path.join(file_path, file_name)
    try:
        with open(src, 'r', encoding='utf-8') as f:
            new_texts = ''
            lines = f.readlines()
            OEpos_exist = False
            for line in lines:

                texts = line[:-1].split(',')
                bbox = ','.join(texts[:8])
                script = '' #texts[8]
                word = ','.join(texts[8:])
                count_total_word += 1

                if word == '':                    # 빈 문자열 지우기
                    # count_neg_word += 1
                    continue
                elif word == '###':
                    script = 'null'
                else:
                    transTable = word.maketrans(trans_char_dict)
                    word = word.translate(transTable)

                    if check_possible_word(word):
                        script = 'OE_pos'
                        OEpos_exist = True
                        count_pos_word+=1
                    else:
                        script = 'Latin'
                        for char in word:
                            if not check_possible_word(char):
                                remain_latin_set.add(char)
                        word = '###'
                new_texts += bbox+','+script+','+word+'\n'
    except:
        print(file_name)
        continue
    if OEpos_exist:
        count_pos_img += 1
        new_src = os.path.join(new_file_path, file_name)
        with open(new_src,'w') as f:
            f.write(new_texts)
        image_name = 'img_' + file_name.split('_')[2].split('.')[0]+'.*'
        image_name = glob.glob(image_path+'/'+image_name)[0].split('/')[-1]
        image_src = os.path.join(image_path, image_name)
        new_image_src = os.path.join(new_image_path, image_name)
        shutil.copyfile(image_src, new_image_src)
    # #     # print(image_src, new_image_src)
    else:
        count_neg_img += 1
print('pos img: '+str(count_pos_img)+', neg img : '+str(count_neg_img))
print('total word: '+str(count_total_word)+', pog word : '+str(count_pos_word))
print(sorted(list(remain_latin_set)))
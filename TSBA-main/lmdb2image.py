import lmdb
from tqdm import tqdm
import six
from PIL import Image
import os
from glob import glob
import concurrent.futures
from concurrent.futures.thread import ThreadPoolExecutor
import functools
import time

def make_txt_lst_dict(data_dir_lst):
    txt_dict = {}
    txt_lst = []
    for idx, data_dir in enumerate(data_dir_lst):
        if data_dir.split('/')[0]=='train':
            path = '/home/ohh/dataset/MJ/' + data_dir+'/train_label.txt'
        else:
            path = '/home/ohh/dataset/MJ' + data_dir + '/valid_label.txt'
        txt_lst.append(path)


    for txt in txt_lst:
        txt_dict[txt] = []

    return txt_lst, txt_dict

root = '/home/ohh/PycharmProject/TSBA-main/data_lmdb_training/MJ/data_lmdb_release/validation/data.mdb'

lmdb_list = sorted(glob(root, recursive= True))

data_dir_lst = [dir_name.split('/')[8:-1] for dir_name in lmdb_list]
data_dir_lst = ['/'.join(val) for val in data_dir_lst]
txt_lst, g_txt_dict = make_txt_lst_dict(data_dir_lst)

def read_lmdb_wr_image(data_dir):
    img_save_dir = '/home/ohh/dataset/MJ/' + data_dir + '/images/'

    lmdb_file = '/home/ohh/PycharmProject/TSBA-main/data_lmdb_training/MJ/data_lmdb_release/training/MJ/MJ_train/'

    if data_dir.split('/')[0] == 'train':
        txtfile = '/home/ohh/dataset/MJ/' + data_dir+'/train_label.txt'
    else:
        txtfile = '/home/ohh/dataset/MJ/' + data_dir+'/valid_label.txt'

    rgb = True
    lmdb_env = lmdb.open(
        lmdb_file,
        max_readers=32,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )

    txn = lmdb_env.begin(write=False)
    lmd_cursor = txn.cursor()

    nSamples = int(txn.get(b"num-samples"))
    print('::Processing %s Total Samples:%d' % (data_dir, nSamples))

    line_list = []
    blank = ''
    txt = open(txtfile, 'w')
    for index in range(nSamples):
        index += 1
        label_key = 'label-%09d'.encode() % index
        label = txn.get(label_key).decode("utf-8")
        img_key = "image-%09d".encode() % index
        img_name = img_key.decode('utf-8')
        imgbuf = txn.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)

        if rgb:
            img = Image.open(buf).convert("RGB")
        else:
            img = Image.open(buf).conver("L")

        img_file = img_save_dir + img_name + '.jpg'
        print(img_file)
        img.save(img_file)
        img_path_lst = img_file.split('/')[-2:]
        img_path_forward = '/'.join(img_path_lst)

        delimiter = '	'

        line = img_path_forward + delimiter + label + '\n'

        txt.write(line)

    txt.close()
    print('Finish')

if __name__ == '__main__':

    for data_dir in data_dir_lst:
        img_save_dir = '/home/ohh/dataset/MJ/' + data_dir + '/images/'
        if not os.path.exists(img_save_dir):
            print('no directory: %s'%img_save_dir)

    with ThreadPoolExecutor(max_workers=4) as excutor:
        excutor.map(read_lmdb_wr_image, [data_dir for data_dir in data_dir_lst])
import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary
temp1 = 0.0
temp2 = 0.0

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def new_dice(pred, label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2 * tp_hard / (2 * tp_hard + fp_hard + fn_hard)

all_vol = []


def dice(pred, label):
    dice_all = []
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        z, x, y = label.shape
        for i in range(z):
            di = binary.dc(pred[i], label[i])
            dice_all.append(di)
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())


def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def process_label(label):
    tz = label == 2
    zg = label == 1

    return zg, tz


def test():

    path = '/remote-home/hhhhh/code_map/nnunetV1/DATASET/nnUNet_raw/nnUNet_raw_data/Task017_Pros'
    label_list = sorted(glob.glob(os.path.join(path, 'labelsTs', '*nii.gz')))

    infer_path = '/remote-home/hhhhh/Seg/Pos/pos_models_out/'

    infer_list = sorted(glob.glob(os.path.join(infer_path, 'output17e', '*nii.gz')))
    print("loading success...")
    Dice_et = []
    Dice_tc = []
    HD_et = []
    HD_tc = []

    file = infer_path + 'output17e/'
    if not os.path.exists(file):
        os.makedirs(file)
    fw = open(file + 'dice.txt', 'w')
    for label_path, infer_path in zip(label_list, infer_list):

        label, infer = read_nii(label_path), read_nii(infer_path)
        label_et, label_tc = process_label(label)
        infer_et, infer_tc = process_label(infer)

        Dice_et.append(dice(infer_et, label_et))
        Dice_tc.append(dice(infer_tc, label_tc))
        HD_et.append(hd(infer_et, label_et))
        HD_tc.append(hd(infer_tc, label_tc))

        fw.write('*' * 20 + '\n', )
        fw.write(infer_path.split('/')[-1] + '\n')
        fw.write('hd_et: {:.4f}\n'.format(HD_et[-1]))
        fw.write('hd_tc: {:.4f}\n'.format(HD_tc[-1]))

        fw.write('Dice_et: {:.4f}\n'.format(Dice_et[-1]))
        fw.write('Dice_tc: {:.4f}\n'.format(Dice_tc[-1]))


    dsc = []
    avg_hd = []
    dsc.append(np.mean(Dice_et))
    dsc.append(np.mean(Dice_tc))

    avg_hd.append(np.mean(HD_et))
    avg_hd.append(np.mean(HD_tc))

    fw.write('--' * 20 + '\n', )
    fw.write('Dice_zg ' + str(np.mean(Dice_et)) + ' ' + '\n')
    fw.write('Dice_tz ' + str(np.mean(Dice_tc)) + ' ' + '\n')

    print('Dice_zg and Dice_tz are: '  + str(np.mean(Dice_et))+'---' + str(np.mean(Dice_tc)))
    print('HD_zg and HD_tz are: ' + str(np.mean(HD_et)) + '---' + str(np.mean(HD_tc)))
    fw.write('--' * 20 + '\n', )
    fw.write('HD_zg ' + str(np.mean(HD_et)) + ' ' + '\n')
    fw.write('HD_tz ' + str(np.mean(HD_tc)) + ' ' + '\n')
    fw.write('--' * 20 + '\n', )
    fw.write('Dice ' + str(np.mean(dsc)) + ' ' + '\n')
    fw.write('HD ' + str(np.mean(avg_hd)) + ' ' + '\n')


    print(len(all_vol))
    print(all_vol)
    return np.mean(Dice_et), np.mean(Dice_tc)

if __name__ == '__main__':
    a, b = test()

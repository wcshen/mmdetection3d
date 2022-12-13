import re, sys, os
import numpy as np
import fire
from pluspy.file_utils import safe_make_dir

from functools import reduce


def main(folder='/home/plusai/Plus_2022/dataset/lidar/pc_label/L4E_origin_data/training',
         out='/home/plusai/Plus_2022/dataset/lidar/pc_label/L4E_origin_data/training',
         train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
         randomize=False,
         gen_slim_trainset=False):
    """

    Args:
        folder:
        out:
        train_ratio:
        val_ratio:
        test_ratio:
        randomize:
        gen_slim_trainset:

    Returns:
        generate all, train, value, test <- lidar_file_name[0]s
        all.txt:
            000000.1612512380.099664.20210205T141034_j7-feidian_25_229to249
            000001.1612512380.599636.20210205T141034_j7-feidian_25_229to249
            ......
    """
    files_list = os.listdir(folder + "/pointcloud")
    print(folder)
    totol_num = len(files_list)
    print('data num:', len(files_list))
    files_list = [file.replace('.bin', '') for file in files_list]
    perm = sorted(files_list)
    train_full_set = perm[:]
    train_set = perm[:int(totol_num * train_ratio)]
    val_set = perm[int(totol_num * train_ratio): int(totol_num * (train_ratio + val_ratio))]
    test_set = perm[int(totol_num * (train_ratio + val_ratio)):]
    print("all", len(train_full_set))
    print("train", len(train_set))
    print("value", len(val_set))
    print("test", len(test_set))
    out += '/../ImageSets'
    safe_make_dir(out)
    with open(out + "/trainval.txt", 'w') as f:
        for fid in train_full_set:
            print("%s" % fid, file=f)

    with open(out + "/train.txt", 'w') as f:
        for fid in train_set:
            print("%s" % fid, file=f)

    if gen_slim_trainset == True:
        # 10*n -> n
        bag2fids = {}
        for fid in train_set:
            # FIXME(swc): how many in one list?
            bag = fid[6:]
            if bag not in bag2fids:
                bag2fids[bag] = []
            bag2fids[bag].append(fid)
        for bag in bag2fids.keys():
            vs = bag2fids[bag]
            bag2fids[bag] = list(np.random.choice(vs, len(vs) // 10, replace=False))
        with open(out + "/train_slim.txt", 'w') as f:
            fids = reduce(lambda s, x: s + x, bag2fids.values(), [])
            for fid in fids:
                print("%s" % fid, file=f)

    with open(out + "/val.txt", 'w') as f:
        for fid in val_set:
            print("%s" % fid, file=f)
    with open(out + "/test.txt", 'w') as f:
        for fid in test_set:
            print("%s" % fid, file=f)


# NOTE(swc): two ways to call it:
# add.py 1 2
# add.py --a 1 --b 2
if __name__ == '__main__':
    fire.Fire(main)

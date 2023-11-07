# You can run this file to convert the rotation-equivariant YOHO-Desc to the rotation-invariant yoho-desc we need on your own datasets.
import os
import argparse
import numpy as np
from tqdm import tqdm
from utils.utils import make_non_exists_dir
from dataops.dataset import get_dataset_name

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',      default='3dmatch',type=str,help='dataset name')
parser.add_argument('--yoho_desc_dir',default='',type=str,help='the folder of YOHO_FCGF/Testset')
args = parser.parse_args()

datasets = get_dataset_name(args.dataset, './data')
for name, dataset in tqdm(datasets.items()):
    if name in ['wholesetname']:continue
    for pid in tqdm(range(len(dataset.pc_ids))):
        desc = np.load(f'{args.yoho_desc_dir}/{dataset.name}/YOHO_Output_Group_feature')
        desc = np.mean(desc, axis = -1)
        desc_dir = f'data/{dataset.name}/yoho_desc/{pid}.npy'
        make_non_exists_dir(desc_dir)
        np.save(f'{desc_dir}/{pid}.npy',desc)
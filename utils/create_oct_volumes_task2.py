import os
import nibabel as nib
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image


def create_vol_from_fid(imgs_path, fid, df, target_slices=25):
    df_vol = df[df['LOCALIZER'] == fid]
    vol_info = df_vol.iloc[[0]]
    vol_info = vol_info.drop(columns=['image', 'BScan', 'case'])
    df_vol = df_vol.sort_values(by='BScan')
    df_vol.reset_index(drop=True, inplace=True)
    images = list(df_vol['image'])
    num_slices = len(images)

    if target_slices == -1:
        vol_size = len(images)
        vol_info.drop(columns=['label'], inplace=True)
        vol_info['labels'] = [df_vol['label'].tolist()]
    elif num_slices < target_slices:
        # Calculate the number of slices to add at the beginning and end
        slices_to_add = target_slices - num_slices
        add_start = slices_to_add // 2
        add_end = slices_to_add - add_start
        images = [images[0]] * add_start + images + [images[-1]] * add_end
        vol_size = len(images)
        assert vol_size == target_slices

    elif num_slices > target_slices:
        # Calculate the number of slices to remove at the beginning and end
        slices_to_remove = num_slices - target_slices
        remove_start = slices_to_remove // 2
        remove_end = slices_to_remove - remove_start
        images = images[remove_start:-remove_end]
        vol_size = len(images)
        assert vol_size == target_slices

    elif num_slices == target_slices:
        vol_size = len(images)
    
    first_img = np.array(Image.open(os.path.join(imgs_path, images[0])).convert('L'))
    vol_ti = np.zeros((first_img.shape[0], first_img.shape[1], vol_size), dtype=first_img.dtype)
    for i in range(vol_size):
        img = np.array(Image.open(os.path.join(imgs_path, images[i])).convert('L'))
        vol_ti[:, :, i] = img

    return vol_ti, vol_info


def main(csv_path: str, imgs_path: str, out_path: str):
    df_all = pd.read_csv(csv_path)
    fundus_ids = df_all['LOCALIZER'].unique()
    df_vols = pd.DataFrame(columns=['id_patient','side_eye','split_type','label','LOCALIZER','sex','age','num_current_visit'])
    for fid in tqdm(fundus_ids, ncols=90):
        vol, vol_info = create_vol_from_fid(imgs_path, fid, df_all)
        df_vols = pd.concat([df_vols, vol_info], ignore_index=True)
        nii = nib.Nifti1Image(vol, np.eye(4))
        nib.save(nii, os.path.join(out_path, f'{fid.replace('.png', '')}.nii.gz'))
    df_vols.rename(columns={'LOCALIZER': 'volume'}, inplace=True)
    df_vols['volume'] = df_vols['volume'].apply(lambda x: x.replace('.png', '.nii.gz'))
    df_vols.to_csv(csv_path, index=False)

import os
import nibabel as nib
import numpy as np
import pandas as pd

from tqdm import tqdm
from PIL import Image


def create_vol_from_fid(imgs_path, fid, df, target_slices=25):
    try:
        df_vol = df[df['LOCALIZER_at_ti'] == fid]
        is_ti = True
        assert len(df_vol) > 0
    except AssertionError:
        df_vol = df[df['LOCALIZER_at_ti+1'] == fid]
        is_ti = False
    vol_info = df_vol.iloc[[0]]
    vol_info = vol_info.drop(columns=['image_at_ti', 'image_at_ti+1', 'BScan', 'label', 'case'])
    df_vol = df_vol.sort_values(by='BScan')
    df_vol.reset_index(drop=True, inplace=True)
    images = list(df_vol['image_at_ti' if is_ti else 'image_at_ti+1'])
    num_slices = len(images)

    if target_slices == -1:
        vol_size = len(images)
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
    
    vol_info['label'] = [df_vol['label'].tolist()]
    vol_info['case'] = [df_vol['case'].tolist()]

    first_img = np.array(Image.open(os.path.join(imgs_path, images[0])).convert('L'))
    vol_ti = np.zeros((first_img.shape[0], first_img.shape[1], vol_size), dtype=first_img.dtype)
    for i in range(vol_size):
        img = np.array(Image.open(os.path.join(imgs_path, images[i])).convert('L'))
        vol_ti[:, :, i] = img

    return vol_ti, vol_info


def main(csv_path: str, imgs_path: str, out_path: str):
    df_all = pd.read_csv(csv_path).sort_values(by='id_patient')
    fundus_ids = list(df_all['LOCALIZER_at_ti'].unique())
    fundus_ids += [case for case in df_all['LOCALIZER_at_ti+1'].unique() if case not in fundus_ids]
    df_vols = pd.DataFrame(columns=['id_patient', 'side_eye', 'split_type', 'LOCALIZER_at_ti',
                                    'LOCALIZER_at_ti+1', 'sex', 'age_at_ti', 'age_at_ti+1',
                                    'num_current_visit_at_i', 'num_current_visit_at_i+1', 
                                    'delta_t', 'case', 'label'])
    for fid in tqdm(fundus_ids, ncols=90):
        vol, vol_info = create_vol_from_fid(imgs_path, fid, df_all)
        df_vols = pd.concat([df_vols, vol_info], ignore_index=True)
        nii = nib.Nifti1Image(vol, np.eye(4))
        nib.save(nii, os.path.join(out_path, f'{fid.replace('.png', '')}.nii.gz'))
    df_vols.rename(columns={'LOCALIZER_at_ti': 'volume_ti', 'LOCALIZER_at_ti+1': 'volume_ti+1'}, inplace=True)
    df_vols['volume_ti'] = df_vols['volume_ti'].apply(lambda x: x.replace('.png', '.nii.gz'))
    df_vols['volume_ti+1'] = df_vols['volume_ti+1'].apply(lambda x: x.replace('.png', '.nii.gz'))
    
    df_vols.drop_duplicates(subset=['id_patient', 'side_eye', 'split_type', 'volume_ti', 'volume_ti+1'], inplace=True)
    df_vols.to_csv(csv_path, index=False)

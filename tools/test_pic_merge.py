import numpy as np
import math
import os
from PIL import Image

def read_image(filepath):
    try:
        with Image.open(filepath) as img:
            img_array = np.array(img)
        return img_array
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def save_image(image, filepath):
    image = (image * 255).astype(np.uint8)
    Image.fromarray(image).save(filepath)

def binarize_image(image, threshold):
    return image > threshold

def extract_unique_sceneids(result_root, preds_dir):
    path_4landtype = os.path.join(result_root, preds_dir[0])
    files_inside_landtype = os.listdir(path_4landtype)
    
    sceneid_lists = []
    for filename in files_inside_landtype:
        if filename.endswith('.png'):
            raw_result_patch_name = filename.replace('.png', '')
            loc1 = raw_result_patch_name.index('LC')
            scene_id = raw_result_patch_name[loc1:]
            sceneid_lists.append(scene_id)
    
    uniq_sceneid = list(set(sceneid_lists))
    return uniq_sceneid


def extract_rowcol_each_patch(name,original_size=(1000, 1000), crop_size=(384, 384), step=308):
    name = name.replace('.png', '')
    # loc1 = name.index('LC')
    loc2 = name.index('_')
    patchbad = name[:loc2]
    num_patches_y = (original_size[0] - crop_size[0]) // step + 1
    num_patches_x = (original_size[1] - crop_size[1]) // step + 1
    row = math.ceil(int(patchbad)/num_patches_y)-1
    col = int(patchbad)-num_patches_y*row-1
    return row, col


def get_patches_for_sceneid(preds_folder_root, preds_folder, sceneid):
    path_4preds = os.path.join(preds_folder_root, preds_folder[0])
    files_inside = os.listdir(path_4preds)
    return [f for f in files_inside if sceneid in f]

def unzeropad(in_dest, in_source):
    ny, nx = in_dest.shape
    nys, nxs = in_source.shape
    # tmpy = (ny - nys) // 2
    # tmpx = (nx - nxs) // 2
    tmpy = np.floor((ny - nys) / 2).astype(int)
    tmpx = np.floor((nx - nxs) / 2).astype(int) 
    return in_dest[tmpy:tmpy + nys, tmpx:tmpx + nxs]

def main():
    gt_folder_path = '/home/data4/zjq/l8cloudmasks/label'
    preds_folder_root = '/home/data4/zjq/l8cloudmasks/'
    preds_folder = ['result_dianet']
    # pr_patch_size_rows = 384
    # pr_patch_size_cols = 384
    crop_size = (200,200)
    step = 200
    classes = [0, 1]
    conf_matrix_print_out = 0
    thresh = 12 / 255
    files_inside_landtype = os.listdir(gt_folder_path)



    all_uniq_sceneid = extract_unique_sceneids(preds_folder_root, preds_folder)

    for n, sceneid in enumerate(all_uniq_sceneid):
        print(f'Working on sceneID # {n + 1} : {sceneid} ...')

        gt_path = os.path.join(gt_folder_path, f'{sceneid}.png')
        gt = read_image(gt_path) //255

        scid_related_patches = get_patches_for_sceneid(preds_folder_root, preds_folder, sceneid)
        h,w=gt.shape
      
        complete_pred_mask = np.zeros((h,w))

        
        ############# 拼接图像 #############
        for patch_name in scid_related_patches:
            predicted_patch_path = os.path.join(preds_folder_root, preds_folder[0], patch_name)
            predicted_patch = read_image(predicted_patch_path)

            predicted_patch = binarize_image(predicted_patch, thresh)

            y, x = extract_rowcol_each_patch(patch_name,original_size=(1000, 1000), crop_size=crop_size, step=step)
            complete_pred_mask[y * step:y * step + crop_size[0], x * step:x * step + crop_size[1]] = predicted_patch

        complete_folder = f'entire_masks_{preds_folder[0]}'
        complete_folder_path = os.path.join(preds_folder_root, complete_folder)

        if not os.path.exists(complete_folder_path):
            os.makedirs(complete_folder_path)

        baseFileName = f'{sceneid}.png'
        save_image(complete_pred_mask, os.path.join(complete_folder_path, baseFileName))


if __name__ == "__main__":
    main()

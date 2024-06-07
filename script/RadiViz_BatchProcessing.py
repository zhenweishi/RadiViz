import SimpleITK as sitk
from radiomics import featureextractor
import yaml
import numpy as np
import monai
from pathlib import Path
import matplotlib.pyplot as plt
# import copy
import time
import pandas as pd
# import os


# 需要统计时间的代码
start_time = time.time()

### For users--------------------------
casetable = r'../data/data4casetable/casetable_data1.csv'
# img_dir = r"../data/DCE_test.nii.gz"
# mask_dir = r"../data/DCE_ROI_test.nii.gz"
# params_dir = r"../Params/RadiomicsParams_HelloRadiViz.yaml"
params_dir = r"../Params/RadiomicsParams_MR_original.yaml"
output_root = Path(r"../output") #if not exist, will create automatically.
fig_format = 'png'

# Load casetable
df_case = pd.read_csv(casetable).T

for entry in df_case:
    img_dir = df_case[entry]['Image']
    mask_dir = df_case[entry]['Mask']
    ptid = df_case[entry]['ID']
    # RadiViz_Feautre_Extraction(ptid,img_dir,mask_dir,params_dir)

    print('Start Processing')
    ## Load the image and mask 
    image = sitk.ReadImage(img_dir) ### For users
    mask = sitk.ReadImage(mask_dir) ### For users

    # check if the image and mask have the same size
    assert image.GetSize() == mask.GetSize()

    # convert the image and mask to numpy arrays
    image_np = sitk.GetArrayFromImage(image)
    mask_np = sitk.GetArrayFromImage(mask)



    ## Load feature extractor
    # select radiomics calculation parameters from a yaml file
    params = yaml.load(open(params_dir, "r"), Loader=yaml.SafeLoader) ### For users
    # params = yaml.load(open("Params/RadiomicsParams.yaml", "r"), Loader=yaml.SafeLoader)
    extractor = featureextractor.RadiomicsFeatureExtractor(params)

    ## ----------------------------------_核心代码------------------------------
    ## Crop image and mask to reduce computation time

    # roi_start, roi_end 后面可用于还原ROI位置
    use_whole_mask = False ### For users
    margin = params.get("voxelSetting", {}).get("kernelRadius", 1)

    if use_whole_mask:
        roi_start, roi_end = monai.transforms.generate_spatial_bounding_box(mask_np[None], margin=margin, allow_smaller=False)
        cropper = monai.transforms.SpatialCrop(roi_start=roi_start, roi_end=roi_end)
    else:
        z_idx = np.argmax(np.sum(mask_np, axis=(1, 2)))
        z_start = max(0, z_idx - margin)
        z_end = min(mask_np.shape[0], z_idx + margin)
        roi_start, roi_end = monai.transforms.generate_spatial_bounding_box(mask_np[z_start:z_end][None], margin=0, allow_smaller=False)
        roi_start[0] += z_start
        roi_end[0] += z_start + 1

    for i in range(3):
        if (roi_end[i] - roi_start[i]) % 2 == 0:
            roi_end[i] += 1

    # ----- crop roi ----
    cropper = monai.transforms.SpatialCrop(roi_start=roi_start, roi_end=roi_end)
    image_roi_np, mask_roi_np = cropper(image_np[None])[0], cropper(mask_np[None])[0]

    # ----- copy information (optional) ------
    image_roi = sitk.GetImageFromArray(image_roi_np)
    image_roi.SetDirection(image.GetDirection())
    image_roi.SetOrigin(image.GetOrigin())
    image_roi.SetSpacing(image.GetSpacing())

    mask_roi = sitk.GetImageFromArray(mask_roi_np)
    mask_roi.SetDirection(mask.GetDirection())
    mask_roi.SetOrigin(mask.GetOrigin())
    mask_roi.SetSpacing(mask.GetSpacing())

    print(image_roi_np.shape)

    print('Start Feature extraction')
    ## Feature extraction
    results = extractor.execute(image_roi, mask_roi, voxelBased=True)

    ## Save feature maps
    for name, feat_map in results.items():
        if name.startswith("diagnostics_"):
            continue
        image_type = name.split("_")[0].split("-")[0]
        
        type_dir = output_root / "feature_map" / image_type
        feat_path = type_dir / f"{name}.nii.gz"
        feat_path.parent.mkdir(parents=True, exist_ok=True)

        sitk.WriteImage(feat_map, feat_path)


    print('Start feature maps visualization')
    ## Visualize feature maps
    norm01 = monai.transforms.ScaleIntensity(minv=0.0, maxv=1.0)
    norm = monai.transforms.NormalizeIntensity()
    cmap = plt.get_cmap("rainbow")


    # Find the slice with the maximum sum of the mask values
    slice_idx = np.argmax(np.sum(mask_np, axis=(1, 2)))

    # Normalize the image slice and expand dimensions for RGB
    image_slice = norm01(image_np[slice_idx])
    image3c_slice = np.repeat(image_slice[..., None], 3, axis=-1)

    # Expand dimensions for RGB mask
    mask_slice = mask_np[slice_idx]
    mask3c_slice = np.repeat(mask_slice[..., None], 3, axis=-1)

    ## ----------------------------------_核心代码 结束------------------------------
    print('Save color map')
    for name, feat_map in results.items():
        if name.startswith("diagnostics_"):
            continue
        
        image_type = name.split("_")[0].split("-")[0]
        print(name)

        # Initialize an empty feature array of the same size as mask_np
        feat_np = np.zeros_like(mask_np).astype(np.float32)

        # Assign the feature map to the ROI in the feature array
        feat_map_np = sitk.GetArrayFromImage(feat_map)
        if feat_map_np.shape != mask_roi_np.shape: # 可能过了resample, 要resize回来
            feat_map_np = monai.transforms.Resize(spatial_size=mask_roi_np.shape)(feat_map_np[None])[0].numpy()

        feat_np[roi_start[0]:roi_end[0], 
                roi_start[1]:roi_end[1], 
                roi_start[2]:roi_end[2]] = feat_map_np
        
        # Normalize the feature slice
        feat_slice = norm01(feat_np[slice_idx])

        # Create RGBA and RGB arrays from the feature slice
        rgba_np = cmap(feat_slice)
        rgb_np = np.delete(rgba_np, 3, 2)

        # Combine the feature RGB array with the original image slice based on the mask
        rgb_np = np.where(mask3c_slice > 0, rgb_np, image3c_slice)

        # Plot using fig and ax
        fig, ax = plt.subplots()
        cax = ax.imshow(rgb_np, cmap=cmap)
        fig.colorbar(cax, ax=ax)
        plt.title(name)

        color_map_dir = output_root / ptid / "color_map" 
        color_map = color_map_dir / f"{name+'.'+fig_format}"
        color_map.parent.mkdir(exist_ok=True, parents=True)

        # type_dir = output_root / "feature_map" / image_type
        # feat_path = type_dir / f"{name}.nii.gz"
        # feat_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(color_map, format=fig_format, dpi=300, bbox_inches='tight') ### For users
        plt.close(fig)  # 关闭图形,确保不会意外地显示

    end_time = time.time()
    print(f"运行时间: {end_time - start_time:.2f} 秒")
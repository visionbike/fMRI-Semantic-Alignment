from typing import Dict
import os
from pathlib import Path
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from nilearn.datasets import load_mni152_gm_mask
from nilearn import maskers



def load_fmri(path_file: Path, path_save: Path) -> Dict:
    """
    Convert nifti image to numpy array.

    Args:
        path_file (Path): input nifti file path.
        path_save (Path): saving path.
    Returns:
        (Dict): dictionary of input responses and corresponding labels.
    """
    subject = path_file.name[:4]
    path_feat = path_save / f"{subject}.npy"
    if os.path.exists(path_feat):
        # load feature that is applied decoding
        features = np.load(path_feat)
    else:
        # apply decoding to image voxel space (50k voxels)
        # data is down-sampled to 3mm isotropic resolution for consistency for reducing computation costs.
        mask = load_mni152_gm_mask(resolution=3)
        masker = maskers.NiftiMasker(
            mask_img=mask,
            verbose=11,
            n_jobs=20,
        )
        features = masker.fit_transform(path_file)
        np.save(path_feat, features)
    # get class labels
    path_label = path_file.parent / f"{subject}_labels.csv"
    labels = pd.read_csv(path_label, header=None)[0].values
    return {"features": features, "labels": labels}


if __name__ == "__main__":
    path_root = Path("./data")
    path_fmri = path_root / "fmri"
    path_stimuli = path_root / "stimuli"
    path_out = path_root / "voxel_feats"
    path_out.mkdir(parents=True, exist_ok=True)
    # load fmri data and decode to voxel features
    num_jobs = 10
    data = Parallel(n_jobs=num_jobs, verbose=11, backend="multiprocessing")(
        delayed(load_fmri)(
            path_file=path,
            path_save=path_out
        )
        for path in path_fmri.glob("*.nii.gz")
    )
    # combine data from different subjects
    data_ = dict(features=[], labels=[])
    for entry in data:
        for key in ["features", "labels"]:
            data_[key].extend(entry[key])
    for key in ["features", "labels"]:
        data_[key] = np.array(data_[key])
    data = data_.copy()
    del data_
    # convert str labels to numerical
    unique_labels, index_labels = np.unique(data["labels"], return_inverse=True)
    data["labels"] = index_labels
    # get stimuli images
    stimuli = {}
    for label in unique_labels:
        files = [str(f) for f in (path_stimuli / label).glob("*.JPEG")]
        stimuli[list(unique_labels).index(label)] = files
    # process data
    path_process = path_root / "processed_fmri"
    path_process.mkdir(parents=True, exist_ok=True)
    np.save(path_process / "label_map.npy", unique_labels)
    np.save(path_process / "stimuli_map.npy", stimuli, allow_pickle=True)
    # train/val/test set
    x_trainval, x_test, y_trainval, y_test = train_test_split(data["features"], data["labels"], test_size=0.1, random_state=1)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, test_size=0.1 / 0.9, random_state=1)
    np.save(path_process / "train.npy", {"data": x_train, "label": y_train}, allow_pickle=True)
    np.save(path_process / "val.npy", {"data": x_val, "label": y_val}, allow_pickle=True)
    np.save(path_process / "test.npy", {"data": x_test, "label": y_test}, allow_pickle=True)
    # load saved data
    map_label = np.load(path_process / "label_map.npy").tolist()
    print("label names: ", map_label)
    map_stimuli = np.load(path_process / "stimuli_map.npy", allow_pickle=True).tolist()
    print("stimuli images: ")
    for i in map_stimuli.keys():
        print(f"Number of {map_label[i]} image: {len(map_stimuli[i])}")
    data_train = np.load(path_process / "train.npy", allow_pickle=True).tolist()
    print(f"Train data - input shape: {data_train['data'].shape}, label shape: {data_train['label'].shape}")
    data_val = np.load(path_process / "val.npy", allow_pickle=True).tolist()
    print(f"Val data - input shape: {data_val['data'].shape}, label shape: {data_val['label'].shape}")
    data_test = np.load(path_process / "test.npy", allow_pickle=True).tolist()
    print(f"Test data - input shape: {data_test['data'].shape}, label shape: {data_test['label'].shape}")

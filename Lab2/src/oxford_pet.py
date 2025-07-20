import os
import torch
import shutil
import numpy as np
import imutils
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
from torchvision import transforms

class OxfordPetDataset(torch.utils.data.Dataset):
    def __init__(self, root, mode="train", transform=None):

        assert mode in {"train", "valid", "test"}

        self.root = root
        self.mode = mode
        self.transform = transform

        self.images_directory = os.path.join(self.root, "images")
        self.masks_directory = os.path.join(self.root, "annotations", "trimaps")

        self.filenames = self._read_split()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        image_path = os.path.join(self.images_directory, filename + ".jpg")
        mask_path = os.path.join(self.masks_directory, filename + ".png")

        image = np.array(Image.open(image_path).convert("RGB"))

        trimap = np.array(Image.open(mask_path))
        mask = self._preprocess_mask(trimap)

        sample = dict(image=image, mask=mask, trimap=trimap)
        if self.transform is not None:
            sample = self.transform(**sample)

        return sample

    @staticmethod
    def _preprocess_mask(mask):
        mask = mask.astype(np.float32)
        mask[mask == 2.0] = 0.0
        mask[(mask == 1.0) | (mask == 3.0)] = 1.0
        return mask

    def _read_split(self):
        split_filename = "test.txt" if self.mode == "test" else "trainval.txt"
        split_filepath = os.path.join(self.root, "annotations", split_filename)
        with open(split_filepath) as f:
            split_data = f.read().strip("\n").split("\n")
        filenames = [x.split(" ")[0] for x in split_data]
        if self.mode == "train":  # 90% for train
            filenames = [x for i, x in enumerate(filenames) if i % 10 != 0]
        elif self.mode == "valid":  # 10% for validation
            filenames = [x for i, x in enumerate(filenames) if i % 10 == 0]
        return filenames

    @staticmethod
    def download(root):

        # load images
        filepath = os.path.join(root, "images.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)

        # load annotations
        filepath = os.path.join(root, "annotations.tar.gz")
        download_url(
            url="https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
            filepath=filepath,
        )
        extract_archive(filepath)


class SimpleOxfordPetDataset(OxfordPetDataset):
    def __getitem__(self, *args, **kwargs):

        sample = super().__getitem__(*args, **kwargs)

        # resize images
        image = np.array(Image.fromarray(sample["image"]).resize((256, 256), Image.BILINEAR))
        mask = np.array(Image.fromarray(sample["mask"]).resize((256, 256), Image.NEAREST))
        trimap = np.array(Image.fromarray(sample["trimap"]).resize((256, 256), Image.NEAREST))

        # convert to float32 and normalize
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32)
        trimap = trimap.astype(np.float32)

        # convert to other format HWC -> CHW
        sample["image"] = np.moveaxis(image, -1, 0)
        sample["mask"] = np.expand_dims(mask, 0)
        sample["trimap"] = np.expand_dims(trimap, 0)

        # convert to torch tensors
        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"])
        sample["trimap"] = torch.from_numpy(sample["trimap"])

        return sample


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, filepath):
    directory = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(directory, exist_ok=True)
    if os.path.exists(filepath):
        return

    with TqdmUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=os.path.basename(filepath),
    ) as t:
        urlretrieve(url, filename=filepath, reporthook=t.update_to, data=None)
        t.total = t.n


def extract_archive(filepath):
    extract_dir = os.path.dirname(os.path.abspath(filepath))
    dst_dir = os.path.splitext(filepath)[0]
    if not os.path.exists(dst_dir):
        shutil.unpack_archive(filepath, extract_dir)

def myTransform(image, mask, trimap):
    #random rotate
    degree = np.random.randint(-15, 15)
    image = imutils.rotate(image, degree)
    mask = imutils.rotate(mask, degree)
    trimap = imutils.rotate(trimap, degree)
    
    #random flipe horizontal
    if np.random.random() > 0.5:
        image = np.flip(image, axis=1)
        mask = np.flip(mask, axis=1)
        trimap = np.flip(trimap, axis=1)
    # vertical filp
    if np.random.random() > 0.5:
        image = np.flip(image, axis=0)
        mask = np.flip(mask, axis=0)
        trimap = np.flip(trimap, axis=0)
    
    return dict(image=image, mask=mask, trimap=trimap)
    

def load_dataset(data_path, mode):
    if not os.path.exists(data_path) or not os.path.exists(os.path.join(data_path, "annotations")):
        OxfordPetDataset.download(data_path)
    if mode not in ["train", "valid", "test"]:
        raise ValueError("Mode must be one of 'train', 'valid', or 'test'.")

    if mode == "train":
        #train transform with data augmentation
        dataset = SimpleOxfordPetDataset(root=data_path, mode=mode)
        dataset.transform = myTransform
        print(f"Loaded {mode} dataset with {len(dataset)} samples from {data_path}.")
    else:
        # valid and test datasets
        dataset = SimpleOxfordPetDataset(root=data_path, mode=mode)
        print(f"Loaded {mode} dataset with {len(dataset)} samples from {data_path}.")
        
    return dataset

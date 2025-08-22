import torch
from torch.utils.data import Dataset
import json
from torchvision import transforms
import numpy as np
from PIL import Image


class Iclevr_Dataset(Dataset):
    def __init__(self, mode='train'):
        self.object_dict = json.load(open('data/object.json'))
        self.data = json.load(open('data/train.json'))
        # convert dict to list of list
        self.data = [['data/iclevr/'+keys, self.data[keys]] for keys in self.data.keys()]
        def pil_to_tensor_no_numpy(im: Image.Image) -> torch.Tensor:
            im = im.convert('RGB')
            w, h = im.size
            # tobytes -> ByteStorage -> ByteTensor, avoid numpy path entirely
            byte_tensor = torch.ByteTensor(torch.ByteStorage.from_buffer(im.tobytes()))
            tensor_chw = byte_tensor.view(h, w, 3).permute(2, 0, 1).to(torch.float32) / 255.0
            return tensor_chw

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.Lambda(pil_to_tensor_no_numpy),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = Image.open(self.data[idx][0]).convert('RGB')
        image = self.transform(image)
        label = torch.zeros(24)
        for obj in self.data[idx][1]:
            label[self.object_dict[obj]] = 1
        # new_label = torch.zeros(24*23*22//6)
        # index = 0
        # for i in range(24):
        #     if label[i] == 1:
        #         index += pow(2, 23-i)
        # new_label[index] = 1
        return image, label
    
if __name__ == '__main__':
    dataset = Iclevr_Dataset()
    print(len(dataset))
    print(dataset[0])
    print(dataset[1][0].shape)
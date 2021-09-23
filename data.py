import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import copy


class DataTransforms:
    def __init__(self):
        self.transformations = {
            'train': transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomErasing()
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
}


class LookItDataset:
    def __init__(self, args):
        self.args = args
        self.transforms = DataTransforms().transformations
        self.img_processor = self.transforms[args.phase]
        self.img_paths = self.collect_paths()

    def collect_paths(self):
        img_paths = []
        for file in self.args.dataset_folder.glob("**/*"):
            if file.parent.name == self.args.phase:
                img_paths.append(file)
        return img_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_files_seg, box_files_seg, class_seg = self.img_paths[index]

        imgs = []
        for img_file in img_files_seg:
            img = Image.open(self.args.dataset_folder / img_file).convert('RGB')
            img = self.img_processor(img)
            imgs.append(img)
        imgs = torch.stack(imgs)

        boxs = []
        for box_file in box_files_seg:
            box = np.load(self.args.dataset_folder / box_file, allow_pickle=True).item()
            box = torch.tensor([box['face_size'], box['face_ver'], box['face_hor'], box['face_height'], box['face_width']])
            boxs.append(box)
        boxs = torch.stack(boxs)
        boxs = boxs.float()

        return {
            'imgs': imgs,  # n x 3 x 100 x 100
            'boxs': boxs,  # n x 5
            'label': class_seg
        }


class MyDataLoader:
    def __init__(self, opt):
        self.opt = copy.deepcopy(opt)
        self.dataset = LookItDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=self.opt.is_train,
            num_workers=int(opt.num_threads)
        )

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
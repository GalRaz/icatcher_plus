from torchvision import transforms, datasets
import torch.utils.data.dataloader as dataloader
from config import *
from pathlib import Path
from .fc_eval import get_fc_data_transforms


def get_dataset_dataloaders(args, input_size, batch_size, shuffle=True, num_workers=4):
    data_transforms = get_fc_data_transforms(args, input_size)

    # Create training and validation datasets
    image_datasets = {'train': datasets.ImageFolder(str(Path(face_data_folder, 'train')), data_transforms['train']),
                      'val': datasets.ImageFolder(str(Path(face_data_folder, 'val')), data_transforms['val']),
                      }
    # print('\n\nImageFolder class to idx: ', image_datasets['val'].class_to_idx)
    # infant - 0, target - 1
    print("# train samples:", len(image_datasets['train']))
    print("# validation samples:", len(image_datasets['val']))

    # Create training and validation dataloaders, never shuffle val and test set
    dataloaders_dict = {x: dataloader.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=False if x != 'train' else shuffle,
                                                 num_workers=num_workers) for x in data_transforms.keys()}
    return dataloaders_dict

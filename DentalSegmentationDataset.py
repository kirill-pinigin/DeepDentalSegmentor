import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import random
from scipy.interpolate import splprep, splev
import imutils

INPUT_DIMENSION = int(3)
OUTPUT_DIMENSION = int(33)
MAXIMUM_MASK_VALUE = int(160)
MINIMUM_MASK_VALUE  = int(0)
TOOTH_ID = np.linspace(MINIMUM_MASK_VALUE, MAXIMUM_MASK_VALUE, OUTPUT_DIMENSION).astype(np.float32)
MIN_RECT_POINTS_DIMENSION = int(160)

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg", ".bmp"])

def load_image(filepath, channels = 1, resolution = 256):
    if channels == 1:
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)

    return cv2.resize(image, (resolution, resolution), interpolation = cv2.INTER_NEAREST if channels == 1 else cv2.INTER_AREA)

def aproximate_contour(contour, n_points = 16, resolution = 256):
    x, y = contour.T
    x = np.clip(x, 0, resolution)
    y = np.clip(y, 0, resolution)
    x = x.tolist()[0]
    y = y.tolist()[0]
    tck, u = splprep([x, y], u=None, s=0.0, per=1)
    u_new = np.linspace(u.min(), u.max(), n_points)
    x_new, y_new = splev(u_new, tck, der=0)
    x_new = np.clip(x_new, 0, resolution)
    y_new = np.clip(y_new, 0, resolution)
    result = np.stack((x_new, y_new), axis = 1)
    return result

class DentalSegmentationDataset(Dataset):
    def __init__(self, data_dir, split : str = 'train', resolution : int = 256, ):
        self.resolution = resolution
        self.image_paths = [os.path.join(data_dir + '/images/', x) for x in os.listdir(data_dir + '/images/') if is_image_file(x)]
        self.image_paths.sort()
        self.mask_paths = [os.path.join(data_dir + '/maskes/', x) for x in os.listdir(data_dir + '/maskes/') if is_image_file(x)]
        self.mask_paths.sort()
        self.edge_paths = None
        self.skeleton_paths = None
        self.point_paths = None
        random_inst = random.Random(42)
        n_items = len(self.image_paths)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        self.split = split

        if self.split == 'train': idxs = [idx for idx in range(n_items) if idx not in idxs]

        self.image_paths = [self.image_paths[i] for i in idxs]
        self.mask_paths = [self.mask_paths[i] for i in idxs]
        self.image_paths.sort()
        self.mask_paths.sort()

        self.tensor_transformation = transforms.ToTensor()

    def __getitem__(self, index):
        image = load_image(self.image_paths[index], channels = INPUT_DIMENSION, resolution = self.resolution)
        image = self.tensor_transformation(image)
        data = load_image(self.mask_paths[index], resolution = self.resolution)
        masks = [(data == id) for id in TOOTH_ID]
        mask = np.stack(masks, axis=2).astype('float32')
        segmentation_map = torch.from_numpy(mask).long().permute(2, 0, 1)
        return image, segmentation_map

    def __len__(self):
        return len(self.image_paths)


class DentalSegmentationDetectionDataset(DentalSegmentationDataset):
    def __init__(self, data_dir, split : str = 'train', resolution : int = 256, ):
        super(DentalSegmentationDetectionDataset, self).__init__(data_dir, split, resolution)

    def pointate(self, contour, resolution=512):
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        center = np.array([int(rect[0][0]), int(rect[0][1])]).astype(np.float32).reshape(1, 2)/(float(resolution))
        result = np.append(center, box.astype(np.float32)/(float(resolution)), axis=0)
        return result

    def __getitem__(self, index):
        image, segmentation_map = super(DentalSegmentationDetectionDataset, self).__getitem__( index)
        mask = segmentation_map.clone().numpy()
        points = []
        for m in range(1, mask.shape[0]):
            if mask[m].mean() > 0.0:
                segment = mask[m]*255.0
                contours_hier = cv2.findContours(segment.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cnts_segs = imutils.grab_contours(contours_hier)
                boundary_segs = max(cnts_segs, key=cv2.contourArea)
                local_points = self.pointate(boundary_segs, self.resolution)
            else:
                local_points = np.zeros(shape=(5,2)).astype(np.float32)
            points.append( local_points)

        data = np.concatenate(points)
        boundary_points = torch.from_numpy(data.astype(np.float32)).view(-1)

        return image, segmentation_map, boundary_points


class DentalSegmentationFolderDataset(Dataset):
    def __init__(self, data_dir, split : str = 'train', resolution : int = 256, points : bool = False):   # initial logic happens like transform
        self.resolution = resolution
        self.image_paths = [os.path.join(data_dir + '/images/', x) for x in os.listdir(data_dir + '/images/') if is_image_file(x)]
        self.image_paths.sort()
        self.mask_paths = [os.path.join(data_dir + '/maskes/', x) for x in os.listdir(data_dir + '/maskes/') if is_image_file(x)]
        self.mask_paths.sort()
        self.point_paths = None
        random_inst = random.Random(42)
        n_items = len(self.image_paths)
        idxs = random_inst.sample(range(n_items), n_items // 5)
        self.split = split

        if self.split == 'train': idxs = [idx for idx in range(n_items) if idx not in idxs]

        self.image_paths = [self.image_paths[i] for i in idxs]
        self.mask_paths = [self.mask_paths[i] for i in idxs]
        self.image_paths.sort()
        self.mask_paths.sort()

        if points:
            self.point_paths =  [os.path.join(data_dir + '/points/', x) for x in os.listdir(data_dir + '/points/')]
            self.point_paths = [self.point_paths[i] for i in idxs]
            self.point_paths.sort()

        self.tensor_transformation = transforms.ToTensor()

    def __getitem__(self, index):
        image = load_image(self.image_paths[index], channels = INPUT_DIMENSION, resolution = self.resolution)
        image = self.tensor_transformation(image)
        data = load_image(self.mask_paths[index], resolution = self.resolution)
        masks = [(data == id) for id in TOOTH_ID]
        mask = np.stack(masks, axis=2).astype('float32')
        segmentation_map = torch.from_numpy(mask).long().permute(2, 0, 1)
        result = (image, segmentation_map)

        if self.point_paths is not None:
            points = np.fromfile(self.point_paths[index], dtype = np.int32).reshape(256 * 2)
            points = torch.from_numpy(np.array(points)).div(float(256 * 2))
            result = result + (points,)

        return result

    def __len__(self):
        return len(self.image_paths)
from torch.utils.data import Dataset
import numpy as np
import cv2, torch
import os
import glob
import lmdb
import pickle
from collections import OrderedDict
import random


class TrainDataset(Dataset):
    def __init__(self, data_path, max_points=2048, keypoint='superpoint'):
        self.width = 512
        self.height = 512
        self.max_points = max_points 
        self.train_path = data_path
        self.keypoint = keypoint
        self.lmdb_path = os.path.join(data_path, f'{self.keypoint}_lmdb')
#        self.lmdb_path = os.path.join(data_path, f'opencv_{self.keypoint.lower()}_lmdb')
        self.datas = OrderedDict()
        
        self.descriptor_dim = {
            'matchanything': 128,
            'superglue': 256, 
            'superpoint': 256, 
            'SIFT': 128,      # SIFT: 128 dim
            'SURF': 128,      # SURF: 128 dim
            'ORB': 32,        # ORB: 32 dim
            'FAST': 32        # FAST+ORB: 32 dim
        }
#        if 'matchanything' in self.keypoint:
#          self.max_points = 5000
        
        #  LMDB 
        self.env = None
        self.txn = None
        if self.lmdb_path and os.path.exists(self.lmdb_path):
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
            self.txn = self.env.begin(write=False)
            print(f"Loaded LMDB database from {self.lmdb_path}")
        
        datas = glob.glob(os.path.join(self.train_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['image'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['image'].sort()
        
        print("Dataset keys:", self.datas.keys())
        print("Total samples:", len(self))


    def _get_lmdb_data(self, index):
        if self.txn is None:
            return None, None, None, None
        
        try:
            image_name = os.path.basename(self.datas['input1']['image'][index])

            key1 = f'{index:08d}'.encode()
            value1 = self.txn.get(key1)
            
            key2 = f'{index:08d}_{image_name}'.encode()
            value2 = self.txn.get(key2)
            
            key3 = f'{self.keypoint}_{index:08d}_{image_name}'.encode()
            value3 = self.txn.get(key3)
            
#            key4 = f'{index:08d}_{[image_name]}'.encode()
#            value4 = self.txn.get(key4)

            key4 = f'{index:08d}_[\'0\']'.encode()
            value4 = self.txn.get(key4)
            
#            print(key1,key2,key3,key4)
            if value1:
                data = pickle.loads(value1)
            elif value2:
                data = pickle.loads(value2)
            elif value3:
                data = pickle.loads(value3)
            elif value4:
                data = pickle.loads(value4)
            else:
                print(f"Warning: No LMDB data found for index {index}, image {image_name}")
                return None, None, None, None
            
            points0 = data.get('keypoints0', None)
            points1 = data.get('keypoints1', None)
            descriptors0 = data.get('descriptors0', None)
            descriptors1 = data.get('descriptors1', None)
            
            if points0 is not None:
                points0 = torch.from_numpy(points0) if isinstance(points0, np.ndarray) else points0
            if points1 is not None:
                points1 = torch.from_numpy(points1) if isinstance(points1, np.ndarray) else points1
            if descriptors0 is not None:
                descriptors0 = torch.from_numpy(descriptors0) if isinstance(descriptors0, np.ndarray) else descriptors0
            if descriptors1 is not None:
                descriptors1 = torch.from_numpy(descriptors1) if isinstance(descriptors1, np.ndarray) else descriptors1
            
            return points0, points1, descriptors0, descriptors1
            
        except Exception as e:
            print(f"Error reading LMDB data for index {index}: {e}")
            return None, None, None, None

    def _create_dummy_data(self, num_points=1000):
        descriptor_dim = self.descriptor_dim[self.keypoint]
        points0 = torch.rand(num_points, 2) * torch.tensor([self.width, self.height]).float()
        points1 = torch.rand(num_points, 2) * torch.tensor([self.width, self.height]).float()
        descriptors0 = torch.randn(num_points, descriptor_dim)
        descriptors1 = torch.randn(num_points, descriptor_dim)
        return points0 *0, points1 *0, descriptors0 *0, descriptors1 *0
    
    def _pad_or_truncate_points(self, points, descriptors, target_num=2000):
        current_num = points.shape[0]
        descriptor_dim = self.descriptor_dim[self.keypoint]
        
        if current_num == target_num:
            return points, descriptors
        
        elif current_num < target_num:
            if current_num == 0 or points is None:
              padded_points = torch.zeros((target_num, 2), dtype=torch.float)
              padded_descriptors = torch.zeros((target_num, descriptor_dim), dtype=torch.float)
            else:
              pad_num = target_num - current_num
              repeat_indices = torch.randint(0, current_num, (pad_num,))
              
              padded_points = torch.cat([points, points[repeat_indices]], dim=0)
              padded_descriptors = torch.cat([descriptors, descriptors[repeat_indices]], dim=0)
            
            return padded_points, padded_descriptors
        
        else:
            keep_indices = torch.arange(target_num)  
            truncated_points = points[keep_indices]
            truncated_descriptors = descriptors[keep_indices]
            
            return truncated_points, truncated_descriptors
    
                  
    def __getitem__(self, index):
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        size1 = input1.shape
        input1 = cv2.resize(input1, (self.width, self.height))
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        input1 = np.transpose(input1, [2, 0, 1])
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        size2 = input2.shape
        input2 = cv2.resize(input2, (self.width, self.height))
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        input2 = np.transpose(input2, [2, 0, 1])
        
        point1, point2, des1, des2 = self._get_lmdb_data(index)
#        print(point1.shape, point2.shape)
        
        if point1 is None:
            point1, point2, des1, des2 = self._create_dummy_data()
            print(f"Using dummy data for index {index}")
        

        point1_padded, des1_padded = self._pad_or_truncate_points(point1, des1, self.max_points)
        point2_padded, des2_padded = self._pad_or_truncate_points(point2, des2, self.max_points)
        
#        print(f"Index {index}: Original points {point1.shape} -> Padded points {point1_padded.shape}")
#        print(f"Index {index}: Original des1 {des1.shape} -> Padded des1_padded {des1_padded.shape}")

        point1_padded[...,0] = point1_padded[...,0] / (size1[1] - 1)
        point1_padded[...,1] = point1_padded[...,1] / (size1[0] - 1)
        point2_padded[...,0] = point2_padded[...,0] / (size2[1] - 1)
        point2_padded[...,1] = point2_padded[...,1] / (size2[0] - 1)
        
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        point1_tensor = point1_padded.float()
        point2_tensor = point2_padded.float()
        des1_tensor = des1_padded.float()
        des2_tensor = des2_padded.float()
        
        if_exchange = random.randint(0, 1)
        if if_exchange == 0:
            return (input1_tensor, input2_tensor, point1_tensor, point2_tensor, des1_tensor, des2_tensor)
        else:
            return (input2_tensor, input1_tensor, point2_tensor, point1_tensor, des2_tensor, des1_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])
    
    def __del__(self):
        if self.env:
            self.env.close()


class TestDataset(Dataset):
    def __init__(self, data_path,  max_points=2048, keypoint='superpoint', is_finetune=False):
        self.width = 512
        self.height = 512
        self.max_points = max_points
        self.test_path = data_path
        self.is_finetune = is_finetune
        self.keypoint = keypoint
        self.lmdb_path = os.path.join(data_path, f'{self.keypoint}_lmdb')
#        self.lmdb_path = os.path.join(data_path, f'opencv_{self.keypoint.lower()}_lmdb')
        self.datas = OrderedDict()
        self.extensions = ['*.png', '*.jpg', '*.PNG', '*.JPG', '*.jpeg', '*.JPEG']
        
        self.descriptor_dim = {
            'matchanything': 128,
            'superglue': 256, 
            'superpoint': 256, 
            'SIFT': 128,      # SIFT: 128 dim
            'SURF': 128,       # SURF: 128 dim
            'ORB': 32,        # ORB: 32 dim
            'FAST': 32        # FAST+ORB: 32 dim
        }
#        if 'matchanything' in self.keypoint:
#          self.max_points = 5000
        
        self.env = None
        self.txn = None
        if self.lmdb_path and os.path.exists(self.lmdb_path):
            self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False, readahead=False)
            self.txn = self.env.begin(write=False)
            print(f"Loaded LMDB database from {self.lmdb_path}")
        
        datas = glob.glob(os.path.join(self.test_path, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input1' or data_name == 'input2':
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                full_img_list = []
                for ex in self.extensions:
                    full_img_list.extend(glob.glob(os.path.join(data, ex)))
               
                self.datas[data_name]['image'] = full_img_list
                self.datas[data_name]['image'].sort()
        print("Test dataset keys:", self.datas.keys())
        print("Total test samples:", len(self))

    def _pad_or_truncate_points(self, points, descriptors, target_num=2000):
        current_num = points.shape[0]
        descriptor_dim = self.descriptor_dim[self.keypoint]
        
        if current_num == target_num:
            return points, descriptors
        
        elif current_num < target_num:
            if current_num == 0 or points is None:
              padded_points = torch.zeros((target_num, 2), dtype=torch.float)
              padded_descriptors = torch.zeros((target_num, descriptor_dim), dtype=torch.float)
            else:
              pad_num = target_num - current_num
              repeat_indices = torch.randint(0, current_num, (pad_num,))
              
              padded_points = torch.cat([points, points[repeat_indices]], dim=0)
              padded_descriptors = torch.cat([descriptors, descriptors[repeat_indices]], dim=0)
            
            return padded_points, padded_descriptors
        
        else:
            keep_indices = torch.arange(target_num)  
            truncated_points = points[keep_indices]
            truncated_descriptors = descriptors[keep_indices]
            
            return truncated_points, truncated_descriptors
    
        
    def _get_lmdb_data(self, index):
        if self.txn is None:
            return None, None, None, None
        
        try:
            image_name = os.path.basename(self.datas['input1']['image'][index])
        
            key1 = f'{index:08d}'.encode()
            value1 = self.txn.get(key1)
            
            key2 = f'{index:08d}_{image_name}'.encode()
            value2 = self.txn.get(key2)
            
            key3 = f'{self.keypoint}_{index:08d}_{image_name}'.encode()
            value3 = self.txn.get(key3)
            
#            key4 = f'{index:08d}_{[image_name]}'.encode()
#            value4 = self.txn.get(key4)

            key4 = f'{index:08d}_[\'0\']'.encode()
            value4 = self.txn.get(key4)
            
#            print(key1,key2,key3,key4)
            if value1:
                data = pickle.loads(value1)
            elif value2:
                data = pickle.loads(value2)
            elif value3:
                data = pickle.loads(value3)
            elif value4:
                data = pickle.loads(value4)
            else:
                print(f"Warning: No LMDB data found for index {index}, image {image_name}")
                return None, None, None, None
            
            points0 = data.get('keypoints0', None)
            points1 = data.get('keypoints1', None)
            descriptors0 = data.get('descriptors0', None)
            descriptors1 = data.get('descriptors1', None)
           
            if points0 is not None:
                points0 = torch.from_numpy(points0) if isinstance(points0, np.ndarray) else points0
            if points1 is not None:
                points1 = torch.from_numpy(points1) if isinstance(points1, np.ndarray) else points1
            if descriptors0 is not None:
                descriptors0 = torch.from_numpy(descriptors0) if isinstance(descriptors0, np.ndarray) else descriptors0
            if descriptors1 is not None:
                descriptors1 = torch.from_numpy(descriptors1) if isinstance(descriptors1, np.ndarray) else descriptors1
            
            return points0, points1, descriptors0, descriptors1
            
        except Exception as e:
            print(f"Error reading LMDB data for test index {index}: {e}")
            return None, None, None, None

    def _create_dummy_data(self, num_points=1000):
        descriptor_dim = self.descriptor_dim[self.keypoint]
        points0 = torch.rand(num_points, 2) * torch.tensor([self.width, self.height]).float()
        points1 = torch.rand(num_points, 2) * torch.tensor([self.width, self.height]).float()
        descriptors0 = torch.randn(num_points, descriptor_dim)
        descriptors1 = torch.randn(num_points, descriptor_dim)
        return points0 *0, points1 *0, descriptors0 *0, descriptors1 *0

    def __getitem__(self, index):
        # load image1
        input1 = cv2.imread(self.datas['input1']['image'][index])
        size1 = input1.shape
        input1 = input1.astype(dtype=np.float32)
        input1 = (input1 / 127.5) - 1.0
        
        # load image2
        input2 = cv2.imread(self.datas['input2']['image'][index])
        size2 = input2.shape
        input2 = input2.astype(dtype=np.float32)
        input2 = (input2 / 127.5) - 1.0
        
        if input1.shape != input2.shape:
            input2 = cv2.resize(input2, (input1.shape[1], input1.shape[0]), interpolation=cv2.INTER_AREA)
            
        input1 = np.transpose(input1, [2, 0, 1])
        input2 = np.transpose(input2, [2, 0, 1])
        
        point1, point2, des1, des2 = self._get_lmdb_data(index)
        
        if point1 is None:
            point1, point2, des1, des2 = self._create_dummy_data()
            print(f"Using dummy data for test index {index}")
  
        point1_padded, des1_padded = self._pad_or_truncate_points(point1, des1, self.max_points) #   point1, des1 # 
        point2_padded, des2_padded = self._pad_or_truncate_points(point2, des2, self.max_points) # point2, des2 # 
#        print(point1.shape, point1_padded.shape, point2.shape, point2_padded.shape)
       
        point1_padded[...,0] = point1_padded[...,0] / (size1[1] - 1)
        point1_padded[...,1] = point1_padded[...,1] / (size1[0] - 1)
        point2_padded[...,0] = point2_padded[...,0] / (size2[1] - 1)
        point2_padded[...,1] = point2_padded[...,1] / (size2[0] - 1)
      
        # convert to tensor
        input1_tensor = torch.tensor(input1)
        input2_tensor = torch.tensor(input2)
        point1_tensor = point1_padded.float()
        point2_tensor = point2_padded.float()
        des1_tensor = des1_padded.float()
        des2_tensor = des2_padded.float()
        
        if self.is_finetune:
          img_name = self.datas['input1']['image'][index]
          return (input1_tensor, input2_tensor, point1_tensor, point2_tensor, des1_tensor, des2_tensor, img_name)
        else:
          return (input1_tensor, input2_tensor, point1_tensor, point2_tensor, des1_tensor, des2_tensor)

    def __len__(self):
        return len(self.datas['input1']['image'])
    
    def __del__(self):
        if self.env:
            self.env.close()

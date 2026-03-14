#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import lmdb
import pickle
import gc
from pathlib import Path
import argparse

def extract_features_and_matches(image0, image1, algorithm='SIFT'):
   # Select detector and matcher based on algorithm
   if algorithm == 'SIFT':
       detector = cv2.SIFT_create()
       norm_type = cv2.NORM_L2
       matcher = cv2.BFMatcher(norm_type)
       
   elif algorithm == 'SURF':
       # Ensure opencv-contrib-python is installed
       try:
           detector = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
       except:
           detector = cv2.SIFT_create()  # Fallback to SIFT
       norm_type = cv2.NORM_L2
       matcher = cv2.BFMatcher(norm_type)
       
   elif algorithm == 'ORB':
       detector = cv2.ORB_create(nfeatures=2000)
       norm_type = cv2.NORM_HAMMING
       matcher = cv2.BFMatcher(norm_type)
       
   elif algorithm == 'FAST':
       # FAST only detects keypoints, need other descriptor
       fast = cv2.FastFeatureDetector_create()
       # Use ORB as descriptor
       orb = cv2.ORB_create()
       norm_type = cv2.NORM_HAMMING
       matcher = cv2.BFMatcher(norm_type)
       
       # Detect keypoints
       kp0 = fast.detect(image0, None)
       kp1 = fast.detect(image1, None)
       
       # Compute descriptors
       kp0, desc0 = orb.compute(image0, kp0)
       kp1, desc1 = orb.compute(image1, kp1)
       
   else:
       raise ValueError(f"not support this algorithm: {algorithm}")
   
   # For SIFT, SURF, ORB algorithms
   if algorithm in ['SIFT', 'SURF', 'ORB']:
       # Detect keypoints and compute descriptors
       kp0, desc0 = detector.detectAndCompute(image0, None)
       kp1, desc1 = detector.detectAndCompute(image1, None)
   
   # Convert to numpy arrays for storage
   def keypoints_to_numpy(keypoints):
       points = np.array([kp.pt for kp in keypoints], dtype=np.float32)
       if len(keypoints) > 0:
           sizes = np.array([kp.size for kp in keypoints], dtype=np.float32)
           angles = np.array([kp.angle for kp in keypoints], dtype=np.float32)
           return points, sizes, angles
       return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
   
   # Feature matching
   matches = []
   if desc0 is not None and desc1 is not None and len(desc0) > 0 and len(desc1) > 0:
       try:
           # Use KNN matching
           knn_matches = matcher.knnMatch(desc0, desc1, k=2)
           
           # Apply Lowe's ratio test
           good_matches = []
           for m, n in knn_matches:
               if m.distance < 0.75 * n.distance:
                   good_matches.append(m)
           
           # Convert to numpy array
           if good_matches:
               matches = np.array([(m.queryIdx, m.trainIdx, m.distance) 
                                  for m in good_matches], 
                                 dtype=[('queryIdx', 'i4'), ('trainIdx', 'i4'), ('distance', 'f4')])
       except Exception as e:
           print(f"match error: {e}")
           matches = np.array([], dtype=[('queryIdx', 'i4'), ('trainIdx', 'i4'), ('distance', 'f4')])
   else:
       matches = np.array([], dtype=[('queryIdx', 'i4'), ('trainIdx', 'i4'), ('distance', 'f4')])
   
   # Convert to numpy arrays
   kp0_points, kp0_sizes, kp0_angles = keypoints_to_numpy(kp0)
   kp1_points, kp1_sizes, kp1_angles = keypoints_to_numpy(kp1)
   
#    # Prepare data for storage
#    data = {
#        'algorithm': algorithm,
#        'keypoints0': kp0_points,       # [N, 2] keypoint coordinates
#        'keypoints1': kp1_points,       # [M, 2]
#        'descriptors0': desc0 if desc0 is not None else np.zeros((0, 128), dtype=np.float32),
#        'descriptors1': desc1 if desc1 is not None else np.zeros((0, 128), dtype=np.float32),
#        'matches': matches,             # matching pairs array
#        'num_matches': len(matches),
#        'num_keypoints0': len(kp0_points),
#        'num_keypoints1': len(kp1_points)
#    }


   # Prepare data for storage
   if len(matches) > 0:
       query_indices = matches['queryIdx']  
       train_indices = matches['trainIdx'] 
       if len(query_indices) == 0 or len(train_indices) == 0:
         print('match key point is zeors !!!!')
       

       matched_kp0 = kp0_points[query_indices]  
       matched_kp1 = kp1_points[train_indices]
       matched_desc0 = desc0[query_indices]
       matched_desc1 = desc1[train_indices]
       
       data = {
           'algorithm': algorithm,
           'keypoints0': matched_kp0,
           'keypoints1': matched_kp1,
           'descriptors0': matched_desc0,
           'descriptors1': matched_desc1,
           'num_matches': len(matches),
           'num_keypoints0': len(matched_kp0),
           'num_keypoints1': len(matched_kp1),
       }
   else:
#        print('no matching points, using original array to padding')

       if len(kp0_points) == 0 or len(kp1_points) == 0:
         print('key point is zeors !!!!')
       

       matched_kp0 = kp0_points  
       matched_kp1 = kp1_points
       matched_desc0 = desc0
       matched_desc1 = desc1
       data = {
           'algorithm': algorithm,
           'keypoints0': matched_kp0,
           'keypoints1': matched_kp1,
           'descriptors0': matched_desc0,
           'descriptors1': matched_desc1,
           'num_matches': len(matches),
           'num_keypoints0': len(matched_kp0),
           'num_keypoints1': len(matched_kp1),
#            'all_keypoints0': kp0_points,
#            'all_keypoints1': kp1_points,
#            'all_descriptors0': desc0,
#            'all_descriptors1': desc1
       }
   
   return data

def load_image_opencv(image_path):
   img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
   if img is None:
       raise ValueError(f"cannot load this image: {image_path}")
   
   return img

def process_opencv_to_lmdb(base_path, lmdb_path, algorithm='SIFT', 
                         batch_size=10, map_size_gb=100):
   print(f"using {algorithm} algorithm to process images...")
   
   # Path handling
   base_path = Path(base_path)
   lmdb_path = Path(lmdb_path)
   lmdb_path.parent.mkdir(parents=True, exist_ok=True)
   
   # Get image pair list
   input1_dir = base_path / 'input1'
   input2_dir = base_path / 'input2'
   
   # Supported image formats
   image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
   
   name_list = sorted([f.name for f in input1_dir.iterdir() 
                      if f.suffix.lower() in image_extensions])
   
   print(f'found {len(name_list)} image pairs')
   
   # Create LMDB environment
   map_size = map_size_gb * 1024 * 1024 * 1024  # Convert to bytes
   env = lmdb.open(str(lmdb_path), map_size=map_size, 
                  max_readers=1024, readonly=False)
   
   stats = {
       'processed': 0,
       'failed': 0,
       'total_keypoints': 0,
       'total_matches': 0
   }
   
   with env.begin(write=True) as txn:
       for batch_start in range(0, len(name_list), batch_size):
           batch_end = min(batch_start + batch_size, len(name_list))
           batch_names = name_list[batch_start:batch_end]
           
           batch_num = batch_start // batch_size + 1
           total_batches = (len(name_list) + batch_size - 1) // batch_size
#            print(f'\nProcessing batch {batch_num}/{total_batches}')
           
           for idx, name in enumerate(batch_names):
               global_idx = batch_start + idx
               
               try:
                   # Load images
                   path1 = input1_dir / name
                   path2 = input2_dir / name
                   
                   # Check if files exist
                   if not path1.exists() or not path2.exists():
                       print(f"Warning: incomplete image pair {name}, skipping")
                       stats['failed'] += 1
                       continue
                   
#                    print(f"Processing: {name} ({global_idx+1}/{len(name_list)})")
                   
                   # Load images
                   image0 = load_image_opencv(str(path1))
                   image1 = load_image_opencv(str(path2))
                   
                   # Extract features and matches
                   data = extract_features_and_matches(image0, image1, algorithm)
                   
                   # Update statistics
                   stats['total_keypoints'] += data['num_keypoints0'] + data['num_keypoints1']
                   stats['total_matches'] += data['num_matches']
                   
                   # Add metadata
                   data['image_name'] = name
                   data['image_shape0'] = image0.shape
                   data['image_shape1'] = image1.shape
                   data['global_idx'] = global_idx
                   
                   # Save to LMDB
#                    key = f"{algorithm}_{global_idx:08d}_{name}".encode()
                   key = f"{global_idx:08d}_{name}".encode()
#                    print('key:',key)
                   value = pickle.dumps(data)
                   txn.put(key, value)
                   
                   stats['processed'] += 1
                   
                   # Clean memory
                   del image0, image1, data
                   
#                    # Print progress every 5 images
#                    if (global_idx + 1) % 5 == 0:
#                        print(f"  Processed {global_idx + 1}/{len(name_list)}")
                       
               except Exception as e:
                   print(f"Error processing {name}: {e}")
                   stats['failed'] += 1
                   continue
           
           # Clean memory after each batch
           gc.collect()
           
           # Print current statistics
           avg_keypoints = stats['total_keypoints'] / max(stats['processed'], 1) / 2
           avg_matches = stats['total_matches'] / max(stats['processed'], 1)
#            print(f"  Stats: Processed {stats['processed']}, Failed {stats['failed']}")
#            print(f"  Average keypoints per image: {avg_keypoints:.1f}")
#            print(f"  Average matches per pair: {avg_matches:.1f}")
   
   env.close()
   
   # Print final statistics
   print(f"\n{'='*50}")
   print(f"Processing completed!")
   print(f"Algorithm: {algorithm}")
   print(f"Successfully processed: {stats['processed']}")
   print(f"Failed: {stats['failed']}")
   print(f"Total keypoints: {stats['total_keypoints']}")
   print(f"Total matches: {stats['total_matches']}")
   print(f"Database location: {lmdb_path}")
   print(f"{'='*50}")
   
   return stats

if __name__ == "__main__":
   # Process all algorithms
   algorithms = ['SIFT', 'ORB', 'SURF', 'FAST']
   
   for algo in algorithms:
       try:
           print(f"\n{'#'*60}")
           print(f"Starting {algo} algorithm processing")
           print(f"{'#'*60}")
           
           lmdb_path = f'/data/my_files/Datasets/UDIS/training/opencv_{algo.lower()}_lmdb'
           
           stats = process_opencv_to_lmdb(
               base_path='/data/my_files/Datasets/UDIS/training/',
               lmdb_path=lmdb_path,
               algorithm=algo,
               batch_size=10,
               map_size_gb=60  # 60GB for each algorithm
           )
           
       except Exception as e:
           print(f"Error processing {algo} algorithm: {e}")
           continue


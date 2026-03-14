# coding: utf-8
import argparse
import torch
from collections import OrderedDict
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim

import cv2
#from torch_homography_model import build_model
from network import get_stitched_result, Network, build_new_ft_model
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import glob
from loss import cal_lp_loss2
import torchvision.transforms as T
from dataset import *
from torch.utils.data import DataLoader

import random
import torch.backends.cudnn as cudnn
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark =  False

setup_seed(114514)

#import PIL
resize_512 = T.Resize((512,512))

# path of project
last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))


#nl: path to save the model files
MODEL_COEF_DIR = os.path.join(last_path, 'model_homo_stage2')


#nl: create folders if it dose not exist
# if not os.path.exists(MODEL_COEF_DIR):
#     os.makedirs(MODEL_COEF_DIR)

def ensure_minimum_size(image, min_size=7):
    h, w = image.shape[:2]
    
    if h >= min_size and w >= min_size:
#        print('original image size:{}'.format(image.shape))
        return image
    
    pad_h = max(0, min_size - h)
    pad_w = max(0, min_size - w)
    
    if len(image.shape) == 2: 
        padded = np.pad(image, 
                       ((pad_h//2, pad_h - pad_h//2),
                        (pad_w//2, pad_w - pad_w//2)),
                       mode='constant')
    else:  
        padded = np.pad(image,
                       ((pad_h//2, pad_h - pad_h//2),
                        (pad_w//2, pad_w - pad_w//2),
                        (0, 0)),
                       mode='constant')
    
    print('original image size:{}, padded size:{}'.format(image.shape, padded.shape))
    return padded
    

def maskSSIM(image1, image2, mask):
    image1 = ensure_minimum_size(image1, min_size=7)
    image2 = ensure_minimum_size(image2, min_size=7)
    mask = ensure_minimum_size(mask, min_size=7)
    image1 = image1 * mask
    image2 = image2 * mask
    _, ssim = compare_ssim(image1, image2, data_range=255, channel_axis=2, full=True)
    ssim = np.sum(ssim * mask) / (np.sum(mask) + 1e-6)
    return ssim

def maskPSNR(image1, image2, mask):
    image1 = image1 * mask / 255
    image2 = image2 * mask / 255
    rmse = np.sqrt(np.sum((image1 - image2) ** 2) / mask.sum())
    psnr = 20 * np.log10(1 / rmse)
    return psnr


def test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt):
    with torch.no_grad():
        output,check_flag = get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt, max_out_height=8000)

    if not check_flag:
        print("image idx:{}, warp size is too huge! pass this image.")
        output = {}
        mask = torch.ones_like(input1_tensor).to(input1_tensor.device)
        output['output_ref'] =  torch.cat((input1_tensor + 1, mask), 1) 
        output['output_tgt'] =  torch.cat((input2_tensor + 1, mask), 1) 
        img1 = output['output_ref'][0,0:3,...]
        img2 = output['output_tgt'][0,0:3,...]
        output['stitched']  =  img1*(img1/(img1+img2+1e-6)) + img2*(img2/(img1+img2+1e-6))


    output_tps_ref = output['output_ref']
    output_tps_tgt = output['output_tgt']
    stitch_result = output['stitched'].cpu().detach().numpy().transpose(1,2,0) *127.5
    warp1 = output_tps_ref[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5
    warp2 = output_tps_tgt[0,0:3,...].cpu().detach().numpy().transpose(1,2,0)  *127.5

    overlap_mask = output_tps_ref[0,3:6,...]  * output_tps_tgt[0,3:6,...]
    overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
    ssim3 = maskSSIM(warp1, warp2, overlap_mask)
    psnr3 = maskPSNR(warp1, warp2, overlap_mask)
    return ssim3,psnr3,stitch_result


def train(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # define the network
    net = Network()
    if torch.cuda.is_available():
        net = net.cuda()
        
    # classical dataset
    test_other_data = TestDataset(data_path=args.test_other_path, is_finetune=True)
    #nl: set num_workers = the number of cpus
    test_other_loader = DataLoader(dataset=test_other_data, batch_size=1, num_workers=1, shuffle=False, drop_last=False)


    # define the optimizer and learning rate
    optimizer = optim.Adam(net.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-08)  # default as 0.0001
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)

    #load the existing models if they are exist
    # homo model
    homo_ckpt_list = glob.glob(MODEL_COEF_DIR + "/*.pth")
    homo_ckpt_list.sort()
    if len(homo_ckpt_list) != 0:
        homo_model_path = homo_ckpt_list[-1]
        homo_checkpoint = torch.load(homo_model_path)

        net.load_state_dict(homo_checkpoint['model'])
        print('load homo model from {}!'.format(homo_model_path))
    else:
        start_epoch = 0
        print('training homo from stratch!')

    fintune_psnr_list = []
    fintune_ssim_list = []

    path_ave_other_fusion = '../ave_other_fusion_finetune/'
    if not os.path.exists(path_ave_other_fusion):
        os.makedirs(path_ave_other_fusion)


    for idx, batch_value in enumerate(test_other_loader):
        input1_tensor = batch_value[0].float()
        input2_tensor = batch_value[1].float()
        point1_tensor = batch_value[2].float()
        point2_tensor = batch_value[3].float()
        descriptor1_tensor = batch_value[4].float()
        descriptor2_tensor = batch_value[5].float()
        img_name = batch_value[6][0].split('/')[-1]
        
        if input1_tensor.shape != input2_tensor.shape:
          print('input1 and input2 have different size!')
          continue
          
#        print(img_name)
  
        if torch.cuda.is_available():
          input1_tensor = input1_tensor.cuda()
          input2_tensor = input2_tensor.cuda()
          point1_tensor = point1_tensor.cuda()
          point2_tensor = point2_tensor.cuda()
          descriptor1_tensor = descriptor1_tensor.cuda()
          descriptor2_tensor = descriptor2_tensor.cuda()
        
        torch.cuda.empty_cache()

        input1_tensor_512 = resize_512(input1_tensor)
        input2_tensor_512 = resize_512(input2_tensor)

        loss_list = []
        
        ###
        net.load_state_dict(homo_checkpoint['model'])
        optimizer.load_state_dict(homo_checkpoint['optimizer'])
        start_epoch = homo_checkpoint['epoch']
        scheduler.last_epoch = start_epoch
        print('finetune for a new image, init homo model from {}!'.format(homo_ckpt_list))
        ###

        print("##################start iteration {} #######################".format(img_name))

        best_ssim, best_psnr, best_stitch_result = 0, 0, None
        for epoch in range(start_epoch, start_epoch + args.max_iter):
            net.train()

            optimizer.zero_grad()

            batch_out = build_new_ft_model(net, input1_tensor_512, input2_tensor_512, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor)
            output_tps_ref = batch_out['output_tps_ref']
            output_tps_tgt = batch_out['output_tps_tgt']
            rigid_mesh = batch_out['rigid_mesh']
            mesh_ref = batch_out['mesh_ref']
            mesh_tgt = batch_out['mesh_tgt']

            total_loss = cal_lp_loss2(output_tps_ref, output_tps_tgt)
            total_loss.backward()
            # clip the gradient
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=3, norm_type=2)
            optimizer.step()
            scheduler.step()

            current_iter = epoch-start_epoch+1
            
            loss_list.append(total_loss)
            print("Training: Iteration[{:0>3}/{:0>3}] Total Loss: {:.4f} lr={:.8f}".format(current_iter, args.max_iter, total_loss, optimizer.state_dict()['param_groups'][0]['lr']))


            # init check
            if current_iter == 1:
                ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                print('init ssim:{}, psnr:{}'.format(ssim3,psnr3))
       
                if ssim3 > best_ssim:
                    best_ssim = ssim3
                    best_psnr = psnr3
                    best_stitch_result = stitch_result

            if current_iter >= 4:
                if torch.abs(loss_list[current_iter-4]-loss_list[current_iter-3]) <= 1e-4 and torch.abs(loss_list[current_iter-3]-loss_list[current_iter-2]) <= 1e-4 \
                and torch.abs(loss_list[current_iter-2]-loss_list[current_iter-1]) <= 1e-4:
                    
                    ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                    print('stop early, current ssim:{}, psnr:{}'.format(ssim3,psnr3))
                    
                    if ssim3 > best_ssim:
                        best_ssim = ssim3
                        best_psnr = psnr3
                        best_stitch_result = stitch_result

                    break

            if current_iter == args.max_iter:
                ssim3,psnr3,stitch_result = test_once(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt)
                print('final ssim:{}, psnr:{}'.format(ssim3,psnr3))
          
                if ssim3 > best_ssim:
                    best_ssim = ssim3
                    best_psnr = psnr3
                    best_stitch_result = stitch_result


            torch.cuda.empty_cache()

        print("##################end iteration {} #######################".format(img_name))
        fintune_ssim_list.append(best_ssim)
        fintune_psnr_list.append(best_psnr)
        path = path_ave_other_fusion + str(img_name).zfill(6)
        cv2.imwrite(path, best_stitch_result)

    fintune_psnr_list.sort(reverse = True)
    list_len = len(fintune_psnr_list)
    print("top 30%", np.mean(fintune_psnr_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(fintune_psnr_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(fintune_psnr_list[int(list_len*0.6):]))
    print('average psnr:', np.mean(fintune_psnr_list))
    
    fintune_ssim_list.sort(reverse = True)
    print("top 30%", np.mean(fintune_ssim_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(fintune_ssim_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(fintune_ssim_list[int(list_len*0.6):]))
    print('average ssim:', np.mean(fintune_ssim_list))


if __name__=="__main__":

    print('<==================== setting arguments ===================>\n')

    #nl: create the argument parser
    parser = argparse.ArgumentParser()

    #nl: add arguments
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--max_iter', type=int, default=50)
    parser.add_argument('--test_other_path', type=str, default='/data/my_files/Datasets/stitch_real/')

    #nl: parse the arguments
    args = parser.parse_args()
    print(args)

    #nl: rain
    train(args)


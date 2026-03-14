# coding: utf-8
import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import imageio
from network import build_model, build_output_model, Network
from dataset import *
import os
import cv2
import skimage
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import grid_res
import numpy as np
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

grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

last_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
MODEL_DIR = os.path.join(last_path, 'model_homo_stage2')



def draw_mesh_on_warp(warp, f_local):

    point_radius = 8
    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8

    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):

            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                
                cv2.circle(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), point_radius, point_color, -1)
                cv2.circle(warp, (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_radius, point_color, -1)
            elif i == grid_h:
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
                
                cv2.circle(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), point_radius, point_color, -1)
                cv2.circle(warp, (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_radius, point_color, -1)
            else :
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_color, thickness, lineType)
                cv2.line(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_color, thickness, lineType)
            
                cv2.circle(warp, (int(f_local[i,j,0]), int(f_local[i,j,1])), point_radius, point_color, -1)
                cv2.circle(warp, (int(f_local[i+1,j,0]), int(f_local[i+1,j,1])), point_radius, point_color, -1)
                cv2.circle(warp, (int(f_local[i,j+1,0]), int(f_local[i,j+1,1])), point_radius, point_color, -1)

    return warp
    

def maskSSIM(image1, image2, mask):
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
        

def test(args):

    os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # dataset
    test_data = TestDataset(data_path=args.test_path)
    #nl: set num_workers = the number of cpus
    test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # classical dataset
    test_other_data = TestDataset(data_path=args.test_other_path)
    #nl: set num_workers = the number of cpus
    test_other_loader = DataLoader(dataset=test_other_data, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=False)

    # define the network
    net = Network()#build_model(args.model_name)
    if torch.cuda.is_available():
        net = net.cuda()

    #load the existing model if it exists
    ckpt_list = glob.glob(MODEL_DIR + "/*.pth")
    ckpt_list.sort()

    if len(ckpt_list) != 0:
        model_path = ckpt_list[-1]
        checkpoint = torch.load(model_path)

        net.load_state_dict(checkpoint['model'])
        print('load model from {}!'.format(model_path))
    else:
        print('No checkpoint found!')

    # fuse the repvgg blocks for faster inference
    net.fuse()
    print("##################start testing#######################")
    ############### testing on classical dataset ##############
    # create a folder if it does not exist
    path_ave_other_fusion = '../ave_other_fusion/'
    if not os.path.exists(path_ave_other_fusion):
       os.makedirs(path_ave_other_fusion)
    
    ssim_other_list = []
    psnr3_other_list = []
    net.eval()
    for i, batch_value in enumerate(test_other_loader):
        input1_tensor = batch_value[0].float()
        input2_tensor = batch_value[1].float()
        point1_tensor = batch_value[2].float()
        point2_tensor = batch_value[3].float()
        descriptor1_tensor = batch_value[4].float()
        descriptor2_tensor = batch_value[5].float()
        
        if input1_tensor.shape != input2_tensor.shape:
            print('input1 and input2 have different size!')
            continue
  
        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
            point1_tensor = point1_tensor.cuda()
            point2_tensor = point2_tensor.cuda()
            descriptor1_tensor = descriptor1_tensor.cuda()
            descriptor2_tensor = descriptor2_tensor.cuda()

        with torch.no_grad():
            batch_out, flag_check = build_output_model(net, input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, max_out_height=8000)
  
      
        if not flag_check:
            print("image idx:{}, warp size is too huge {}! use the original image pairs.".format(i+1, batch_out))
            batch_out = {}
            mask = torch.ones_like(input1_tensor).to(input1_tensor.device)
            batch_out['output_tps_ref'] = torch.cat((input1_tensor+1, mask), 1)
            batch_out['output_tps_tgt'] = torch.cat((input2_tensor+1, mask), 1)
    
    
        output_tps_ref = batch_out['output_tps_ref']
        output_tps_tgt = batch_out['output_tps_tgt']
        
        
        # SSIM 3
        output_ref = (output_tps_ref[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
        output_tgt = (output_tps_tgt[0,0:3,...]*127.5).cpu().detach().numpy().transpose(1,2,0)
        overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
        overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
        ssim3 = maskSSIM(output_ref, output_tgt, overlap_mask)
        ssim_other_list.append(ssim3)
        
        psnr3 = maskPSNR(output_ref, output_tgt, overlap_mask)
        psnr3_other_list.append(psnr3)
        
        # fusion = np.zeros_like(output_ref)
        # fusion[...,0] = output_tgt[...,0]
        # fusion[...,1] = output_ref[...,1]*0.5 +  output_tgt[...,1]*0.5
        # fusion[...,2] = output_ref[...,2]
        # ave_fusion = fusion      
        
        ave_fusion = output_ref * (output_ref/ (output_ref+output_tgt+1e-6)) + output_tgt * (output_tgt/ (output_ref+output_tgt+1e-6)) 
        path = path_ave_other_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, ave_fusion)
        
#        if flag_check:
#          f_local = batch_out['mesh_trans_ref'][0]
#          print(f_local.shape,output_ref.shape)
#          ave_fusion = draw_mesh_on_warp(output_ref, f_local)
#          path = path_ave_other_fusion + str(i+1).zfill(6) + ".jpg"
#          cv2.imwrite(path, ave_fusion)
               
#        print('i = {}'.format( i+1))
        print('i = {}, ssim = {}, psnr = {}'.format( i+1, ssim3, psnr3))
        torch.cuda.empty_cache()
    

    psnr3_other_list.sort(reverse = True)
    list_len = len(psnr3_other_list)
    print("top 30%", np.mean(psnr3_other_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(psnr3_other_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(psnr3_other_list[int(list_len*0.6):]))
    print('average psnr:', np.mean(psnr3_other_list))
    
    ssim_other_list.sort(reverse = True)
    print("top 30%", np.mean(ssim_other_list[:int(list_len*0.3)]))
    print("top 30~60%", np.mean(ssim_other_list[int(list_len*0.3):int(list_len*0.6)]))
    print("top 60~100%", np.mean(ssim_other_list[int(list_len*0.6):]))
    print('average ssim:', np.mean(ssim_other_list))

    ############### testing on UDIS-D #################
    path_ave_fusion = '../ave_fusion/'
    if not os.path.exists(path_ave_fusion):
        os.makedirs(path_ave_fusion)


    ssim3_list = []
    psnr3_list = []
    net.eval()
    
    for i, batch_value in enumerate(test_loader):

        input1_tensor = batch_value[0].float()
        input2_tensor = batch_value[1].float()
        point1_tensor = batch_value[2].float()
        point2_tensor = batch_value[3].float()
        descriptor1_tensor = batch_value[4].float()
        descriptor2_tensor = batch_value[5].float()

        if torch.cuda.is_available():
            input1_tensor = input1_tensor.cuda()
            input2_tensor = input2_tensor.cuda()
            point1_tensor = point1_tensor.cuda()
            point2_tensor = point2_tensor.cuda()
            descriptor1_tensor = descriptor1_tensor.cuda()
            descriptor2_tensor = descriptor2_tensor.cuda()


        with torch.no_grad():
            batch_out, flag_check = build_output_model(net, input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, max_out_height=8000)

        if not flag_check:
            print("image idx:{}, warp size is too huge {}! use the original image pairs.".format(i+1, batch_out))
            batch_out = {}
            mask = torch.ones_like(input1_tensor).to(input1_tensor.device)
            batch_out['output_tps_ref'] = torch.cat((input1_tensor+1, mask), 1)
            batch_out['output_tps_tgt'] = torch.cat((input2_tensor+1, mask), 1)


        # result: tps
        output_tps_ref = batch_out['output_tps_ref']
        output_tps_tgt = batch_out['output_tps_tgt']


        # SSIM 3
        output_ref = ((output_tps_ref[0,0:3,...])*127.5).cpu().detach().numpy().transpose(1,2,0)
        output_tgt = ((output_tps_tgt[0,0:3,...])*127.5).cpu().detach().numpy().transpose(1,2,0)
        overlap_mask = output_tps_ref[0,3:6,...] * output_tps_tgt[0,3:6,...]
        overlap_mask = overlap_mask.cpu().detach().numpy().transpose(1,2,0)
        ssim3 = maskSSIM(output_ref, output_tgt, overlap_mask)
        ssim3_list.append(ssim3)

        psnr3 = maskPSNR(output_ref, output_tgt, overlap_mask)
        psnr3_list.append(psnr3)


        # fusion = np.zeros_like(output_ref)
        # fusion[...,0] = output_tgt[...,0]
        # fusion[...,1] = output_ref[...,1]*0.5 +  output_tgt[...,1]*0.5
        # fusion[...,2] = output_ref[...,2]
        # ave_fusion = fusion     

        ave_fusion = output_ref * (output_ref/ (output_ref+output_tgt+1e-6)) + output_tgt * (output_tgt/ (output_ref+output_tgt+1e-6)) 
        path = path_ave_fusion + str(i+1).zfill(6) + ".jpg"
        cv2.imwrite(path, ave_fusion)


#        print('i = {}'.format( i+1))
        print('i = {}, ssim = {}, psnr = {}'.format( i+1, ssim3, psnr3))
        torch.cuda.empty_cache()
    

    psnr3_list.sort(reverse = True)
    psnr3_list_30 = psnr3_list[ : 331]
    psnr3_list_60 = psnr3_list[331: 663]
    psnr3_list_100 = psnr3_list[663: ]
    print("top 30%", np.mean(psnr3_list_30))
    print("top 30~60%", np.mean(psnr3_list_60))
    print("top 60~100%", np.mean(psnr3_list_100))
    print('average psnr:', np.mean(psnr3_list))

    ssim3_list.sort(reverse = True)
    ssim3_list_30 = ssim3_list[ : 331]
    ssim3_list_60 = ssim3_list[331: 663]
    ssim3_list_100 = ssim3_list[663: ]
    print("top 30%", np.mean(ssim3_list_30))
    print("top 30~60%", np.mean(ssim3_list_60))
    print("top 60~100%", np.mean(ssim3_list_100))
    print('average ssim:', np.mean(ssim3_list))
    print("##################end testing#######################")



if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--test_path', type=str, default='/data/my_files/Datasets/UDIS/testing/')
    parser.add_argument('--test_other_path', type=str, default='/media/my123/datasets/ClassicalDataset/stitch_real/')


    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)
    test(args)

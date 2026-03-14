#
import torch
import torch.nn as nn
import utils.torch_DLT as torch_DLT
import utils.torch_homo_transform as torch_homo_transform
#import utils.torch_tps_transform as torch_tps_transform
import utils.torch_ffd_transform as torch_tps_transform
#import utils.torch_new_transform as torch_tps_transform
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.models as models
import torchvision.transforms as T
import random
from collections import namedtuple


import grid_res
grid_h = grid_res.GRID_H
grid_w = grid_res.GRID_W

import torchvision.transforms as T
resize_512 = T.Resize((512,512))


#Covert global homo into mesh
def H2Mesh(H, rigid_mesh):

    H_inv = torch.inverse(H)
    ori_pt = rigid_mesh.reshape(rigid_mesh.size()[0], -1, 2)
    ones = torch.ones(rigid_mesh.size()[0], (grid_h+1)*(grid_w+1),1)
    if torch.cuda.is_available():
        ori_pt = ori_pt.cuda()
        ones = ones.cuda()

    ori_pt = torch.cat((ori_pt, ones), 2) # bs*(grid_h+1)*(grid_w+1)*3
    tar_pt = torch.matmul(H_inv, ori_pt.permute(0,2,1)) # bs*3*(grid_h+1)*(grid_w+1)

    mesh_x = torch.unsqueeze(tar_pt[:,0,:]/tar_pt[:,2,:], 2)
    mesh_y = torch.unsqueeze(tar_pt[:,1,:]/tar_pt[:,2,:], 2)
    mesh = torch.cat((mesh_x, mesh_y), 2).reshape([rigid_mesh.size()[0], grid_h+1, grid_w+1, 2])

    return mesh

# get rigid mesh
def get_rigid_mesh(batch_size, height, width):

    ww = torch.matmul(torch.ones([grid_h+1, 1]), torch.unsqueeze(torch.linspace(0., float(width), grid_w+1), 0))
    hh = torch.matmul(torch.unsqueeze(torch.linspace(0.0, float(height), grid_h+1), 1), torch.ones([1, grid_w+1]))
    if torch.cuda.is_available():
        ww = ww.cuda()
        hh = hh.cuda()

    ori_pt = torch.cat((ww.unsqueeze(2), hh.unsqueeze(2)),2) # (grid_h+1)*(grid_w+1)*2
    ori_pt = ori_pt.unsqueeze(0).expand(batch_size, -1, -1, -1)

    return ori_pt

# normalize mesh from -1 ~ 1
def get_norm_mesh(mesh, height, width):
    batch_size = mesh.size()[0]
    mesh_w = mesh[...,0]*2./float(width) - 1.
    mesh_h = mesh[...,1]*2./float(height) - 1.
    norm_mesh = torch.stack([mesh_w, mesh_h], 3) # bs*(grid_h+1)*(grid_w+1)*2

    return norm_mesh.reshape([batch_size, -1, 2]) # bs*-1*2

    
# for train.py / test.py
def build_model(net, input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, is_training = True, is_stage2=False):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # network
#    if is_training == True:
    if True:
        aug_input1_tensor, aug_input2_tensor = input1_tensor, input2_tensor
        aug_point1_tensor, aug_point2_tensor = point1_tensor, point2_tensor
        aug_descriptor1_tensor, aug_descriptor2_tensor = descriptor1_tensor, descriptor2_tensor
          
        H_motion, mesh_motion_ref, mesh_motion_tgt, df_loss = net(aug_input1_tensor, aug_input2_tensor, aug_point1_tensor, aug_point2_tensor, aug_descriptor1_tensor, aug_descriptor2_tensor, is_stage2)
    else:
        H_motion, mesh_motion_ref, mesh_motion_tgt, df_loss = net(input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, is_stage2)

    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion_ref = mesh_motion_ref.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(-1, grid_h+1, grid_w+1, 2)

    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
    if torch.cuda.is_available():
        src_p = src_p.cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    # solve homo using DLT
    H = torch_DLT.tensor_DLT(src_p, dst_p)

    M_tensor = torch.tensor([[img_w / 2.0, 0., img_w / 2.0],
                      [0., img_h / 2.0, img_h / 2.0],
                      [0., 0., 1.]]).cuda()
    M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
    M_tensor_inv = torch.inverse(M_tensor)
    M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)
    mask = torch.ones_like(input2_tensor).cuda()

    ######## begin: differential bidirectional decomposition #######
    dst_p_tgt = src_p + (H_motion/2.)
    H_tgt = torch_DLT.tensor_DLT(src_p, dst_p_tgt)
    H_ref = torch.matmul(torch.inverse(H), H_tgt)

    # normalization
    H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref), M_tile)
    H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt), M_tile)

    output_H_ref = torch_homo_transform.transformer(torch.cat((input1_tensor, mask), 1), H_mat_ref, (img_h, img_w))
    output_H_tgt = torch_homo_transform.transformer(torch.cat((input2_tensor, mask), 1), H_mat_tgt, (img_h, img_w))

    ##### stage 2 ####
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    mesh_ref = ini_mesh_ref + mesh_motion_ref
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)
    mesh_tgt = ini_mesh_tgt + mesh_motion_tgt
    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    norm_mesh_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)

    output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor, mask), 1), norm_mesh_ref, norm_rigid_mesh, (img_h, img_w))
    output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor, mask), 1), norm_mesh_tgt, norm_rigid_mesh, (img_h, img_w))


    out_dict = {}
    out_dict.update(df_loss=df_loss)
    out_dict.update(output_H_ref=output_H_ref, output_H_tgt=output_H_tgt)
    out_dict.update(mesh_rigid=rigid_mesh)
    out_dict.update(output_tps_ref = output_tps_ref, mesh_ref = mesh_ref)
    out_dict.update(output_tps_tgt = output_tps_tgt, mesh_tgt = mesh_tgt)


    return out_dict

# for test_output.py
def build_output_model(net, input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, max_out_height=2000):
    batch_size, _, img_h, img_w = input1_tensor.size()

    # input resize
    resized_input1 = resize_512(input1_tensor)
    resized_input2 = resize_512(input2_tensor)
  
    H_motion, mesh_motion_ref, mesh_motion_tgt, df_loss = net(resized_input1, resized_input2, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor)

    # output mesh motion resize
    H_motion = H_motion.reshape(batch_size, 4, 2)
    mesh_motion_ref = mesh_motion_ref.reshape(batch_size, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(batch_size, grid_h+1, grid_w+1, 2)

    H_motion = torch.stack([H_motion[...,0]*img_w/512, H_motion[...,1]*img_h/512], 2)
    mesh_motion_ref = torch.stack([mesh_motion_ref[...,0]*img_w/512, mesh_motion_ref[...,1]*img_h/512], 3)
    mesh_motion_tgt = torch.stack([mesh_motion_tgt[...,0]*img_w/512, mesh_motion_tgt[...,1]*img_h/512], 3)

    ######### warping img1 and img2 to the middle plane ########
    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    dst_p_tgt = src_p + (H_motion/2.)
    H = torch_DLT.tensor_DLT(src_p, dst_p)
    H_tgt = torch_DLT.tensor_DLT(src_p, dst_p_tgt)
    H_ref = torch.matmul(torch.inverse(H), H_tgt)


    # then, calculate the final mesh
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    mesh_ref = ini_mesh_ref + mesh_motion_ref
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)
    mesh_tgt = ini_mesh_tgt + mesh_motion_tgt


    # calculate the size of stitched image
    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0]))
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    out_width = width_max - width_min
    out_height = height_max - height_min

    # in case of the original image resoultion is so huge.
    if max(out_height,out_width) >= max_out_height:
        return None, False
        
    # convert the mesh from [img_h, img_w] to [out_h, out_w]
    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)

    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    # # transformation
    mask = torch.ones_like(input2_tensor).cuda()
    output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    out_dict = {}
    out_dict.update(output_tps_ref=output_tps_ref, output_tps_tgt = output_tps_tgt)
    #out_dict.update(output_H_ref=output_H_ref, output_H_tgt = output_H_tgt)

    return out_dict, True



### for finetune
def build_new_ft_model(net,input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor):
    batch_size, _, img_h, img_w = input1_tensor.size()

    H_motion, mesh_motion_ref, mesh_motion_tgt, df_loss = net(input1_tensor, input2_tensor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor)
    
    H_motion = H_motion.reshape(-1, 4, 2)
    mesh_motion_ref = mesh_motion_ref.reshape(-1, grid_h+1, grid_w+1, 2)
    mesh_motion_tgt = mesh_motion_tgt.reshape(-1, grid_h+1, grid_w+1, 2)
    #mesh_motion = torch.stack([mesh_motion[...,0]*img_w/512, mesh_motion[...,1]*img_h/512], 3)

    ######### warping img1 and img2 to the middle plane ########
    # initialize the source points bs x 4 x 2
    src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]]).cuda()
    src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
    # target points
    dst_p = src_p + H_motion
    dst_p_tgt = src_p + (H_motion/2.)
    H = torch_DLT.tensor_DLT(src_p, dst_p)
    H_tgt = torch_DLT.tensor_DLT(src_p, dst_p_tgt)
    H_ref = torch.matmul(torch.inverse(H), H_tgt)

    # then, calculate the final mesh
    rigid_mesh = get_rigid_mesh(batch_size, img_h, img_w)
    ini_mesh_ref = H2Mesh(H_ref, rigid_mesh)
    mesh_ref = ini_mesh_ref + mesh_motion_ref
    ini_mesh_tgt = H2Mesh(H_tgt, rigid_mesh)
    mesh_tgt = ini_mesh_tgt + mesh_motion_tgt

    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_ref = get_norm_mesh(mesh_ref, img_h, img_w)
    norm_mesh_tgt = get_norm_mesh(mesh_tgt, img_h, img_w)

    mask = torch.ones_like(input2_tensor)
    if torch.cuda.is_available():
        mask = mask.cuda()
    output_tps_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_ref, norm_rigid_mesh, (img_h, img_w))
    output_tps_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_tgt, norm_rigid_mesh, (img_h, img_w))


    out_dict = {}
    out_dict.update(rigid_mesh = rigid_mesh)
    out_dict.update(output_tps_ref = output_tps_ref, mesh_ref = mesh_ref)
    out_dict.update(output_tps_tgt = output_tps_tgt, mesh_tgt = mesh_tgt)

    return out_dict


def get_stitched_result(input1_tensor, input2_tensor, rigid_mesh, mesh_ref, mesh_tgt, max_out_height=6000):
    batch_size, _, img_h, img_w = input1_tensor.size()

    rigid_mesh = torch.stack([rigid_mesh[...,0]*img_w/512, rigid_mesh[...,1]*img_h/512], 3)
    mesh_ref = torch.stack([mesh_ref[...,0]*img_w/512, mesh_ref[...,1]*img_h/512], 3)
    mesh_tgt = torch.stack([mesh_tgt[...,0]*img_w/512, mesh_tgt[...,1]*img_h/512], 3)

    ######################################

    # calculate the size of stitched image
    width_max = torch.maximum(torch.max(mesh_ref[...,0]), torch.max(mesh_tgt[...,0]))
    width_min = torch.minimum(torch.min(mesh_ref[...,0]), torch.min(mesh_tgt[...,0]))
    height_max = torch.maximum(torch.max(mesh_ref[...,1]), torch.max(mesh_tgt[...,1]))
    height_min = torch.minimum(torch.min(mesh_ref[...,1]), torch.min(mesh_tgt[...,1]))

    out_width = width_max - width_min
    out_height = height_max - height_min
   

    # in case of image size is so huge.
    if max(out_height,out_width) >= max_out_height:
        print(out_width)
        print(out_height)
        return None, False

    # convert the mesh from [img_h, img_w] to [out_h, out_w]
    mesh_trans_ref = torch.stack([mesh_ref[...,0]-width_min, mesh_ref[...,1]-height_min], 3)
    mesh_trans_tgt = torch.stack([mesh_tgt[...,0]-width_min, mesh_tgt[...,1]-height_min], 3)

    # normalization
    norm_rigid_mesh = get_norm_mesh(rigid_mesh, img_h, img_w)
    norm_mesh_trans_ref = get_norm_mesh(mesh_trans_ref, out_height, out_width)
    norm_mesh_trans_tgt = get_norm_mesh(mesh_trans_tgt, out_height, out_width)

    # transformation
    mask = torch.ones_like(input2_tensor).cuda()
    output_ref = torch_tps_transform.transformer(torch.cat((input1_tensor+1, mask), 1), norm_mesh_trans_ref, norm_rigid_mesh, (out_height.int(), out_width.int()))
    output_tgt = torch_tps_transform.transformer(torch.cat((input2_tensor+1, mask), 1), norm_mesh_trans_tgt, norm_rigid_mesh, (out_height.int(), out_width.int()))

    img1 = output_ref[0,0:3,...]
    img2 = output_tgt[0,0:3,...]
    stitched =  img1*(img1/(img1+img2+1e-6)) + img2*(img2/(img1+img2+1e-6))

    out_dict = {}
    out_dict.update(output_ref=output_ref, output_tgt = output_tgt)
    out_dict.update(stitched=stitched)

    return out_dict, True
###


def get_res18_FeatureMap(resnet18_model):

    layers_list = []

    layers_list.append(resnet18_model.conv1)    #stride 2*2     H/2
    layers_list.append(resnet18_model.bn1)
    layers_list.append(resnet18_model.relu)
    layers_list.append(resnet18_model.maxpool)  #stride 2       H/4
    layers_list.append(resnet18_model.layer1)                  #H/4
    layers_list.append(resnet18_model.layer2)                  #H/8

    feature_extractor_stage1 = nn.Sequential(*layers_list)
    feature_extractor_stage2 = nn.Sequential(resnet18_model.layer3)

    return feature_extractor_stage1, feature_extractor_stage2


# define and forward
class Network(nn.Module):

    def __init__(self, descriptor_dim=256):
        super(Network, self).__init__()

        self.regressNet1 = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 3, 5
            
            nn.Flatten(),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=128, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=128, out_features=8, bias=True)
        )


        self.regressNet2_ref = nn.Sequential(
            nn.Conv2d(121, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 23, 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 3, 5
            
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )


        self.regressNet2_tgt = nn.Sequential(
            nn.Conv2d(121, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 23, 40

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 12, 20

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 6, 10

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # 3, 5
            
            nn.Flatten(),
            nn.Linear(in_features=4096, out_features=1024, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=1024, out_features=512, bias=True),
            nn.ReLU(inplace=True),

            nn.Linear(in_features=512, out_features=(grid_w+1)*(grid_h+1)*2, bias=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        resnet18_model = models.resnet.resnet18(weights="DEFAULT")
        if torch.cuda.is_available():
            resnet18_model = resnet18_model.cuda()
        self.feature_extractor_stage1, self.feature_extractor_stage2 = get_res18_FeatureMap(resnet18_model)
        #-----------------------------------------
        self.point_backbone = PointBackBoneWithDescriptorV2(descriptor_dim=descriptor_dim)
        self.fusion_block_32 = RobustTaskAwareFusionMOE(256, 256, num_blocks=3, balance_weight=0.01)
        self.fusion_block_64 = RobustTaskAwareFusionMOE(128, 128, num_blocks=3, balance_weight=0.01)

    def fuse(self):
        print('Fusing layers... ')
        for m in self.modules():
            if isinstance(m, RepVggBlock):
                m.convert_to_deploy()
        return self

    # forward
    def forward(self, input1_tesnor, input2_tesnor, point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor, is_stage2=False):
        batch_size, _, img_h, img_w = input1_tesnor.size()

#         feature extraction
        img_feature_1_64 = self.feature_extractor_stage1(input1_tesnor)
        img_feature_1_32 = self.feature_extractor_stage2(img_feature_1_64)
        img_feature_2_64 = self.feature_extractor_stage1(input2_tesnor)
        img_feature_2_32 = self.feature_extractor_stage2(img_feature_2_64)
        
        point_feature_1_32, point_feature_2_32, point_feature_1_64, point_feature_2_64 = self.point_backbone(point1_tensor, point2_tensor, descriptor1_tensor, descriptor2_tensor)

#        # fused features
        fusion_feature_1_64, b1_loss = self.fusion_block_64(point_feature_1_64, img_feature_1_64, is_stage2)
        fusion_feature_2_64, b2_loss = self.fusion_block_64(point_feature_2_64, img_feature_2_64, is_stage2)
        fusion_feature_1_32, b3_loss = self.fusion_block_32(point_feature_1_32, img_feature_1_32, is_stage2)
        fusion_feature_2_32, b4_loss = self.fusion_block_32(point_feature_2_32, img_feature_2_32, is_stage2)
        df_loss = (b3_loss + b4_loss) / 2

        ######### stage 1
        correlation_32 = self.CCL(fusion_feature_1_32, fusion_feature_2_32)
        offset_1 = self.regressNet1(correlation_32)

        # bidirectional decomposition
        H_motion_1 = offset_1.reshape(-1, 4, 2)
        src_p = torch.tensor([[0., 0.], [img_w, 0.], [0., img_h], [img_w, img_h]])
        if torch.cuda.is_available():
            src_p = src_p.cuda()
        src_p = src_p.unsqueeze(0).expand(batch_size, -1, -1)
        dst_p = src_p + H_motion_1
        dst_p_tgt = src_p + (H_motion_1 / 2.)
        H = torch_DLT.tensor_DLT(src_p/8, dst_p/8)
        H_tgt = torch_DLT.tensor_DLT(src_p/8, dst_p_tgt/8)
        H_ref = torch.matmul(torch.inverse(H), H_tgt)

        M_tensor = torch.tensor([[img_w/8 / 2.0, 0., img_w/8 / 2.0],
                      [0., img_h/8 / 2.0, img_h/8 / 2.0],
                      [0., 0., 1.]]).cuda()
        M_tile = M_tensor.unsqueeze(0).expand(batch_size, -1, -1)
        M_tensor_inv = torch.inverse(M_tensor)
        M_tile_inv = M_tensor_inv.unsqueeze(0).expand(batch_size, -1, -1)

        # warping by two homo
        H_mat_ref = torch.matmul(torch.matmul(M_tile_inv, H_ref), M_tile)
        warp_feature_1_64_ref = torch_homo_transform.transformer(fusion_feature_1_64, H_mat_ref, (int(img_h/8), int(img_w/8)))
        H_mat_tgt = torch.matmul(torch.matmul(M_tile_inv, H_tgt), M_tile)
        warp_feature_2_64_tgt = torch_homo_transform.transformer(fusion_feature_2_64, H_mat_tgt, (int(img_h/8), int(img_w/8)))
#        print(warp_feature_2_64_tgt.shape)
       ######### stage 2
        # for img1
        correlation_ref = self.cost_volume(warp_feature_1_64_ref, warp_feature_2_64_tgt, search_range=5, norm=False)
        offset_2_ref = self.regressNet2_ref(correlation_ref)

        # for img2
        correlation_tgt = self.cost_volume(warp_feature_2_64_tgt, warp_feature_1_64_ref, search_range=5, norm=False)
        offset_2_tgt = self.regressNet2_tgt(correlation_tgt)

        return offset_1, offset_2_ref, offset_2_tgt, df_loss

    @staticmethod
    def cost_volume(x1, x2, search_range, norm=True, fast=True):
        if norm:
            x1 = F.normalize(x1, p=2, dim=1)
            x2 = F.normalize(x2, p=2, dim=1)
        bs, c, h, w = x1.shape
        padded_x2 = F.pad(x2, [search_range] * 4)  # [b,c,h,w] -> [b,c,h+sr*2,w+sr*2]
        max_offset = search_range * 2 + 1

        if fast:
            # faster(*2) but cost higher(*n) GPU memory
            patches = F.unfold(padded_x2, (max_offset, max_offset)).reshape(bs, c, max_offset ** 2, h, w)
            cost_vol = (x1.unsqueeze(2) * patches).mean(dim=1, keepdim=False)
        else:
            # slower but save memory
            cost_vol = []
            for j in range(0, max_offset):
                for i in range(0, max_offset):
                    x2_slice = padded_x2[:, :, j:j + h, i:i + w]
                    cost = torch.mean(x1 * x2_slice, dim=1, keepdim=True)
                    cost_vol.append(cost)
            cost_vol = torch.cat(cost_vol, dim=1)

        cost_vol = F.leaky_relu(cost_vol, 0.1)

        return cost_vol

 
    def extract_patches(self, x, kernel=3, stride=1):
        if kernel != 1:
            x = nn.ZeroPad2d(1)(x)
        x = x.permute(0, 2, 3, 1)
        all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
        return all_patches


    def CCL(self, feature_1, feature_2):
        bs, c, h, w = feature_1.size()

        norm_feature_1 = F.normalize(feature_1, p=2, dim=1)
        norm_feature_2 = F.normalize(feature_2, p=2, dim=1)
        #print(norm_feature_2.size())

        patches = self.extract_patches(norm_feature_2)
        if torch.cuda.is_available():
            patches = patches.cuda()

        matching_filters  = patches.reshape((patches.size()[0], -1, patches.size()[3], patches.size()[4], patches.size()[5]))

        match_vol = []
        for i in range(bs):
            single_match = F.conv2d(norm_feature_1[i].unsqueeze(0), matching_filters[i], padding=1)
            match_vol.append(single_match)

        match_vol = torch.cat(match_vol, 0)

        # scale softmax
        softmax_scale = 10
        match_vol = F.softmax(match_vol*softmax_scale,1)

        channel = match_vol.size()[1]

        h_one = torch.linspace(0, h-1, h)
        one1w = torch.ones(1, w)
        if torch.cuda.is_available():
            h_one = h_one.cuda()
            one1w = one1w.cuda()
        h_one = torch.matmul(h_one.unsqueeze(1), one1w)
        h_one = h_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        w_one = torch.linspace(0, w-1, w)
        oneh1 = torch.ones(h, 1)
        if torch.cuda.is_available():
            w_one = w_one.cuda()
            oneh1 = oneh1.cuda()
        w_one = torch.matmul(oneh1, w_one.unsqueeze(0))
        w_one = w_one.unsqueeze(0).unsqueeze(0).expand(bs, channel, -1, -1)

        c_one = torch.linspace(0, channel-1, channel)
        if torch.cuda.is_available():
            c_one = c_one.cuda()
        c_one = c_one.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(bs, -1, h, w)

        flow_h = match_vol*(c_one//w - h_one)
        flow_h = torch.sum(flow_h, dim=1, keepdim=True)
        flow_w = match_vol*(c_one%w - w_one)
        flow_w = torch.sum(flow_w, dim=1, keepdim=True)

        feature_flow = torch.cat([flow_w, flow_h], 1)
        #print(flow.size())

        return feature_flow


############  MOE-based fusion module ###########
def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 
    
class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)
            if hasattr(self, 'conv3'):
                y = y + self.conv3(x)
        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)
        
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias

        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'conv3'):
            self.__delattr__('conv3')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernel_id, bias_id = self._fuse_identity_tensor()
        
        kernel = (
            kernel3x3 +
            self._pad_1x1_to_3x3_tensor(kernel1x1, kernel3x3) +
            self._pad_1x1_to_3x3_tensor(kernel_id, kernel3x3)
        )
        bias = bias3x3 + bias1x1 + bias_id
        
        return kernel, bias

    def _pad_1x1_to_3x3_tensor(self, kernel1x1, ref_kernel):
        if isinstance(kernel1x1, int) and kernel1x1 == 0:
            return torch.zeros_like(ref_kernel)
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: 'ConvNormLayer'):
        if branch is None:
            return 0, 0
        
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        fused_kernel = kernel * t
        fused_bias = beta - running_mean * gamma / std
        
        return fused_kernel, fused_bias

    def _fuse_identity_tensor(self):
        if hasattr(self, 'conv3'):
            return self._fuse_bn_tensor(self.conv3)
        return 0, 0

    def add_identity_branch(self):
        if self.ch_in == self.ch_out and not hasattr(self, 'conv3'):
            id_conv = nn.Conv2d(self.ch_in, self.ch_out, 1, 1, padding=0, bias=False)
            nn.init.zeros_(id_conv.weight)
            for i in range(self.ch_in):
                id_conv.weight.data[i, i, 0, 0] = 1.0
            
            id_bn = nn.BatchNorm2d(self.ch_out)
            
            self.conv3 = ConvNormLayer(self.ch_in, self.ch_out, 1, 1, padding=0, act=None)
            self.conv3.conv = id_conv
            self.conv3.norm = id_bn


class CSPRepFusion(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepFusion, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        x = torch.cat([x1,x2],dim=1)
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)
    
class RouterNetwork(nn.Module):
    def __init__(self, in_channels, num_expert):
        super(RouterNetwork, self).__init__()
        
        def make_analyzer(input_channels):
            return nn.Sequential(
                nn.Linear(input_channels * 2, input_channels), 
                nn.ReLU(),
                nn.Linear(input_channels, input_channels // 2),
                nn.ReLU()
            )

        self.point_analyzer = make_analyzer(in_channels)
        self.image_analyzer = make_analyzer(in_channels)

        self.fusion_analyzer = make_analyzer(in_channels * 2) 
        
        total_feat_dim = (in_channels // 2) * 2 + (in_channels * 2 // 2)
        
        self.decision = nn.Sequential(
            nn.Linear(total_feat_dim, total_feat_dim // 2),
            nn.ReLU(),
            nn.Linear(total_feat_dim // 2, num_expert) 
        )

    def _get_statistics(self, x):
        mean_feat = x.mean(dim=[2, 3]) 
        std_feat = x.std(dim=[2, 3])
        return torch.cat([mean_feat, std_feat], dim=1)

    def forward(self, x1, x2):
        stats_x1 = self._get_statistics(x1) # [B, C*2]
        stats_x2 = self._get_statistics(x2) # [B, C*2]
        
        point_feat = self.point_analyzer(stats_x1)
        image_feat = self.image_analyzer(stats_x2)

        stats_fusion = torch.cat([stats_x1, stats_x2], dim=1)
        fusion_feat = self.fusion_analyzer(stats_fusion)
        
        combined = torch.cat([point_feat, image_feat, fusion_feat], dim=1)
        
        return self.decision(combined)

class GeometricExpert(nn.Module):
    def __init__(self,
                 in_channels, hidden_channels, out_channels, num_blocks, act, bias=None):
        super(GeometricExpert, self).__init__()
        expansion=1.0
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        x_f = x1

        x_1 = self.conv1(x_f)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x_f)
        return self.conv3(x_1 + x_2)
        
class SemanticExpert(nn.Module):
    def __init__(self,
                 in_channels, hidden_channels, out_channels, num_blocks, act, bias=None):
        super(SemanticExpert, self).__init__()
        expansion=1.0
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        x_f = x2

        x_1 = self.conv1(x_f)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x_f)
        return self.conv3(x_1 + x_2)
    
class InteractiveExpert(nn.Module):
    def __init__(self,
                 in_channels, hidden_channels, out_channels, num_blocks, act, bias=None):
        super(InteractiveExpert, self).__init__()
        expansion=1.0
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x1, x2):
        x_f = torch.cat([x1, x2],dim=1)

        x_1 = self.conv1(x_f)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x_f)
        return self.conv3(x_1 + x_2)
    

class RobustTaskAwareFusionMOE(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu",
                 balance_weight=1,
                 sensor_dropout_prob=0.3,
                 sensor_noise_std=0.1,
                 sensor_channel_dropout=0.2,
                 sensor_failure_types=['complete', 'noisy']):
        super(RobustTaskAwareFusionMOE, self).__init__()
        hidden_channels = int(out_channels * expansion)
        
        self.balance_weight = balance_weight
        self.sensor_dropout_prob = sensor_dropout_prob
        self.sensor_noise_std = sensor_noise_std
        self.sensor_channel_dropout = sensor_channel_dropout
        self.sensor_failure_types = sensor_failure_types
        self.sensor_intermittent_rate = 0.3  
        self.sensor_bias_std = 0.1 
        self.sensor_saturation_threshold = 0.8
        
        self.experts = nn.ModuleList([
            GeometricExpert(in_channels, hidden_channels, out_channels, num_blocks, act),
            SemanticExpert(in_channels, hidden_channels, out_channels, num_blocks, act),
            InteractiveExpert(in_channels*2, hidden_channels, out_channels, num_blocks, act)
        ])
        
        self.num_expert = len(self.experts)
        self.router = RouterNetwork(in_channels, num_expert=self.num_expert)


    def forward(self, x1, x2, is_stage2=False):
        batch_size = x1.shape[0]
        
        if self.training and is_stage2:
            x1_noisy, x2_noisy = self._apply_sensor_failures(x1, x2)
        else:
            x1_noisy, x2_noisy = x1, x2
            
        route_scores = self.router(x1_noisy, x2_noisy)
        expert_weights = F.softmax(route_scores, dim=1)  # [B, 4]
#        print("[W_g,W_s,W_h]",expert_weights)
        
        specific_outputs = []
        for expert in self.experts:
            specific_outputs.append(expert(x1_noisy, x2_noisy))
        specific_outputs = torch.stack(specific_outputs, dim=1)

        specific_weights = expert_weights

        specific_weights_view = specific_weights.view(batch_size, self.num_expert, 1, 1, 1)
        specific_contrib = torch.sum(specific_weights_view * specific_outputs, dim=1)

        final_output = specific_contrib
        
        balance_loss = 0
        if self.training:
            balance_loss = self._compute_balance_loss(specific_weights, x1_noisy, x2_noisy)
        
        return final_output, balance_loss
    
    def _apply_sensor_failures(self, x1, x2):
      batch_size = x1.shape[0]
      
      x1_corrupted = x1.clone()
      x2_corrupted = x2.clone()
      
      for b in range(batch_size):
        if torch.rand(1) < 0.25:  
          
            failure_type = np.random.choice(self.sensor_failure_types)
            
            if failure_type == 'complete':
#            
                if torch.rand(1) < 0.5:
                    x1_corrupted[b] = 0  
                else:
                    x2_corrupted[b] = 0  
                  
            elif failure_type == 'noisy':
             
                noise_level = torch.rand(1).to(x1.device) * self.sensor_noise_std
                if torch.rand(1) < 0.5:
                    noise = torch.randn_like(x1[b]).to(x1.device) * noise_level
                    x1_corrupted[b] = x1[b] + noise.to(x1.device)
                else:
                    noise = torch.randn_like(x2[b]).to(x1.device) * noise_level
                    x2_corrupted[b] = x2[b] + noise
                    
            elif failure_type == 'degraded':
          
                if torch.rand(1) < 0.5:
                  
                    x1_corrupted[b] = F.avg_pool2d(
                        x1[b].unsqueeze(0), 
                        kernel_size=3, stride=1, padding=1
                    ).squeeze(0)
                 
                    noise = torch.randn_like(x1[b]).to(x1.device) * self.sensor_noise_std * 0.5
                    x1_corrupted[b] = x1_corrupted[b] + noise
                else:
      
                    orig_size = x2.shape[2:]
             
                    scale_factor = 0.3 + torch.rand(1).item() * 0.4  
                    downsampled = F.interpolate(
                        x2[b].unsqueeze(0), 
                        scale_factor=scale_factor, 
                        mode='bilinear',
                        align_corners=False
                    )
                    x2_corrupted[b] = F.interpolate(
                        downsampled, 
                        size=orig_size, 
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                    
            elif failure_type == 'intermittent':
          
                if torch.rand(1) < 0.5:
                
                    mask = (torch.rand_like(x1[b]) > self.sensor_intermittent_rate).float()
                    x1_corrupted[b] = x1[b] * mask
                else:
    
                    mask = (torch.rand_like(x2[b]) > self.sensor_intermittent_rate).float()
                    x2_corrupted[b] = x2[b] * mask
                    
            elif failure_type == 'bias':
                bias = torch.randn(1).to(x1.device) * self.sensor_bias_std
                if torch.rand(1) < 0.5:
                    x1_corrupted[b] = x1[b] + bias

                    x1_corrupted[b] = torch.clamp(x1_corrupted[b], 0, 1)
                else:
                    x2_corrupted[b] = x2[b] + bias
                    x2_corrupted[b] = torch.clamp(x2_corrupted[b], 0, 1)
                    
            elif failure_type == 'saturation':
                if torch.rand(1) < 0.5:
                    max_val = torch.max(x1[b])
                    saturation_value = max_val * self.sensor_saturation_threshold
            
                    mask = x1[b] > saturation_value
                    x1_corrupted[b] = x1[b].clone()
                    x1_corrupted[b][mask] = saturation_value
              
                    noise = torch.randn_like(x1[b]).to(x1.device) * self.sensor_noise_std * 0.3
                    x1_corrupted[b] = x1_corrupted[b] + noise
                    x1_corrupted[b] = torch.clamp(x1_corrupted[b], 0, 1)
                else:
                    max_val = torch.max(x2[b])
                    saturation_value = max_val * self.sensor_saturation_threshold
                    mask = x2[b] > saturation_value
                    x2_corrupted[b] = x2[b].clone()
                    x2_corrupted[b][mask] = saturation_value
                    noise = torch.randn_like(x2[b]).to(x1.device) * self.sensor_noise_std * 0.3
                    x2_corrupted[b] = x2_corrupted[b] + noise
                    x2_corrupted[b] = torch.clamp(x2_corrupted[b], 0, 1)
      
      return x1_corrupted, x2_corrupted
    
    def _compute_balance_loss(self, expert_weights, x1_noisy=None, x2_noisy=None): 
        importance = expert_weights.mean(dim=0)
        importance_loss = torch.std(importance)
        
        entropy = -torch.sum(expert_weights * torch.log(expert_weights + 1e-10), dim=1).mean()
        entropy_loss = -entropy
        
        return (importance_loss + 0.1 * entropy_loss) * self.balance_weight



############  Neural Point Transform Module ###########

class LayerNorm1d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.ln = nn.BatchNorm1d(channels, eps=eps)
        
    def forward(self, x):
        # x shape: (B, C, N)
        return self.ln(x)

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.ln = nn.BatchNorm2d(channels, eps=eps)
        
    def forward(self, x):
        return self.ln(x)



def exists(val):
    return val is not None


def square_distance(src, dst):
    return torch.cdist(src, dst, p=2).pow(2)


def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    
    centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
    distance = torch.ones(B, N, device=device) * 1e10
    
    farthest = torch.randint(0, N, (B,), dtype=torch.long, device=device)
    
    batch_indices = torch.arange(B, dtype=torch.long, device=device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, -1)[1]
    
    return centroids

def ball_query(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    
    return group_idx


def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def three_interpolation(points, features, query_points):
    B, N, C = points.shape
    _, M, _ = query_points.shape
    _, _, D = features.shape
    
    dists = square_distance(query_points, points)
    dists, idx = dists.sort(dim=-1)
    dists, idx = dists[:, :, :3], idx[:, :, :3] 

    dist_recip = 1.0 / (dists + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    
    grouped_features = index_points(features.transpose(1, 2), idx)  # (B, M, 3, D)
    
    interpolated_features = torch.sum(grouped_features * weight.unsqueeze(-1), dim=2)  # (B, M, D)
    return interpolated_features.transpose(1, 2)  # (B, D, M)
    

def downsample_fps(xyz, n_sample):
    # xyz: (b, 2, n)
    if n_sample == xyz.shape[-1]:
        sample_idx = torch.arange(n_sample, device=xyz.device)
        sample_idx = sample_idx.unsqueeze(0).repeat(xyz.shape[0], 1)  # (b, n)
        return SampleResult(None, xyz.clone(), sample_idx, None)
    
    _xyz = xyz.transpose(1, 2).contiguous()  # (b, n, 2)
    sample_idx = farthest_point_sample(_xyz, n_sample).long()  # (b, k)
    
    batch_size = xyz.shape[0]
    sample_xyz = torch.gather(xyz, 2, 
                             sample_idx.unsqueeze(1).repeat(1, xyz.shape[1], 1))  # (b, 2, k)
    return SampleResult(None, sample_xyz, sample_idx, None)

def _ball_query(src, query, radius, k):
    # conduct ball query on dim 1
    src = src.transpose(1, 2).contiguous()  # (b, n, 2)
    query = query.transpose(1, 2).contiguous()  # (b, m, 2)
    idx = ball_query(radius, k, src, query).long()
    dists = None
    return idx, dists

 
def gather(x, idx):
    B, D, N = x.shape
    _, M, K = idx.shape
    
    idx_expanded = idx.unsqueeze(1).expand(-1, D, -1, -1)
    x_expanded = x.unsqueeze(2).expand(-1, -1, M, -1)
    
    return torch.gather(x_expanded, 3, idx_expanded)

SampleResult = namedtuple('SampleResult', ['x', 'xyz', 'sample_idx', 'neighbor_idx'])

class SABlock(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1, layers=1, radius=0.1, k=16):
        super().__init__()
        self.stride = stride
        self.radius = radius
        self.layers = layers
        self.k = k

        dims = [in_dim + 2] + [out_dim] * layers  

        if layers == 1:
            self.convs = nn.Conv2d(dims[0], dims[1], 1, bias=False)
            self.norm = LayerNorm1d(out_dim)
            self.act = nn.ReLU()
        else:
            self.skip_conv = nn.Conv1d(in_dim, out_dim, 1, bias=False) if in_dim != out_dim else nn.Identity()
            self.convs = nn.Sequential(*[
                nn.Sequential(nn.Conv2d(in_d, out_d, 1, bias=False),
                              LayerNorm2d(out_d),
                              nn.ReLU())
                for in_d, out_d in zip(dims[:-2], dims[1:-1])
            ])
            self.convs.append(nn.Conv2d(dims[-2], dims[-1], 1, bias=False))
            self.norm = LayerNorm1d(out_dim)
            self.act = nn.ReLU()

    def route(self, src_x, src_xyz, xyz, radius, k, neighbor_idx=None):
        # src_x: (b, d, n)
        # src_xyz: (b, 2, n)
        # xyz: (b, 2, m)
        if not exists(neighbor_idx):
            neighbor_idx = _ball_query(src_xyz, xyz, radius, k)[0]  # (b, m, k)
        neighbor_xyz = gather(src_xyz, neighbor_idx)  # (b, 2, m, k)
        neighbor_xyz -= xyz.unsqueeze(-1)
        neighbor_xyz /= radius
        x = gather(src_x, neighbor_idx)  # (b, d, m, k)
        x = torch.cat([x, neighbor_xyz], dim=1)  # (b, d+2, m, k)
        return SampleResult(x, xyz, None, neighbor_idx)

    def forward(self, x, xyz):
        # x: (b, d, n)
        # xyz: (b, 2, n)
        # out: (b, d', n // stride)
        sample = downsample_fps(xyz, n_sample=xyz.shape[-1] // self.stride)
        
        batch_size, feat_dim = x.shape[0], x.shape[1]
        inputs = torch.gather(x, 2, 
                            sample.sample_idx.unsqueeze(1).repeat(1, feat_dim, 1))
        
        sample = self.route(x, xyz, sample.xyz, self.radius, self.k)
        x = self.convs(sample.x)
        x = x.max(dim=-1)[0]
        if hasattr(self, 'skip_conv'):
            x = self.skip_conv(inputs) + x
        x = self.act(self.norm(x))
        return SampleResult(x, sample.xyz, sample.sample_idx, sample.neighbor_idx)

class InvResMLP(nn.Module):
    def __init__(self, in_dim, expansion=4, radius=0.1, k=16):
        super().__init__()
        self.sa_conv = SABlock(in_dim, in_dim, stride=1, layers=1, radius=radius, k=k)

        dims = [in_dim, in_dim * expansion, in_dim]
        self.conv = nn.Sequential(
            nn.Conv1d(dims[0], dims[1], 1, bias=False),
            LayerNorm1d(dims[1]),
            nn.ReLU(),
            nn.Conv1d(dims[1], dims[2], 1, bias=False),
            LayerNorm1d(dims[2])
        )
        self.act = nn.ReLU()

    def forward(self, x, xyz):
        inputs = x
        x = self.sa_conv(x, xyz).x
        x = self.conv(x)
        x = self.act(inputs + x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=3, eps=1e-5):
        super().__init__()
        self.k = k
        assert k == 3, "only support k=3"
        self.eps = eps
        dims = [in_dim, out_dim, out_dim]
        self.conv = nn.Sequential(*[
            nn.Sequential(nn.Conv1d(in_d, out_d, 1, bias=False),
                          LayerNorm1d(out_d),
                          nn.ReLU())
            for in_d, out_d in zip(dims[:-1], dims[1:])
        ])

    def route(self, src_x, src_xyz, dst_x, dst_xyz, neighbor_idx=None, dists=None):
        src_xyz = src_xyz.transpose(1, 2).contiguous()  # (b, n, 2)
        dst_xyz = dst_xyz.transpose(1, 2).contiguous()  # (b, m, 2)
        lerp_x = three_interpolation(src_xyz, src_x, dst_xyz)
        dst_x = torch.cat([dst_x, lerp_x], dim=1)  # (b, d+d', m)
        return dst_x

    def forward(self, x, xyz, sub_x, sub_xyz):
        x = self.route(sub_x, sub_xyz, x, xyz)
        x = self.conv(x)
        return x

class PointNextEncoder(nn.Module):  
    def __init__(
            self,
            in_dim=3,
            dims=[32, 64, 128, 256], 
            blocks=[2, 2, 2],
            strides=[4, 4, 4],
            radius=0.1,
            k=16,
            sa_layers=1,
    ):
        super().__init__()
        self.encoder_dims = dims

        self.stem = nn.Sequential(
            nn.Conv1d(in_dim, dims[0], 1, bias=False),
            LayerNorm1d(dims[0]),
            nn.ReLU()
        )

        radius_scaling = 2
        radii = [radius * (radius_scaling ** i) for i in range(len(blocks))]
        self.encoder = nn.ModuleList()
        for i in range(len(blocks)):
            layers = nn.Sequential(
                SABlock(dims[i], dims[i + 1], stride=strides[i], layers=sa_layers, radius=radii[i], k=k),
                *[InvResMLP(dims[i + 1], radius=radii[i] * radius_scaling, k=k) for _ in range(blocks[i] - 1)]
            )
            self.encoder.append(layers)

        self.out_dim = dims[-1]

    def forward_features(self, x, xyz):
        x = self.stem(x)
        features = [(x, xyz)]
        for block in self.encoder:
            sample = block[0](x, xyz)
            x, xyz = sample.x, sample.xyz
            for layer in block[1:]:
                x = layer(x, xyz)
            features.append((x, xyz))
        return features

    def forward(self, x, xyz):
        return self.forward_features(x, xyz)

class PointNextDecoder(nn.Module):
    def __init__(self, encoder_dims=[32, 64, 128, 256]):
        super().__init__()
        self.decoder = nn.ModuleList()

        decoder_dims = encoder_dims[::-1]
        for i in range(len(decoder_dims) - 1):
            self.decoder.append(UpBlock(decoder_dims[i] + decoder_dims[i + 1], decoder_dims[i + 1]))

        self.out_dim = decoder_dims[-1]

    def forward(self, feats):
        sub_x, sub_xyz = feats.pop()
        x_list, xyz_list = [], []
        for i, block in enumerate(self.decoder):
            x, xyz = feats.pop()
#            print(x.shape, xyz.shape)
            x = block(x, xyz, sub_x, sub_xyz)
            x_list.append(x)
            xyz_list.append(xyz)
            sub_x, sub_xyz = x, xyz
        return x_list, xyz_list

class PointNextWithDescriptor(nn.Module):
    def __init__(self, input_dim=2, descriptor_dim=256, encoder_dims=[32, 64, 128, 256], 
                 blocks=[2, 2, 2], strides=[4, 4, 4], radius=0.1, k=16):
        super().__init__()

        self.combined_dim = input_dim + descriptor_dim
        self.descriptor_dim = descriptor_dim

        self.encoder = PointNextEncoder(
            in_dim=self.combined_dim,
            dims=[encoder_dims[0]] + encoder_dims[1:],
            blocks=blocks,
            strides=strides,
            radius=radius,
            k=k,
            sa_layers=1
        )
        
        self.decoder = PointNextDecoder(encoder_dims=[encoder_dims[0]] + encoder_dims[1:])
        
        self.out_dim = encoder_dims

    def forward(self, x, descriptors):
#        print(self.descriptor_dim,descriptors.shape)
        if self.descriptor_dim > 0:
          x_combined = torch.cat([x, descriptors], dim=1)  # (B, 2+descriptor_dim, N)
        else:
          x_combined = x  # (B, 2+descriptor_dim, N)
#          print(x_combined.shape)

        xyz = x # (B, 2, N)
        if xyz.max() > 1.0:
          print('waring, wrong norm. xyz:',xyz.shape,xyz.max(),xyz.min())
        feats = self.encoder.forward_features(x_combined, xyz)
        
        final_feat_list, final_xyz_list = self.decoder(feats)  # (B, out_dim, N)
        
        return final_feat_list, final_xyz_list

        
class PointBackBoneWithDescriptorV2(nn.Module):
    def __init__(self,
                 voxel_size_list: list = [32, 64],
                 feature_transform: bool = True,
                 descriptor_dim: int = 256):
        super().__init__()
        self.feature_transform = feature_transform
        self.descriptor_dim = descriptor_dim
        
        self.pointnext_feat = PointNextWithDescriptor(
            input_dim=2,
            descriptor_dim=descriptor_dim,
            encoder_dims=[64, 128,  256, 512],  
            blocks=[ 2, 2, 4],  
            strides=[4, 4, 4],  
            radius=0.1,
            k=16
        )
        
        self.voxelizer_32 = IndexPutVoxelizer(
            voxel_size=voxel_size_list[0]
        )

        self.voxelizer_64 = IndexPutVoxelizer(
            voxel_size=voxel_size_list[1]
        )
        
        # channel dim mapping
        self.conv32 = nn.Sequential(
          nn.Conv2d(128, 256, 3, 1, 1),
          LayerNorm2d(256),
          nn.ReLU(),
          nn.Conv2d(256, 256, 3, 1, 1),
          LayerNorm2d(256),
          nn.ReLU(),
        )
        self.conv64 = nn.Sequential(
          nn.Conv2d(64, 128, 3, 1, 1),
          LayerNorm2d(128),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, 1),
          LayerNorm2d(128),
          nn.ReLU(),
        )

    def forward(self, mkpt0, mkpt1, des0, des1):
        kpt0_t = mkpt0.transpose(1, 2)  # (B, 2, N)
        kpt1_t = mkpt1.transpose(1, 2)  # (B, 2, N)
        des0_t = des0.transpose(1, 2)   # (B, descriptor_dim, N)
        des1_t = des1.transpose(1, 2)   # (B, descriptor_dim, N)
        
        point_feat0_list, point_xyz0_list = self.pointnext_feat(kpt0_t, des0_t)  # (B, out_dim, N)
        point_feat1_list, point_xyz1_list = self.pointnext_feat(kpt1_t, des1_t)  # (B, out_dim, N)
        
        # get features and point coord in resoultion 32 and 64
        point_feat0_32 = point_feat0_list[-2]
        point_feat1_32 = point_feat1_list[-2]
        mkpt0_32 = point_xyz0_list[-2].permute(0, 2, 1)
        mkpt1_32 = point_xyz1_list[-2].permute(0, 2, 1)
        
        point_feat0_64 = point_feat0_list[-1]
        point_feat1_64 = point_feat1_list[-1]
        mkpt0_64 = point_xyz0_list[-1].permute(0, 2, 1)
        mkpt1_64 = point_xyz1_list[-1].permute(0, 2, 1)

        point_feat0_32 = point_feat0_32.permute(0, 2, 1)  # (B, N, out_dim)
        point_feat1_32 = point_feat1_32.permute(0, 2, 1)  # (B, N, out_dim)
        
#        print(mkpt0_32.shape,mkpt0_32.max(),mkpt0_32.min())
        voxels_0_32 = self.voxelizer_32(point_feat0_32, mkpt0_32)  # (B, 32, 32, out_dim)
        voxels_1_32 = self.voxelizer_32(point_feat1_32, mkpt1_32)  # (B, 32, 32, out_dim)
        
        voxels_0_32 = voxels_0_32.permute(0, 3, 1, 2)  # (B, out_dim, 32, 32)
        voxels_1_32 = voxels_1_32.permute(0, 3, 1, 2)  # (B, out_dim, 32, 32)
        
        point_feat0_64 = point_feat0_64.permute(0, 2, 1)  # (B, N, out_dim)
        point_feat1_64 = point_feat1_64.permute(0, 2, 1)  # (B, N, out_dim)

        voxels_0_64 = self.voxelizer_64(point_feat0_64, mkpt0_64)  # (B, 64, 64, out_dim)
        voxels_1_64 = self.voxelizer_64(point_feat1_64, mkpt1_64)  # (B, 64, 64, out_dim)
        
        voxels_0_64 = voxels_0_64.permute(0, 3, 1, 2)  # (B, out_dim, 64, 64)
        voxels_1_64 = voxels_1_64.permute(0, 3, 1, 2)  # (B, out_dim, 64, 64)
        
#        print(voxels_0_32.shape, voxels_1_32.shape, voxels_0_64.shape, voxels_1_64.shape)
        # modify channel dim       
        voxels_0_32 = self.conv32(voxels_0_32)
        voxels_1_32 = self.conv32(voxels_1_32)
        voxels_0_64 = self.conv64(voxels_0_64)
        voxels_1_64 = self.conv64(voxels_1_64)
        
        return voxels_0_32, voxels_1_32, voxels_0_64, voxels_1_64


class IndexPutVoxelizer(nn.Module):
    def __init__(self, voxel_size=32):
        super().__init__()
        self.voxel_size = voxel_size
        
    def forward(self, local_features: torch.Tensor, keypoint_coords: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, feature_dim = local_features.shape
        
        grid_coords = self.normalize_and_quantize(keypoint_coords, image_size=(512, 512))
        
        voxel_grid = torch.zeros(batch_size, self.voxel_size, self.voxel_size, feature_dim,
                                device=local_features.device)
        
        flat_spatial_indices = grid_coords[:, :, 1] * self.voxel_size + grid_coords[:, :, 0]
        batch_offsets = torch.arange(batch_size, device=local_features.device).view(batch_size, 1) * (self.voxel_size * self.voxel_size)
        flat_indices = (flat_spatial_indices + batch_offsets).long()
        
        flat_voxel_grid = voxel_grid.view(batch_size * self.voxel_size * self.voxel_size, feature_dim)
        
        flat_features = local_features.contiguous().view(batch_size * num_points, feature_dim)
        
        flat_voxel_grid.scatter_reduce_(
            dim=0,
            index=flat_indices.view(-1, 1).expand(-1, feature_dim),
            src=flat_features,
            reduce='amax',
            include_self=False
        )
        
        voxel_grid = flat_voxel_grid.view(batch_size, self.voxel_size, self.voxel_size, feature_dim)
        return voxel_grid
    
    def normalize_and_quantize(self, coords, image_size=None, margin=0.05, is_norm = False):
        batch_size, num_points, _ = coords.shape
        
        if not is_norm:
          normalized_x = coords[:, :, 0]
          normalized_y = coords[:, :, 1]
        else:
          if image_size is not None and coords.max() > 2:
              img_h, img_w = image_size
              normalized_x = coords[:, :, 0] / (img_w - 1)
              normalized_y = coords[:, :, 1] / (img_h - 1)
          else:
              coords_min = coords.min(dim=1, keepdim=True)[0]
              coords_max = coords.max(dim=1, keepdim=True)[0]
              coords_center = (coords_min + coords_max) / 2
              coords_range = (coords_max - coords_min) * (1 + 2 * margin)
              
              coords_range = torch.max(coords_range, torch.tensor(1.0, device=coords.device))
              normalized = (coords - coords_center) / coords_range + 0.5
              normalized = torch.clamp(normalized, 0.0, 1.0)
              normalized_x, normalized_y = normalized[:, :, 0], normalized[:, :, 1]
                    
        grid_x = (normalized_x * (self.voxel_size - 1)).long()
        grid_y = (normalized_y * (self.voxel_size - 1)).long()
        
        grid_coords = torch.stack([grid_x, grid_y], dim=-1)
        grid_coords = torch.clamp(grid_coords, 0, self.voxel_size - 1)
        
        return grid_coords

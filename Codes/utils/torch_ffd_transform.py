import torch
import numpy as np
import grid_res
grid_h = grid_res.GRID_H *2
grid_w = grid_res.GRID_W *2

# transforming an image (U) from target (control points) to source (control points)
# all the points should be normalized from -1 ~1

def transformer(U, source, target, out_size):

    def _repeat(x, n_repeats):

        rep = torch.ones([n_repeats, ]).unsqueeze(0)
        rep = rep.int()
        x = x.int()

        x = torch.matmul(x.reshape([-1,1]), rep)
        return x.reshape([-1])

    def _interpolate(im, x, y, out_size):

        num_batch, num_channels , height, width = im.size()

        height_f = height
        width_f = width
        out_height, out_width = out_size[0], out_size[1]

        zero = 0
        max_y = height - 1
        max_x = width - 1

        x = (x + 1.0)*(width_f) / 2.0
        y = (y + 1.0) * (height_f) / 2.0

        # do sampling
        x0 = torch.floor(x).int()
        x1 = x0 + 1
        y0 = torch.floor(y).int()
        y1 = y0 + 1

        x0 = torch.clamp(x0, zero, max_x)
        x1 = torch.clamp(x1, zero, max_x)
        y0 = torch.clamp(y0, zero, max_y)
        y1 = torch.clamp(y1, zero, max_y)
        dim2 = torch.from_numpy( np.array(width) )
        dim1 = torch.from_numpy( np.array(width * height) )

        base = _repeat(torch.arange(0,num_batch) * dim1, out_height * out_width)
        if torch.cuda.is_available():
            dim2 = dim2.cuda()
            dim1 = dim1.cuda()
            y0 = y0.cuda()
            y1 = y1.cuda()
            x0 = x0.cuda()
            x1 = x1.cuda()
            base = base.cuda()
        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # channels dim
        im = im.permute(0,2,3,1)
        im_flat = im.reshape([-1, num_channels]).float()


        idx_a = idx_a.unsqueeze(-1).long()
        idx_a = idx_a.expand(out_height * out_width * num_batch,num_channels)
        Ia = torch.gather(im_flat, 0, idx_a)

        idx_b = idx_b.unsqueeze(-1).long()
        idx_b = idx_b.expand(out_height * out_width * num_batch, num_channels)
        Ib = torch.gather(im_flat, 0, idx_b)

        idx_c = idx_c.unsqueeze(-1).long()
        idx_c = idx_c.expand(out_height * out_width * num_batch, num_channels)
        Ic = torch.gather(im_flat, 0, idx_c)

        idx_d = idx_d.unsqueeze(-1).long()
        idx_d = idx_d.expand(out_height * out_width * num_batch, num_channels)
        Id = torch.gather(im_flat, 0, idx_d)

        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()

        wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
        wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
        wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
        wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
        output = wa*Ia+wb*Ib+wc*Ic+wd*Id

        return output

    def _meshgrid(height, width, source):

        x_t = torch.matmul(torch.ones([height, 1]), torch.unsqueeze(torch.linspace(-1.0, 1.0, width), 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(-1.0, 1.0, height), 1), torch.ones([1, width]))
        if torch.cuda.is_available():
            x_t = x_t.cuda()
            y_t = y_t.cuda()

        x_t_flat = x_t.reshape([1, 1, -1])
        y_t_flat = y_t.reshape([1, 1, -1])

        num_batch = source.size()[0]
        px = torch.unsqueeze(source[:,:,0], 2)  # [bn, pn, 1]
        py = torch.unsqueeze(source[:,:,1], 2)  # [bn, pn, 1]
        if torch.cuda.is_available():
            px = px.cuda()
            py = py.cuda()
        d2 = torch.square(x_t_flat - px) + torch.square(y_t_flat - py)
        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, h*w]
        x_t_flat_g = x_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        y_t_flat_g = y_t_flat.expand(num_batch, -1, -1)  # [bn, 1, h*w]
        ones = torch.ones_like(x_t_flat_g) # [bn, 1, h*w]
        if torch.cuda.is_available():
            ones = ones.cuda()

        grid = torch.cat((ones, x_t_flat_g, y_t_flat_g, r), 1) # [bn, 3+pn, h*w]

        #if torch.cuda.is_available():
        #    grid = grid.cuda()
        return grid
    
    

    def _transform(T, source, input_dim, out_size):
        num_batch, num_channels, height, width = input_dim.size()

        # grid = _meshgrid(grid_h, grid_w, source) # [bn, 3+pn, h*w]
        out_height, out_width = out_size[0], out_size[1]
        grid = _meshgrid(out_height, out_width, source) # [bn, 3+pn, h*w]

        # transform A x (1, x_t, y_t, r1, r2, ..., rn) -> (x_s, y_s)
        # [bn, 2, pn+3] x [bn, pn+3, h*w] -> [bn, 2, h*w]
        T_g = torch.matmul(T, grid)
        x_s = T_g[:,0,:]
        y_s = T_g[:,1,:]

        flow_x = (x_s - grid[:,1,:])
        flow_y = (y_s - grid[:,2,:])

        flow = torch.stack([flow_x, flow_y], 1)
        # flow = flow.reshape([num_batch, 2, grid_h, grid_w])
        flow = flow.reshape([num_batch, 2, out_height, out_width])

        return flow


    def _solve_system(source, target):
        num_batch  = source.size()[0]
        num_point  = source.size()[1]

        np.set_printoptions(precision=8)

        ones = torch.ones(num_batch, num_point, 1).float()
        if torch.cuda.is_available():
            ones = ones.cuda()
        p = torch.cat([ones, source], 2) # [bn, pn, 3]

        p_1 = p.reshape([num_batch, -1, 1, 3]) # [bn, pn, 1, 3]
        p_2 = p.reshape([num_batch, 1, -1, 3])  # [bn, 1, pn, 3]
        d2 = torch.sum(torch.square(p_1-p_2), 3) # p1 - p2: [bn, pn, pn, 3]   final output: [bn, pn, pn]

        r = d2 * torch.log(d2 + 1e-6) # [bn, pn, pn]

        zeros = torch.zeros(num_batch, 3, 3).float()
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        W_0 = torch.cat((p, r), 2) # [bn, pn, 3+pn]
        W_1 = torch.cat((zeros, p.permute(0,2,1)), 2) # [bn, 3, pn+3]
        W = torch.cat((W_0, W_1), 1) # [bn, pn+3, pn+3]

        W_inv = torch.inverse(W.type(torch.float64))


        zeros2 = torch.zeros(num_batch, 3, 2)
        if torch.cuda.is_available():
            zeros2 = zeros2.cuda()
        tp = torch.cat((target, zeros2), 1) # [bn, pn+3, 2]

        T = torch.matmul(W_inv, tp.type(torch.float64)) # [bn, pn+3, 2]
        T = T.permute(0, 2, 1) # [bn, 2, pn+3]


        return T.type(torch.float32)
    

    def torch_Bspline(uv, kl):
        return (
            torch.where(kl == 0, (1 - uv) ** 3 / 6,
                        torch.where(kl == 1, uv ** 3 / 2 - uv ** 2 + 2 / 3,
                                    torch.where(kl == 2, (-3 * uv ** 3 + 3 * uv ** 2 + 3 * uv + 1) / 6,
                                                torch.where(kl == 3, uv ** 3 / 6, torch.zeros_like(uv).to(kl.device)))))
        )


    def torch_transformation(pos, mesh, delta_x, delta_y):
        pos_reg = pos.clone()
        pos_reg[0, :, :] /= delta_y
        pos_reg[1, :, :] /= delta_x

        # print(pos_reg.max(),pos_reg.min())
        pos_floor = pos_reg.floor().long()
        uv = pos_reg - pos_floor
        ij = pos_floor - 2

        kls = torch.stack(torch.meshgrid(torch.arange(4), torch.arange(4), indexing='ij')).reshape(2, -1).T.to(mesh.device)  # [16, 2]

        uv_expanded = uv.unsqueeze(0).expand(kls.size(0), -1, -1, -1)  # [16, 2, H, W]
        kl_expanded = kls.view(-1, 2, 1, 1)  # [16, 2, 1, 1]

        B = torch_Bspline(uv_expanded, kl_expanded)  # [16, 2, H, W]
        B_prod = B.prod(dim=1)  # [16, H, W]

        ij_expanded = ij.unsqueeze(0).expand(kls.size(0), -1, -1, -1)  # [16, 2, H, W]
        pivots = (ij_expanded + 1 + kl_expanded).clamp(0, mesh.size(-1) - 1)  # [16, 2, H, W]

        mesh_values = mesh[:, pivots[:, 0], pivots[:, 1]].permute(1,0,2,3)  # [16, C, H, W]
        result = (B_prod.unsqueeze(1) * mesh_values).sum(dim=0)  # [C, H, W]
        return result

    
    def get_mesh(H,W,source):
        # Create target grid [2, H, W]
        y_coords = torch.linspace(0, 1.0, H, device=source.device)  * H
        x_coords = torch.linspace(0, 1.0, W, device=source.device) * W
        grid = torch.stack(torch.meshgrid(y_coords, x_coords, indexing='ij'))  # [2, H, W]
        return grid
    
    
    def compute_flow_tps(tps_flow, source, delta_x, delta_y, image_size):
        b = source.shape[0]
        H, W = image_size

        tps_flow = tps_flow.permute(1,2,3,0)
        rigid_grid = get_mesh(H,W,source)

        # Compute transformed positions
        flow_list = []
        for i in range(b):
            flow = torch_transformation(rigid_grid, tps_flow[...,i], delta_x, delta_y)

            flow.unsqueeze(0)  # Add batch dimension [1, 2, H, W]
            flow_list.append(flow)
        return torch.stack(flow_list,dim=0)


    def warp_image(image, flow ,out_size):
        # image: [b, c, H, W]
        # flow: [b, 2, H, W]
        num_batch, num_channels, height, width = image.size()
        
        # Create coordinate grid
        H, W = out_size[0],out_size[1]

        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H),
            torch.linspace(-1, 1, W)
        )
        grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).to(image.device)
        
        # Apply flow displacements
        warped_grid = grid.permute(0,3,1,2) + flow

        x_s_flat = warped_grid[:,0,:,:].reshape([-1])
        y_s_flat = warped_grid[:,1,:,:].reshape([-1])

        input_transformed = _interpolate(image, x_s_flat, y_s_flat,(H,W))
        output = input_transformed.reshape([num_batch,  H, W, num_channels])
        return output.permute(0,3,1,2)


    # print(source.shape)
    b, h, w = source.shape[0], grid_h, grid_w  # batch, mesh height, mesh width
    H,W = out_size[0],out_size[1]
    delta_x, delta_y = W/(w-1), H/(h-1)

    T = _solve_system(source, target)
    flow_t = _transform(T, source, U, (h,w))

    flow = compute_flow_tps(flow_t, source, delta_x, delta_y, (H, W))
    output = warp_image(U, flow, (H, W))
    return output
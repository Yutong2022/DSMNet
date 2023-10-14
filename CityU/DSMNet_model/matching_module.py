import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

def euclidean_distance(x, y):
    out = -2 * torch.matmul(x, y)
    out += (x ** 2).sum(dim=-1, keepdim=True)
    out += (y ** 2).sum(dim=-2, keepdim=True)
    return out

class selective_matching_crossview(nn.Module):
    def __init__(self, an2, channels, args, k_nbr = 6, patch_size = 4, candidate = 9, stride = 2):
        super(selective_matching_crossview, self).__init__()
        self.an2 = an2
        self.an = int(an2**0.5)
        self.channels = channels
        # self.k_nbr = k_nbr
        self.k_nbr = args.k_nbr
        self.patch_size = patch_size
        self.candidate = candidate
        self.stride = stride

        # agg
        self.agg1 = nn.Sequential(
            nn.Conv2d(channels*self.k_nbr, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.agg2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, lf_fea):

        # candidate1
        _, _, H, _ = lf_fea.size()
        # window partition
        patch_size = self.patch_size
        step_size = H//patch_size

        # patch_partition
        lf_fea_patch = einops.rearrange(lf_fea, '(N an2) c (pnh psh) (pnw psw) -> (N pnh pnw) (c psh psw) an2', an2=self.an2,
                                   psh=patch_size, psw=patch_size)  # N an c h w k
    
        # select candidate
        # adopt torch.roll to unfold the LF for the acceleration
        candidate_nbr = []
        distance_nbr = []
        length = int(self.candidate**0.5)
        radius = length//2 - 1
        for i in range(self.candidate):
            x_shift = (i//length - radius)*self.stride
            y_sihift = (i%length - radius)*self.stride
            lf_fea_shift = torch.roll(lf_fea, shifts=(x_shift, y_sihift), dims=(2,3))
            lf_fea_shift = einops.rearrange(lf_fea_shift, '(N an2) c (pnh psh) (pnw psw) -> (N pnh pnw) (c psh psw) an2', an2=self.an2,
                            psh=patch_size, psw=patch_size)  # N an c h w ks
            dense_distance = euclidean_distance(lf_fea_patch.permute(0, 2, 1), lf_fea_shift)
            candidate_nbr.append(lf_fea_shift)
            distance_nbr.append(dense_distance)
        candidate = torch.cat(candidate_nbr, dim=2)
        distance = torch.cat(distance_nbr, dim=2)
        _, index_dense = torch.topk(distance, self.k_nbr, dim=2, largest=False, sorted=True)

        # matching
        index_dense = index_dense.unsqueeze(1).expand(-1, self.channels*patch_size*patch_size, -1, -1)
        index_dense = einops.rearrange(index_dense, 'a b pn2 k -> a b (pn2 k)')
        select_patch = torch.gather(candidate, dim=2, index = index_dense)
        select_patch = einops.rearrange(select_patch, '(N pnh pnw) (c psh psw) (an2 k) -> (N an2) (k c) (pnh psh) (pnw psw)', an2 = self.an2, psh=patch_size, psw=patch_size, 
                                        pnh=step_size, pnw=step_size)

        # agg
        select_patch = self.agg1(select_patch)
        lf_fea = torch.cat([lf_fea, select_patch], dim=1)
        lf_fea = self.agg2(lf_fea)

        return lf_fea


class selective_matching_interview(nn.Module):
    def __init__(self, an2, channels, args, k_nbr = 3, patch_size = 4, radius = 7):
        super(selective_matching_interview, self).__init__()
        self.an2 = an2
        self.an = int(an2**0.5)
        self.channels = channels
        # self.k_nbr = k_nbr
        self.k_nbr = args.k_nbr//2
        self.patch_size = patch_size
        self.radius = radius

        # agg
        self.agg1 = nn.Sequential(
            nn.Conv2d(channels*self.k_nbr, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.agg2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, lf_fea):

        B, _, H, W = lf_fea.size()
        patch_size = self.patch_size

        # -------------------------------------------------------------------------------------------- #
        # This methed occupy a large GPU memory. If large GPU memory is limited, you can adopt torch.roll to unfold the LF instead.
        lf_fea_patch = einops.rearrange(lf_fea, '(N an2) c (pnh psh) (pnw psw) -> (N an2) (c psh psw) pnh pnw', an2=self.an2,
                                   psh=patch_size, psw=patch_size)  # rearrange the patch to the channel # [N, C_update, H_update*W_update]
        lf_fea_candidate = F.unfold(lf_fea_patch, kernel_size=(2*self.radius+1, 2*self.radius+1), padding=self.radius, stride=1) # [N, C_update*ks*ks, Hr*Wr]
        lf_fea_candidate = einops.rearrange(lf_fea_candidate, 'N (cu k1 k2) hrwr -> (N hrwr) cu (k1 k2)', k1 = 2*self.radius+1, k2 = 2*self.radius+1) # can be simple

        # dense_distance
        lf_fea_patch_for_calculate = einops.rearrange(lf_fea_patch, 'N CU N1 N2 -> (N N1 N2) CU 1')
        dense_distance = euclidean_distance(lf_fea_patch_for_calculate.permute(0, 2, 1), lf_fea_candidate)

        # select candidate
        _, index_dense = torch.topk(dense_distance, self.k_nbr, dim=2, largest=False, sorted=True)

        # matching
        index_dense = index_dense.unsqueeze(1).expand(-1, self.channels*patch_size*patch_size, -1, -1)
        index_dense = einops.rearrange(index_dense, 'a b pn2 k -> a b (pn2 k)')
        select_patch = torch.gather(lf_fea_candidate, dim=2, index = index_dense)
        # lf_fea_candidate -> lf_fea_patch
        select_patch = einops.rearrange(select_patch, '(N hr wr) (cu psh psw) K -> N (K cu) (hr psh) (wr psw)', 
                                        hr=H//patch_size, wr=W//patch_size, 
                                        psh=patch_size, psw=patch_size)
        # -------------------------------------------------------------------------------------------- #

        # agg
        select_patch = self.agg1(select_patch)
        lf_fea = torch.cat([lf_fea, select_patch], dim=1)
        lf_fea = self.agg2(lf_fea)

        return lf_fea


class selective_matching_ver(nn.Module):
    def __init__(self, an2, channels, args, k_nbr = 6, patch_length = 8):
        super(selective_matching_ver, self).__init__()
        self.an2 = an2
        self.an = int(an2**0.5)
        self.channels = channels
        # self.k_nbr = k_nbr
        self.k_nbr = args.k_nbr
        self.patch_length = patch_length

        # agg
        self.agg1 = nn.Sequential(
            nn.Conv2d(channels*self.k_nbr, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.agg2 = nn.Sequential(
            nn.Conv2d(channels*2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, lf_fea):

        _, _, H, W = lf_fea.size()

        patch_length = self.patch_length
        partition_num = H // patch_length

        # select
        lf_fea_ver = einops.rearrange(lf_fea, '(N u v) c h w -> (N v w) c u h', u=self.an, v=self.an)  # N an c h w k
        lf_fea = einops.rearrange(lf_fea, '(N u v) c (pnh psh) w -> (N pnh) (c u psh) (v w)', u=self.an, v=self.an, psh=patch_length)  # N an c h w k

        dense_distance = euclidean_distance(lf_fea.permute(0, 2, 1), lf_fea)
        _, idx_ki = torch.topk(dense_distance, self.k_nbr, dim=2, largest=False, sorted=True)

        idx_ki = idx_ki.unsqueeze(1).expand(-1, self.channels * self.an * patch_length, -1, -1)
        idx_ki = einops.rearrange(idx_ki, 'B C P K -> B C (P K)')
        select_patch = torch.gather(lf_fea, dim=2, index=idx_ki)
        select_patch = einops.rearrange(select_patch, '(N pnh) (c u psh) (v w k) -> (N v w) (k c) u (pnh psh)'
                                     , u=self.an, v=self.an, pnh=partition_num, psh=patch_length, k=self.k_nbr)

        # agg
        select_patch = self.agg1(select_patch)
        lf_fea_epi = torch.cat([lf_fea_ver, select_patch], dim=1)
        lf_fea_epi = self.agg2(lf_fea_epi)

        # -> lf_fea
        lf_fea_epi = einops.rearrange(lf_fea_epi, '(N v w) c u h -> (N u v) c h w', w = W, u=self.an, v=self.an)

        return lf_fea_epi


class selective_matching_hor(nn.Module):
    def __init__(self, an2, channels, args, k_nbr = 6, patch_length = 8):
        super(selective_matching_hor, self).__init__()
        self.an2 = an2
        self.an = int(an2**0.5)
        self.channels = channels
        # self.k_nbr = k_nbr
        self.k_nbr = args.k_nbr
        self.patch_length = patch_length

        # agg
        self.agg1 = nn.Sequential(
            nn.Conv2d(channels * self.k_nbr, channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.agg2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, lf_fea):

        _, _, H, W = lf_fea.size()
        partition_num = H // self.patch_length

        # ver -> hor
        lf_fea = einops.rearrange(lf_fea, '(N u v) c h w -> (N v u) c w h', u=self.an, v=self.an)  # N an c h w k

        # select
        lf_fea_ver = einops.rearrange(lf_fea, '(N u v) c h w -> (N v w) c u h', u=self.an, v=self.an)  # N an c h w k
        lf_fea = einops.rearrange(lf_fea, '(N u v) c (pnh psh) w -> (N pnh) (c u psh) (v w)', u=self.an, v=self.an, psh=self.patch_length)  # N an c h w k

        dense_distance = euclidean_distance(lf_fea.permute(0, 2, 1), lf_fea)
        _, idx_ki = torch.topk(dense_distance, self.k_nbr, dim=2, largest=False, sorted=True)

        idx_ki = idx_ki.unsqueeze(1).expand(-1, self.channels * self.an * self.patch_length, -1, -1)
        idx_ki = einops.rearrange(idx_ki, 'B C P K -> B C (P K)')
        select_patch = torch.gather(lf_fea, dim=2, index=idx_ki)
        select_patch = einops.rearrange(select_patch, '(N pnh) (c u psh) (v w k) -> (N v w) (k c) u (pnh psh)'
                                     , u=self.an, v=self.an, pnh=partition_num, psh=self.patch_length, k=self.k_nbr)

        # agg
        select_patch = self.agg1(select_patch)
        lf_fea_epi = torch.cat([lf_fea_ver, select_patch], dim=1)
        lf_fea_epi = self.agg2(lf_fea_epi)

        # -> lf_fea
        lf_fea_epi = einops.rearrange(lf_fea_epi, '(N v w) c u h -> (N u v) c h w', w = W, u=self.an, v=self.an)

        # ver -> hor
        lf_fea_epi = einops.rearrange(lf_fea_epi, '(N u v) c h w -> (N v u) c w h', u=self.an, v=self.an)  # N an c h w k

        return lf_fea_epi


class matching_selective(nn.Module):
    def __init__(self, an2, nf, args):
        super(matching_selective, self).__init__()

        # selective_matching
        self.crossview = selective_matching_crossview(an2, nf, args)
        self.interview = selective_matching_interview(an2, nf, args)
        self.ver = selective_matching_ver(an2, nf, args)
        self.hor = selective_matching_hor(an2, nf, args)

        self.fuse = nn.Sequential(
            nn.Conv2d(nf*4, nf, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, kernel_size=3, stride=1, padding=1, bias=False),
        )

    def forward(self, lf_fea):

        feainter = self.crossview(lf_fea)
        feacross = self.interview(lf_fea)
        feaEpiH = self.hor(lf_fea)
        feaEpiV = self.ver(lf_fea)

        buffer = torch.cat((feainter, feacross, feaEpiH, feaEpiV), dim=1)
        buffer = self.fuse(buffer)

        return buffer + lf_fea


class matching_group(nn.Module):
    def __init__(self, args):
        super(matching_group, self).__init__()

        self.scale = args.scale
        self.num_nbr = 5
        self.nf = args.nf
        self.an = args.angRes_in
        self.an2 = args.angRes_in*args.angRes_in

        n_block = 4
        self.n_block = n_block
        Blocks = []
        for i in range(n_block):
            Blocks.append(matching_selective(self.an2, self.nf, args))
        self.Block = nn.Sequential(*Blocks)
        self.conv = nn.Conv2d(self.nf, self.nf, 3, 1, 1)

    def forward(self, lf_fea):

        lf_fea1 = self.Block(lf_fea)

        return self.conv(lf_fea1) + lf_fea
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]

        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)

        self.register_buffer("weightx", kernelx)
        self.register_buffer("weighty", kernely)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


class L_Grad(nn.Module):
    def __init__(self):
        super(L_Grad, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)
        gradient_fused = self.sobelconv(image_fused)

        gradient_joint = torch.max(gradient_A, gradient_B)
        loss_gradient = F.l1_loss(gradient_fused, gradient_joint)
        return loss_gradient

def safe_ssim(x, y):
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    return ssim(x, y, data_range=1.0, size_average=True)
def safe_ms_ssim(x, y):
    x = torch.clamp(x, 0.0, 1.0)
    y = torch.clamp(y, 0.0, 1.0)
    return ms_ssim(x, y, data_range=1.0, size_average=True)

class L_SSIM(nn.Module):
    def __init__(self):
        super(L_SSIM, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_A, image_B, image_fused):
        gradient_A = self.sobelconv(image_A)
        gradient_B = self.sobelconv(image_B)

        weight_A = torch.mean(gradient_A) / (torch.mean(gradient_A) + torch.mean(gradient_B) + 1e-8)
        weight_B = torch.mean(gradient_B) / (torch.mean(gradient_A) + torch.mean(gradient_B) + 1e-8)

        # ssim_A = ssim(image_A, image_fused, data_range=1.0, size_average=True)
        # ssim_B = ssim(image_B, image_fused, data_range=1.0, size_average=True)

        # ssim_A = ms_ssim(image_A, image_fused, data_range=1.0, size_average=True)
        # ssim_B = ms_ssim(image_B, image_fused, data_range=1.0, size_average=True)

        # loss_ssim = weight_A * ssim_A + weight_B * ssim_B
        ssim_A = safe_ssim(image_A, image_fused)
        ssim_B = safe_ssim(image_B, image_fused)
        loss_ssim = 0.5 * ssim_A + 0.5 * ssim_B
        return loss_ssim


class L_Intensity(nn.Module):
    def __init__(self):
        super(L_Intensity, self).__init__()

    def forward(self, image_A, image_B, image_fused):
        intensity_joint = torch.max(image_A, image_B)
        loss_intensity = F.l1_loss(image_fused, intensity_joint)
        return loss_intensity


class fusion_loss_vif(nn.Module):
    def __init__(self):
        super(fusion_loss_vif, self).__init__()
        self.L_Grad = L_Grad()
        self.L_Inten = L_Intensity()
        self.L_SSIM = L_SSIM()

    def forward(self, image_A, image_B, image_fused):
        loss_l1 = 20 * self.L_Inten(image_A, image_B, image_fused)
        loss_gradient = 20 * self.L_Grad(image_A, image_B, image_fused)
        loss_ssim = 10 * (1 - self.L_SSIM(image_A, image_B, image_fused))

        # fusion_loss = loss_l1 + loss_gradient + loss_ssim
        # fusion_loss = loss_l1 + loss_gradient
        fusion_loss = loss_l1
        return fusion_loss, loss_gradient, loss_l1, loss_ssim

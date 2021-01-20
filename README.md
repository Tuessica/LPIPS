# LPIPS
import lpips
import torch
from Calpackage.calpackage import calpackage
from torch.autograd import Variable

img1 = Variable(torch.rand(100, 3, 256, 256)) # image should be RGB, IMPORTANT: normalized to [-1,1]
img2 = Variable(torch.rand(100, 3, 256, 256)) # image should be RGB, IMPORTANT: normalized to [-1,1]
caltool = calpackage()
LPIPS_value_alexnetbase,LPIPS_value_vggnetbase,PSNR, SSIM = caltool.call(img1,img2)
print(LPIPS_value_alexnetbase.mean(),LPIPS_value_vggnetbase.mean(),PSNR, SSIM)
print(LPIPS_value_alexnetbase.detach().numpy().mean(),LPIPS_value_vggnetbase.detach().numpy().mean(),PSNR, SSIM.cpu().detach().numpy())

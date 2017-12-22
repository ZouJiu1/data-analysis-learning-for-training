import matplotlib.pyplot as plt
from skimage import io
import skimage
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#获取图片大小和部分区域
ic=io.imread('./lena512color.tiff')
print(type(ic),ic.shape)
fig,(ax1,ax2,ax3)=plt.subplots(ncols=3,figsize=(15,5))
ax1.imshow(ic)
ict=ic[160:400,140:360,:]
ax2.imshow(ict)
icthr=ic[:,:,0]
ax3.imshow(icthr)
#直方图调整对比对
from skimage import exposure
hist,bin_centers=exposure.histogram(ic)
plt.figure(figsize=(15,5))
plt.subplot(131)
plt.fill_between(bin_centers,hist)
plt.ylim(0)
high_constrat=exposure.rescale_intensity(ic,in_range=(10,216))
plt.subplot(132)
plt.imshow(high_constrat)
equalized=exposure.equalize_hist(ic)
plt.subplot(133)
plt.imshow(ic)
#高斯滤波
from skimage.morphology import disk
from skimage.filters import gaussian
gas1=gaussian(ic,sigma=3)
gas2=gaussian(ic,sigma=5)
fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(10,10))
ax1.imshow(gas1)
ax2.imshow(gas2)
plt.title('高斯滤波')
#边缘检测
from skimage.filters import prewitt, sobel
icc_image=skimage.color.rgb2gray(ic)#灰度化图片，检测只能使用二维灰度图片
edge_prewitt = prewitt(icc_image)
edge_sobel = sobel(icc_image)
fig, (ax_1, ax_2) = plt.subplots(ncols=2, figsize=(10, 5))

# Prewitt 边缘检测
ax_1.imshow(edge_prewitt)

# Sobel 边缘检测
ax_2.imshow(edge_sobel, cmap=plt.cm.gray)
plt.tight_layout()
plt.show()
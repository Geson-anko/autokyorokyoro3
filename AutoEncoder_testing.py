#%% import libs
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

from AutoEncoder_model import AutoEncoder
from hparams import AE_default as hparams
#%% setting
resizer = torchvision.transforms.Resize((128,128))
cmap = plt.get_cmap('jet')
# %% load model
param_file = "params/AutoEncoder_2021-08-29_22-13-14.pth"
model = AutoEncoder(hparams)
model.load_state_dict(torch.load(param_file))
model.eval()
# %% load imgs
imgpath = "data/myimgs/jidori.jpg"
img = torchvision.io.read_image(imgpath)
img = resizer(img) / 255
plt.imshow(img.permute(1,2,0))
plt.show()
# %% test AutoEncoder

with torch.no_grad():
    output = model(img[None,:])
    errormap = model.channels_wise_mse(output,img).squeeze(0)
    output = output.squeeze(0)
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.imshow(img.permute(1,2,0))
    ax1.imshow(output.permute(1,2,0))
    ax2.imshow(errormap.numpy(),cmap=cmap)
    plt.show()

    row,col = model._get_point(errormap[None,:])
    for i in range(row.shape[1]):
        print((row[0][i],col[0][i]))

#%% create dummy img
dummyimg = torch.zeros(128,128)
x_points,y_points = row.reshape(-1)[-3:], col.reshape(-1)[-3:]
for i in range(3):
    dummyimg[x_points[i],y_points[i]] = 1.0

g_x,g_y = int(np.sum(x_points) / 3), int(np.sum(y_points) / 3)
dummyimg[g_x,g_y] = 2.0
plt.imshow(dummyimg,cmap=cmap)
plt.show()

# %% cutting
margin = 3 # pixels

dx,dy = np.abs(x_points - g_x),np.abs(y_points - g_y)
delta_max = np.max(np.concatenate([dx,dy])) + margin
print(delta_max)
start = np.clip([g_x - delta_max, g_y - delta_max],a_min=0,a_max=128)
end = (g_x + delta_max, g_y + delta_max)
print(start,end)
cut = dummyimg[start[0]:end[0],start[1]:end[1]]
plt.imshow(cut,cmap=cmap)
plt.show()

#%% 
cutimg = img[:,start[0]:end[0],start[1]:end[1]]
cutimg = resizer(cutimg)
plt.imshow(cutimg.squeeze(0).permute(1,2,0))
plt.show()

#%%
with torch.no_grad():
    output = model(cutimg[None,:])
    errormap = model.channels_wise_mse(output,cutimg).squeeze(0)
    output = output.squeeze(0)
    fig = plt.figure()
    ax0 = fig.add_subplot(131)
    ax1 = fig.add_subplot(132)
    ax2 = fig.add_subplot(133)
    ax0.imshow(cutimg.permute(1,2,0))
    ax1.imshow(output.permute(1,2,0))
    ax2.imshow(errormap.numpy(),cmap=cmap)
    plt.show()

    row,col = model._get_point(errormap[None,:])
    for i in range(row.shape[1]):
        print((row[0][i],col[0][i]))
import torch
import torch.nn as nn
from torchsummaryX import summary
import pytorch_lightning as pl
from torchvision.utils import make_grid
from hparams import AE_default as hparams
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
from matplotlib.patheffects import withStroke
import numpy as np
from addtional_layers import ResBlocks2d,ConvNorm2d,ConvTransposeNorm2d

class AutoEncoder(pl.LightningModule):
    channels,height,width = (3,128,128)
    def __init__(self,hparams:hparams):
        super().__init__()
        self.input_size = (1,self.channels,self.height,self.width)
        self.output_size = self.input_size
        self.my_hparams = hparams
        self.my_hparams_dict = hparams.get()
        self.model_name = hparams.model_name

        # set criterion
        self.reconstruction_loss = nn.MSELoss()

        # layers
            # encoder
        e_layers = []
        e_channels = [self.channels] + [8*(2**i) for i in range(hparams.encoder_nlayers)]
        for i in range(hparams.encoder_nlayers):
            p = ConvNorm2d(e_channels[i],e_channels[i+1],kernel_size=3,stride=2)
            a = nn.ReLU()
            l = ResBlocks2d(e_channels[i+1],e_channels[i+1],kernel_size=3,nlayers=1)
            e_layers.extend([p,a,l])
        self.encoder = nn.Sequential(*e_layers)

            # decoder 
        d_layers = []
        d_channels = ([8*(2**i) for i in range(hparams.decoder_nlayers)] + [e_channels[-1]])[::-1]
        for i in range(hparams.decoder_nlayers):
            p = ConvTransposeNorm2d(d_channels[i],d_channels[i+1],kernel_size=3,stride=2)
            a = nn.ReLU()
            l = ResBlocks2d(d_channels[i+1],d_channels[i+1],kernel_size=3,nlayers=1)
            d_layers.extend([p,a,l])
        self.decoder = nn.Sequential(
            *d_layers,nn.ConvTranspose2d(d_channels[-1],self.channels,kernel_size=2),nn.Sigmoid()
        )

        # set others
        self.colormap = plt.get_cmap('jet')
        self.effects = [withStroke(linewidth=2,foreground='white')]

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(),self.my_hparams.lr)
        return optim
    
    def on_fit_start(self) -> None:
        self.logger.log_hyperparams(self.my_hparams_dict)

    def training_step(self,batch,idx):
        data = self.preprocess(batch)
        self.view_data = data
        output = self(data)

        # loss
        loss = self.reconstruction_loss(output,data)

        # log
        self.log('reconstruction_loss',loss)
        return loss
    
    alpha:float = 0.7
    cols:int = 4
    @torch.no_grad()
    def on_epoch_end(self) -> None:
        if (self.current_epoch+1) % self.my_hparams.view_interval == 0:
            # input/output logging
            data = self.view_data[:self.my_hparams.max_view_imgs].float()
            data_len = len(data)
            output = self(data)
            errormap = self.channels_wise_mse(output,data)
            rows,cols = self._get_point(errormap)
            errormap = errormap.cpu().numpy()

            fig = plt.figure(figsize=(self.cols,data_len))
            fig.subplots_adjust(wspace=0.1,hspace=0.01)
            for i in range(data_len):
                real_im = data[i].permute(1,2,0).cpu().numpy()
                out_im = output[i].permute(1,2,0).cpu().numpy()
                # real image
                ax0 = fig.add_subplot(data_len,self.cols,self.cols*i + 1)
                ax0.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
                #ax0.axis('off')
                im0 = ax0.imshow(real_im)
                # output image
                ax1 = fig.add_subplot(data_len,self.cols,self.cols*i + 2)
                ax1.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
                #ax1.axis('off')
                im1 = ax1.imshow(out_im)
                # log alpha blended errormap nad real image
                ax2 = fig.add_subplot(data_len,self.cols,self.cols*i + 3)
                ax2.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
                #ax2.axis('off')
                him = self.colormap(errormap[i])
                ax2.imshow(him,alpha=self.alpha,cmap=self.colormap)
                ax2.imshow(real_im,alpha=1-self.alpha)
                # errormap
                ax3 = fig.add_subplot(data_len,self.cols,self.cols*i + 4)
                ax3.tick_params(left=False,labelleft=False,bottom=False,labelbottom=False)
                #ax3.axis('off')
                divider= mpl_toolkits.axes_grid1.make_axes_locatable(ax3)
                cax = divider.append_axes('right','5%',pad='3%')
                im3 = ax3.imshow(errormap[i],cmap=self.colormap)
                fig.colorbar(im3,cax=cax)

                # insert error ranking
                r,c = rows[i],cols[i]
                for idx in [*range(self.my_hparams.view_point_num)][::-1]:
                    p = (r[idx],c[idx])
                    ax2.text(*p,str(idx),fontsize=6,color='black',path_effects=self.effects)
                    #ax3.text(*p,str(idx),fontsize=8,color='black',path_effects=self.effects)
                
                if i == 0:
                    ax0.set_title('input')
                    ax1.set_title('output')
                    ax2.set_title('blend')
                    ax3.set_title('errormap')


            self.logger.experiment.add_figure('Images',fig,self.current_epoch)


    def forward(self,x:torch.Tensor):
        h = self.encoder(x)
        h = self.decoder(h)
        return h
    
    def summary(self,tensorboard=False):
        from torch.utils.tensorboard import SummaryWriter
        dummy = torch.randn(self.input_size)
        summary(self,dummy)

        if tensorboard:
            writer = SummaryWriter(comment=self.model_name)
            writer.add_graph(self,dummy)
            writer.close()

    @torch.no_grad()
    def error_heatmap(self,x:torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        out = self(x)
        error_map = self.channels_wise_mse(out,x)
        return error_map

    def channels_wise_mse(self,output:torch.Tensor,target:torch.Tensor) -> torch.Tensor:
        error = (output - target) ** 2
        errr = error.mean(dim=1)
        return errr
    def preprocess(self,x:torch.Tensor) -> torch.Tensor:
        return (x/255).float()    

    def _get_point(self,heatmap:torch.Tensor) -> tuple:
        assert heatmap.size(1) == self.height
        assert heatmap.size(2) == self.width
        
        hm = heatmap.view(heatmap.size(0),-1)
        sortedidx = torch.argsort(hm,dim=1).flip(1)[:,:self.my_hparams.view_point_num]
        cols = torch.div(sortedidx,self.height,rounding_mode='trunc').cpu().numpy()
        rows = (sortedidx % self.height).cpu().numpy()
        return rows,cols

        


if __name__ == '__main__':
    model = AutoEncoder(hparams)
    model.summary(False)
    dummy = torch.randn(model.input_size)
    print(model.error_heatmap(dummy).shape)
    """
    import torchvision
    import mpl_toolkits.axes_grid1
    import matplotlib.patheffects as pef

    image = torchvision.io.read_image("data/Linnaeus 5 128X128/test/berry/1_128.jpg") /255
    output = image
    heatmap = image.mean(0)
    textpoint = (64,64)
    alpha = 0.5
    print(image.shape)
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.05,hspace=0.1)

    ax0 = fig.add_subplot(1,4,1)
    ax0.set_title('real')
    im0 = ax0.imshow(image.permute(1,2,0))
    
    ax1 = fig.add_subplot(1,4,2)
    ax1.tick_params(left=False,labelleft=False)
    im1 = ax1.imshow(output.permute(1,2,0))

    ax2 = fig.add_subplot(1,4,3)
    ax2.tick_params(left=False,labelleft=False)
    colormap = plt.get_cmap('jet')
    him = colormap(heatmap)
    rim = image.T.numpy()
    ax2.imshow(him,alpha=alpha)
    ax2.imshow(rim,alpha=1-alpha)

    ax3 = fig.add_subplot(1,4,4)
    ax3.tick_params(left=False,labelleft=False)
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax3)
    cax = divider.append_axes('right','5%',pad='3%')
    im3 = ax3.imshow(heatmap,cmap=colormap)
    effects = [pef.withStroke(linewidth=2,foreground='white')]
    ax3.text(*textpoint,'1',fontsize=8,color='black',path_effects=effects)
    fig.colorbar(im3,cax=cax)

    plt.show()
    """
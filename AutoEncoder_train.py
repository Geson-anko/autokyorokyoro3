# %% importing
import torch
import torchvision
import pytorch_lightning as pl
from torch.utils import data as DataUtil
from pytorch_lightning import loggers as pl_loggers
from AutoEncoder_model import AutoEncoder
from hparams import AE_default as hparams
from DatasetLib import get_now
import glob

#%% define dataset
class Linnaeus5(DataUtil.Dataset):
    def __init__(self,using_length:int):
        super().__init__()
        self.files = glob.glob('data/Linnaeus 5 128X128/**/**/*.jpg')[:using_length]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index) -> torch.Tensor:
        return torchvision.io.read_image(self.files[index])
if __name__ == '__main__':
    data_set = Linnaeus5(8000)
# %% 
if __name__ == '__main__':
    EPOCHS = 1000
    batch_size = 512
    hparams.batch_size = batch_size
    model = AutoEncoder(hparams)
    data_loader = DataUtil.DataLoader(data_set,batch_size,shuffle=True,num_workers=0,pin_memory=True,drop_last=True)

    logger = pl_loggers.TensorBoardLogger('ErrormapLog',name='AutoEncoder')
    trainer = pl.Trainer(gpus=1,num_nodes=1,precision=16,max_epochs=EPOCHS,logger=logger,log_every_n_steps=5)
    trainer.fit(model,data_loader)
    torch.save(model.state_dict(),f'params/{model.model_name}_{get_now()}.pth')
# %%

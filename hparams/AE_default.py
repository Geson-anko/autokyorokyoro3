model_name:str = 'AutoEncoder'
max_view_imgs:int = 8
view_interval:int = 10
view_point_num:int = 5

lr:float = 0.001
batch_size:int = 64

encoder_nlayers:int = 4
decoder_nlayers:int = 4

def get() ->dict:
    hparam:dict = {
        "model_name":model_name,
        "lr":lr,
        "batch_size":batch_size,
        "encoder_nlayers":encoder_nlayers,
        "decoder_nlayers":decoder_nlayers,
    }
    return hparam

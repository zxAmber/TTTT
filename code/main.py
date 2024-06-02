import world
import utils
from world import cprint
import torch
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
import register
from register import dataset

# ==============================
utils.set_seed(world.seed)

print(">>SEED:", world.seed)
# ==============================

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
# weight_file = None
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
# if world.tensorboard:
#     w : SummaryWriter = SummaryWriter(
#                                     join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
#                                     )
# else:
# aliyun path
tensorboardPath = '/mnt/data/'
if world.tensorboard:
    w: SummaryWriter = SummaryWriter(
        join(tensorboardPath + 'runs/', time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
    )
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch %10 == 0:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)

        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()
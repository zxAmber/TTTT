import world
import dataloader
import model
from pprint import pprint
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book']:

    dataset = dataloader.Loader(path=parent_directory+"/data/"+world.dataset)
# elif world.dataset == 'lastfm':
#     dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    # 'mf': model.PureMF,
    'cgcn': model.CGCN
}
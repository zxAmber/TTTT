import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Go CGCN")
    parser.add_argument('--bpr_batch', type=int, default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int, default=64,
                        help="the embedding size of CGCN")
    parser.add_argument('--layer', type=int, default=3,
                        help="the layer num of CGCN")
    # -------------------- todo number of k --------------------
    parser.add_argument('--num_clusters', type=int, default=5,
                        help="the number of clusters")
    # -----------------------------------------------------------
    parser.add_argument('--lr', type=float, default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float, default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int, default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float, default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='gowalla',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int, default=1,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str, default="cgcn")
    parser.add_argument('--load', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='cgcn', help='rec-model, support [mf, cgcn]')
    parser.add_argument('--rank', type=int, default=0, help='rec-model, support [mf, cgcn]')
    parser.add_argument('--world_size', type=int, default=0, help='rec-model, support [mf, cgcn]')
    return parser.parse_args()

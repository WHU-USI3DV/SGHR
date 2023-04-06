# Import python dependencies
import argparse

base_dir='./data'
arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

Dirs=add_argument_group('Dirs')
Model_Args=add_argument_group('Model')
Dataset_Args=add_argument_group('Dataset')
Train_Args=add_argument_group("Training_Args")
Val_Args=add_argument_group("Validation_Args")

backbone = 'yoho'

############################################# Base ###################################################
#Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,
                        help="base dir containing the whole project")
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}",
                        help="the dir containing whole datas")
Dirs.add_argument("--save_dir",type=str,default=f"./pre",
                        help="for eval results")
Dirs.add_argument("--backbone",type=str,default=backbone,
                        help='well trained model path')
Dirs.add_argument("--model_fn",type=str,default=f"./checkpoints/{backbone}",
                        help='well trained model path')
Dirs.add_argument("--input_feat_dir",type=str,default=f"{base_dir}",
                        help="eval cache dir")
############################################# Trainset  ##############################################
Dataset_Args.add_argument("--aug_r_range",type=float,default=180,
                        help="rotation augmentation range")
Dataset_Args.add_argument("--aug_t_range",type=float,default=3,
                        help="translation augmentation range")
Dataset_Args.add_argument("--aug_n_range",type=float,default=0.01,
                        help="noise augmentation range")
############################################# backbone  ##############################################
Model_Args.add_argument("--model_type",type=str,default='vlad',
                        help="model type")
Model_Args.add_argument("--vlad_cluster",type=int,default=64,
                        help="the output feature dimension")
Model_Args.add_argument("--vlad_dim",type=int,default=32,
                        help="the output feature dimension")
Model_Args.add_argument("--output_dim",type=int,default=256,
                        help="the output feature dimension")
Model_Args.add_argument("--drop_out",type=str,default=0.3,
                        help="drop out ratio")
############################################ loss ###################################################
Train_Args.add_argument("--loss_type",type=list,default=['l1'],
                        help="loss type")
Train_Args.add_argument("--loss_weights",type=list,default=[1],
                        help="loss weights")
############################################ Train ###################################################
# hyperparameters
Train_Args.add_argument("--batch_size",type=int,default=1,
                        help="Training batch size")
Train_Args.add_argument("--batch_size_val",type=int,default=1,
                        help="Training batch size")
Train_Args.add_argument("--worker_num",type=int,default=0,
                        help="the threads used for dataloader")
Train_Args.add_argument("--epochs",type=int,default=300,
                        help="num of epoches")
Train_Args.add_argument("--multi_gpus",type=bool,default=False,
                        help="whether use the mutli gpus")
Train_Args.add_argument("--lr_init",type=float,default=0.001,
                        help="The initial learning rate")
Train_Args.add_argument("--lr_decay_rate",type=float,default=0.7,
                        help="the decay rate of the learning rate per epoch")
Train_Args.add_argument("--lr_decay_step",type=float,default=50,
                        help="the decay step of the learning rate (how many epoches)")
############################################ saving ##################################################
#log
Train_Args.add_argument("--train_log_step",type=int,default=50,
                        help="logger internal")
Val_Args.add_argument("--val_interval",type=int,default=100,
                        help="the interval to validation")
Val_Args.add_argument("--save_interval",type=int,default=50,
                        help="the interval to save the model")
############################################ pkls ###################################################
# datalist 
Dataset_Args.add_argument("--trainset",type=str,default="3dmatch_train",
                        help="train dataset name")
Dataset_Args.add_argument("--trainlist",type=str,default=f'./train/pkls/train.pkl',
                        help="training tuples (dataset.name, sim_gt)")
Dataset_Args.add_argument("--valset",type=str,default="3dmatch_train",
                        help="validation dataset name")
Dataset_Args.add_argument("--vallist",type=str,default=f'./train/pkls/val.pkl',
                        help="validation tuples (dataset.name, sim_gt)")
Dataset_Args.add_argument("--testset",type=str,default=f'3dmatch',
                        help="name of testset")
Dataset_Args.add_argument("--testlist",type=str,default=f'./train/pkls/test_3dmatch.pkl',
                        help="validation tuples (dataset.name, sim_gt)")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



from train_test_caps import *
from torchvision import datasets, transforms
import models
import os
import sys
from helpers import util,visualize,augmenters
import random
# import dataset
import numpy as np
import torch
# import save_visualizations
import argparse
import cv2
import itertools


def script_train(lr,
                out_dir_pre,
                model_name='cifar_10_alex',
                epoch_stuff=[30,
                60],
                exp = False,
                gpu_id = 0,
                aug_more = 'flip',
                model_to_test = None,
                save_after = 1,
                test_after = 1,
                batch_size = 32,
                batch_size_val = 32,
                weight_decay = 0.004,
                criterion = nn.CrossEntropyLoss()
                ):

    
    # def train_model(out_dir_train,
    #             train_data,
    #             test_data,
    #             batch_size = None,
    #             batch_size_val =None,
    #             num_epochs = 100,
    #             save_after = 20,
    #             disp_after = 1,
    #             plot_after = 10,
    #             test_after = 1,
    #             lr = 0.0001,
    #             dec_after = 100, 
    #             model_name = 'alexnet',
    #             criterion = nn.CrossEntropyLoss(),
    #             gpu_id = 0,
    #             num_workers = 0,
    #             model_file = None,
    #             epoch_start = 0,
    #             margin_params = None,
    #             network_params = None,
    #             just_encoder = False,
    #             weight_decay = 0):



    num_epochs = epoch_stuff[1]
    
    if model_to_test is None:
        model_to_test = num_epochs -1

    epoch_start = 0
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]

    lr = lr
    im_resize = 256
    im_size = 224
    model_file = None
    margin_params = None

    
    out_dir_train = get_out_dir_train_name(out_dir_pre,lr,epoch_stuff, exp, aug_more,weight_decay)
    # out_dir_train = '../scratch/trying_cifar_10'
    print out_dir_train
    # raw_input()

    final_model_file = os.path.join(out_dir_train,'model_'+str(num_epochs-1)+'.pt')
    if os.path.exists(final_model_file):
        print 'skipping',final_model_file
        return
    else:
        print 'not skipping', final_model_file
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    
    list_transforms = []
    if 'flip' in aug_more:
        list_transforms.append(transforms.RandomHorizontalFlip(p=0.5))

    if 'translate' in aug_more: 
        list_transforms.append(torchvision.transforms.RandomAffine(0, translate=(0.15,0.15)))

    list_transforms = list_transforms+[transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    list_transforms_val = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    data_transforms = {}
    data_transforms['train']= transforms.Compose(list_transforms)
    data_transforms['val']= transforms.Compose(list_transforms_val)


    # train_data = dataset.Bp4d_Dataset_with_mean_std_val(train_file, bgr = bgr, binarize = True, mean_std = mean_std, transform = data_transforms['train'],resize=train_resize)
    # test_data = dataset.Bp4d_Dataset_with_mean_std_val(test_file, bgr = bgr, binarize= True, mean_std = mean_std, transform = data_transforms['val'], resize = im_size)



    
    train_data = torchvision.datasets.CIFAR10(root='../data/cifar_10/', train=True,
                                            download=True, transform=data_transforms['train'])
    test_data = torchvision.datasets.CIFAR10(root='../data./cifar_10/', train=False,
                                           download=True, transform=data_transforms['val'])
    print len(train_data)
    print len(test_data)


    network_params = dict(n_classes = 10,init=False)
        
    util.makedirs(out_dir_train)
    
    train_params = dict(out_dir_train = out_dir_train,
                train_data = train_data,
                test_data = test_data,
                batch_size = batch_size,
                batch_size_val = batch_size_val,
                num_epochs = num_epochs,
                save_after = save_after,
                disp_after = 1,
                plot_after = 100,
                test_after = test_after,
                lr = lr,
                dec_after = dec_after, 
                model_name = model_name,
                criterion = criterion,
                gpu_id = gpu_id,
                num_workers = 0,
                model_file = None,
                epoch_start = epoch_start,
                network_params = network_params,
                weight_decay = weight_decay,
                def_data = True)
    test_params = dict(out_dir_train = out_dir_train,
                model_num = model_to_test, 
                train_data = train_data,
                test_data = test_data,
                gpu_id = gpu_id,
                model_name = model_name,
                batch_size_val = batch_size_val,
                criterion = criterion,
                network_params = network_params,
                post_pend = '',
                def_data = True)
    
    # print train_params
    # param_file = os.path.join(out_dir_train,'params.txt')
    # all_lines = []
    # for k in train_params.keys():
    #     str_print = '%s: %s' % (k,train_params[k])
    #     print str_print
    #     all_lines.append(str_print)
    
    train_model(**train_params)
    test_model(**test_params)
        






def get_out_dir_train_name(out_dir_pre, lr, epoch_stuff=[30,60], exp = False, aug_more = ['flip'],weight_decay=0):
    
    num_epochs = epoch_stuff[1]
    if exp:
        dec_after = ['exp',0.96,epoch_stuff[0],1e-6]
    else:
        dec_after = ['step',epoch_stuff[0],0.1]


    post_pend = aug_more+[num_epochs]+dec_after+lr+['weight_decay',weight_decay]

    out_dir_train =  '_'.join([str(val) for val in [out_dir_pre]+post_pend]);

    return out_dir_train






def make_command_str():
    out_dir = '../experiments_pain'

    train_file_pre = '../data/pain/train_test_files_loo_1_thresh_au_only/train_'
    test_file_pre =  '../data/pain/train_test_files_loo_1_thresh_au_only/test_'
    util.mkdir(out_dir)
    exp_name = 'pain_train_1_thresh'
    out_dir_logs = os.path.join(out_dir,exp_name)
    util.mkdir(out_dir_logs)

    out_file_sh = os.path.join(out_dir,exp_name+'.sh')

    wdecay = 0
    route_iter = 3
    n_classes = 6
    model_name = 'vgg_capsule_7_3_with_dropout'
    epoch_stuff = [350,20]
    aug_more = [['flip','rotate','scale_translate']]
    folds = [[0]]
    exp = True
    reconstruct = True
    batch_size_val = 32
    batch_size = 32

    dropout = [0]
    lr_meta = [[0.0001,0.001,0.001]]
    loss_weights = [1.,1.]

    commands_all = []

    params_arr = itertools.product(aug_more,dropout,folds,lr_meta)
    params_arr = [tuple(list(tups)+[val]) for val,tups in enumerate(params_arr)]
    for p in params_arr:
        print p

    for aug_more, dropout, folds, lr, gpu_id in params_arr:
        out_file = os.path.join(out_dir_logs,'_'.join([str(val) for val in aug_more+[dropout]+folds])+'.txt')

        out_dir_pre = os.path.join(out_dir,model_name+'_'+str(route_iter),'au_only_1_pain_thresh_rerun')
        
        command_str = []
        command_str.extend(['python','exp_pain.py'])
        command_str.append('train')
        command_str.extend(['--lr']+lr)
        command_str.extend(['--route_iter', route_iter])
        
        command_str.extend(['--out_dir_pre', out_dir_pre])
        command_str.extend(['--train_file_pre', train_file_pre])
        command_str.extend(['--test_file_pre', test_file_pre])
        command_str.extend(['--n_classes',n_classes])

        command_str.extend(['--batch_size', batch_size])
        command_str.extend(['--batch_size_val', batch_size_val])

        command_str.extend(['--folds']+ folds)
        command_str.extend(['--model_name', model_name])
        command_str.extend(['--epoch_stuff']+ epoch_stuff)
        command_str.extend(['--aug_more']+aug_more)
        command_str.extend(['--dropout', dropout])
        command_str.extend(['--gpu_id', gpu_id])

        command_str.extend(['--loss_weights']+loss_weights)

        if reconstruct:
            command_str.extend(['--reconstruct'])
        if exp:
            command_str.extend(['--exp'])

        command_str.extend(['>', out_file,'&'])
        command_str = ' '.join([str(val) for val in command_str])
        print command_str
        commands_all.append(command_str)

    print out_file_sh
    util.writeFile(out_file_sh, commands_all)


def check():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print torch.__version__
    a = 255*np.ones((100,100,3));
    print type(a[0][0][0])
    print a.shape
    ts = transforms.Compose([
                transforms.ToTensor()
                ])
    output = ts(a)
    print output.size(),torch.min(output),torch.max(output),type(output)


def main(args):

    out_dir_pre = '../experiments/cifar_10_alex'

    script_train([0.001,0.001],
                out_dir_pre,
                model_name='cifar_10_alex',
                epoch_stuff=[25,
                50],
                exp = False,
                gpu_id = 0,
                aug_more = ['flip','translate'],
                save_after = 10,
                test_after = 1,
                batch_size = 200,
                batch_size_val = 200,
                weight_decay = 0,
                criterion = nn.CrossEntropyLoss())



    # # check()
    # # return
    # if len(args)>1 and args[1]=='train':
    #     parser = argparse.ArgumentParser(description='Process some integers.')
    #     parser.add_argument('--lr', metavar='lr', type=float, nargs='+', default = [0.001,0.001], help='learning rate')
    #     parser.add_argument('--route_iter', metavar='route_iter',default = 3, type=int, help='route_iter')
    #     parser.add_argument('--train_file_pre', metavar='train_file_pre',default = '', type=str, help='train_file_pre')
    #     parser.add_argument('--test_file_pre', metavar='test_file_pre',default = '', type=str, help='test_file_pre')
    #     parser.add_argument('--out_dir_pre', metavar='out_dir_pre',default = '', type=str, help='out_dir_pre')
    #     parser.add_argument('--n_classes', metavar='n_classes',default = 0, type=int, help='n_classes')

    #     parser.add_argument('--folds', metavar='folds', type=int, nargs = '+', default = [0,1,2], help='folds')
    #     parser.add_argument('--model_name', metavar='model_name', type=str, default = 'khorrami_capsule_7_3_gray', help='model_name')
    #     parser.add_argument('--epoch_stuff', metavar='epoch_stuff', type=int, default = [15,15], nargs = '+', help='epoch_stuff')
    #     parser.add_argument('--aug_more', dest='aug_more', nargs='+',default = ['flip'], type=str, help='aug_more')
    #     parser.add_argument('--dropout', metavar='dropout', type=float, default = 0., help='dropout')
    #     parser.add_argument('--gpu_id', metavar='gpu_id', type=int, default = 0, help='gpu_id')
    #     parser.add_argument('--reconstruct', dest='reconstruct', default = False, action='store_true', help='reconstruct')
    #     parser.add_argument('--batch_size', metavar='batch_size', default = 32, type=int, help='batch_size')
    #     parser.add_argument('--batch_size_val', metavar='batch_size_val', default = 32, type=int, help='batch_size_val')
    #     parser.add_argument('--exp', dest='exp', default = False, action='store_true', help='exp')
    #     parser.add_argument('--loss_weights', dest='loss_weights', default = [1.,1.],nargs = '+', type = float,help='loss_weights')
        

    #     if len(args)>2:
    #         args = parser.parse_args(args[2:])
    #         args = vars(args)
    #         print args
    #         train_with_vgg(**args)

    # else:
    #     make_command_str()
        

    

if __name__=='__main__':
    main(sys.argv)
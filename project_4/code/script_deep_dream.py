import os
import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream_exp import Deep_Dream
import scipy.misc
import numpy as np
from helpers import util,visualize
import sys
import argparse
# 'num_iterations': 200, 'learning_rate': 1e-05, 'out_dir_im': '../scratch/demo', 'sigma': [0.6, 0.5], 'in_file': '../scratch/rand_im_224.jpg'}
def deep_dream(in_file=None, out_dir_im=None, octave_n=None, num_iterations=None, learning_rate=None, sigma = None):
	 # in_file = os.path.join('../scratch','rand_im_224.jpg')
        # 'deep_dream_new_code','rand_im_224.jpg')
    # scratch/deep_dream_new_code/rand_im_224.jpg

#     background_color = np.float32([200.0, 200.0, 200.0])
# # generate initial random image
#     input_img = np.random.normal(background_color, 8, (224, 224, 3))
#     scipy.misc.imsave(in_file,input_img)


#     return
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'


    # out_dir = os.path.join('../experiments/visualizing',model_name)
    # util.makedirs(out_dir)
    

    model_file = os.path.join('../../eccv_18/experiments', model_name, model_file_name)

    type_data = 'train_test_files_256_color_align'; n_classes = 12;
    train_pre = os.path.join('../data/bp4d',type_data)
    test_pre =  os.path.join('../data/bp4d',type_data)
    

    split_num = 0
    train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
    test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
    assert os.path.exists(train_file)
    assert os.path.exists(test_file)



    mean_file = 'vgg'
    std_file = 'vgg'

    test_im = [line.split(' ')[0] for line in util.readLinesFromFile(test_file)]
    # in_file = test_im[0]

    # bl_khorrami_ck_96/split_0_100_100_0.01_0.01/model_99.pt';
    model = torch.load(model_file)
    print model

    dreamer = Deep_Dream(mean_file,std_file)

    au_list = [1,2,4,6,7,10,12,14,15,17,23,24]
    
    # out_dir_im = os.path.join(out_dir,'au_color_gauss_5e-1_200')
    util.mkdir(out_dir_im)

    for control in range(n_classes):
        au = au_list[control]
        out_file = os.path.join(out_dir_im, str(au))
        print octave_n, sigma
        out_im = dreamer.dream_fc_caps(model,in_file, octave_n = octave_n, control =control ,color = True,num_iterations = num_iterations, learning_rate = learning_rate, sigma= sigma)[:,:,::-1]
        scipy.misc.imsave(out_file+'.jpg', out_im)

    visualize.writeHTMLForFolder(out_dir_im)


def main(args):

    # run_for_caps_exp()
    # # script_explore_lr_steps()

    # return
   

    parser = argparse.ArgumentParser(description='Deep Dream AUs')
    parser.add_argument('--in_file', metavar='in_file', type=str,  default = '../scratch/rand_im_224.jpg',  help='input image')
    parser.add_argument('--out_dir_im', metavar='out_dir_im', default = '../scratch/demo', type=str, help='out directory')
    parser.add_argument('--octave_n', metavar='octave_n', default = 2, type=int, help='number of octaves')
    
    parser.add_argument('--num_iterations', metavar='num_iterations', default = 200, type=int, help='number of iterations')
    parser.add_argument('--learning_rate', metavar='learning_rate', default = 1e-5, type=float, help='learning rate')
    parser.add_argument('--sigma', metavar='sigma', nargs = '+',default = [0.6,0.5], type=float, help='sigma for gaussian')

    # parser.add_argument('--interpolant_type', metavar='interpolant_type',default = 'shepard', type=str, help='type of interpolant. shepard,hardy,local_hardy, or local_shepard')
    # parser.add_argument('--resolution', metavar='resolution',default = 10, type=int, help='resolution of uniform grid')
    # parser.add_argument('--r_sq', metavar='r_sq',default = 0., type=int, help='r_sq for hardy. -1 is mean of all distances. 0 is min. 1 is max. inbetween values linearly interpolate between min and max')
    # parser.add_argument('--num_k', metavar='num_k',default = 10, type=int, help='num k for local')
    # parser.add_argument('--colormap', metavar='colormap',default = 'jet', type=str, help='color map')
    # parser.add_argument('--idx_slice', metavar='idx_slice',default = None, type=int, help='index for slicing in each dimension. defaults to middle slice')

    args = parser.parse_args(args[1:])
    args = vars(args)
    print args
    # main_loop(**args)
    # test_ridge_list(**args)
    deep_dream(**args)


    # plt.ion()
    # plt.figure()
    # plt.imshow(np.random.randint(256, size=(224,224,3)))
    # plt.show()
    
    # print 'hello'

if __name__=='__main__':
    main(sys.argv)
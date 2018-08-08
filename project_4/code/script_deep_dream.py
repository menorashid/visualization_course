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
from analysis import checking_au as ca

import dataset
from torch.autograd import Variable
import glob
import sklearn.metrics.pairwise
import itertools

import shutil

dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
str_replace_viz = ['..',os.path.join(dir_server,'maheen_data/viz_project')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'
au_map = [1,2,4,6,7,10,12,14,15,17,23,24]   

def deep_dream(in_file=None, out_dir_im=None, octave_n=None, num_iterations=None, learning_rate=None, sigma = None, return_caps = False, primary = False, filter_range = range(32), x_range = range(6), y_range = range(6)):

    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'
    

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

    if primary:
        for filter_num,x,y in itertools.product(filter_range, x_range, y_range):

        # range(32):
        #     for x,y in [(2,),(2,3),(3,2),(3,3)]:

            control = [filter_num,x,y]
            print control
            file_str = '_'.join([str(val) for val in control])+'.jpg'
            out_file = os.path.join(out_dir_im, file_str)
            out_im = dreamer.dream_primary_caps(model,in_file, octave_n = octave_n, control =control ,color = True,num_iterations = num_iterations, learning_rate = learning_rate, sigma= sigma, max_jitter = 14)[:,:,::-1]
            print out_file
            scipy.misc.imsave(out_file, out_im)
    else:
        for control,au in enumerate(au_list):
            out_file = os.path.join(out_dir_im, str(au))
            print octave_n, sigma
            out_im = dreamer.dream_fc_caps(model,in_file, octave_n = octave_n, control =control ,color = True,num_iterations = num_iterations, learning_rate = learning_rate, sigma= sigma, return_caps = return_caps)[:,:,::-1]
            

    visualize.writeHTMLForFolder(out_dir_im)


def get_correct_for_au():

    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    results_dir = os.path.join('../../eccv_18/experiments',model_name,'results_'+model_file_name[:model_file_name.rindex('.')])
    assert os.path.exists(results_dir)

    out_dir = os.path.join('../experiments',model_name,'au_info')
    util.makedirs(out_dir)


    test_file = '../data/bp4d/train_test_files_256_color_align/test_0.txt'
    test_im = [line_curr.split(' ')[0] for line_curr in util.readLinesFromFile(test_file)]
    test_im = np.array(test_im)

    labels,preds = ca.collate_files([results_dir])
    preds[preds<=0.5]=0
    preds[preds>0.5]=1
        
    for au_num in range(labels.shape[1]):
        bin_keep = np.logical_and(labels[:,au_num]==preds[:,au_num],labels[:,au_num]==1)
        im_keep = test_im[bin_keep]
        out_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')
        util.writeFile(out_file,im_keep)
        print out_file
        print au_num
        print bin_keep.shape, np.sum(bin_keep), im_keep.shape


def save_au_caps():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')

    model = torch.load(model_file)
    model = model.cuda()

    mean_std = np.array([[93.5940,104.7624,129.1863],[1.,1.,1.]]) #bgr
    std_div = np.array([0.225*255,0.224*255,0.229*255])
    print std_div
    # raw_input()
    bgr= True
    binarize = False
    data_transforms = {}
    data_transforms['val']= transforms.Compose([
                transforms.ToTensor(),
                # lambda x: x*255
                ])

    batch_size_val = 64
    im_resize = 256
    im_size = 224
    

    for au_num in range(8,12):
        # out_file_mags = os.path.join(out_dir,'au_'+str(au_num)+'_mags')
        out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
        util.mkdir(out_dir_caps)

        test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')

        
        test_data = dataset.Bp4d_Dataset_with_mean_std_val(test_file, bgr = bgr, binarize= binarize, mean_std = mean_std, transform = data_transforms['val'], resize = im_size,no_anno = True)

        test_dataloader = torch.utils.data.DataLoader(test_data, 
                            batch_size=batch_size_val,
                            shuffle=False, 
                            num_workers=1)
        

        for num_iter_train,batch in enumerate(test_dataloader):
            print num_iter_train
            data = Variable(batch['image'].float()).cuda()
            mags, caps = model.forward_for_viz(data,return_caps = True)
            
            mags = mags.data.cpu().numpy()
            
            caps = caps.data.cpu().numpy()
            caps = caps[:,au_num,:]
            
            assert np.all(mags[:,au_num]>0.5)

            # out_file_mags = os.path.join(out_dir_mags,str(num_iter_train)+'.npy')
            out_file_caps = os.path.join(out_dir_caps,str(num_iter_train)+'.npy')
            # np.save(out_file_mags, mags)
            print out_file_caps
            np.save(out_file_caps, caps)

def get_direction_differences():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')

    min_mag = 0.9

    for au_num, au_curr in enumerate(au_map):
        out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
        test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')

        out_file = os.path.join(out_dir_caps,'min_idx_'+str(min_mag)+'.txt')

        num_files = glob.glob(os.path.join(out_dir_caps,'*.npy'))
        # print num_files[0]
        num_files = [int(file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')]) for file_curr in num_files]
        num_files.sort()
        # print len(num_files)
        # print num_files

        caps_all = []

        for num_curr in num_files:

            caps_all.append(np.load(os.path.join(out_dir_caps, str(num_curr)+'.npy')))
        caps_all = np.concatenate(caps_all,axis = 0)
        mags_all = np.linalg.norm(caps_all, axis = 1)

        idx_keep = np.where(mags_all>=min_mag)[0]
        caps_rel = caps_all[idx_keep, :]

        

        distances = sklearn.metrics.pairwise.cosine_similarity(caps_rel, dense_output=True)
        dist_tri = np.triu(distances,k = 1)

        min_dist = np.min(distances)
        max_dist = np.max(dist_tri)
        # print au_num, min_dist, max_dist

        idx_min = np.where(dist_tri==min_dist)
        idx_min = np.array([idx_min[0][0],idx_min[1][0]])
        idx_min = idx_keep[idx_min]

        # print au_num, au_curr        
        # print mags_all.shape
        # print np.min(mags_all), np.max(mags_all), np.sum(mags_all>=0.7)
        # print mags_all[idx_min[0]], mags_all[idx_min[1]]
        # print min_dist
        # raw_input()


        idx_min = [str(val) for val in idx_min]
        util.writeFile(out_file,idx_min)

def visualize_max_direction_im():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')
    min_mag = 0.9
    out_file_html = os.path.join(out_dir,'max_au_diff_'+str(min_mag)+'.html')
    ims_html = []
    captions_html = []
    
    
    for au_num,au_curr in enumerate(au_map):
    # range(len(au_map)):
        out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
        test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')
        out_file = os.path.join(out_dir_caps,'min_idx_'+str(min_mag)+'.txt')
        im_files = util.readLinesFromFile(test_file)
        idx_rel = [int(val) for val in util.readLinesFromFile(out_file)]

        ims_curr = [util.getRelPath(im_files[idx_curr].replace(str_replace[0],str_replace[1]),dir_server) for idx_curr in idx_rel]
        # range(0,len(im_files),100)]
        # ]
        # for idx_curr in idx_rel:
        #     im_curr = im_files[idx_curr]
        #     im_curr = im_curr.replace(str_replace[0],str_replace[1])
        #     im_curr = util.getRelPath(dir_server, im_curr)
        #     ims_curr.append(im_curr)

        ims_html.append(ims_curr)
        captions_html.append([str(au_curr)]*len(ims_curr))

    visualize.writeHTML(out_file_html, ims_html, captions_html, 256, 256)
    print out_file_html.replace(str_replace_viz[0],str_replace_viz[1]).replace(dir_server,click_str)



def deep_dream_au_max_dist():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')
    min_mag = 0.7
    
    


    mean_file = 'vgg'
    std_file = 'vgg'

    model = torch.load(model_file)
    print model

    dreamer = Deep_Dream(mean_file,std_file)
    octave_n = 2
    num_iterations = 100
    learning_rate = 1e-1
    sigma = [0.6,0.5]
    max_jitter_all = [0,5,10,20]
    
    for max_jitter in max_jitter_all:  
        out_dir_im = os.path.join(out_dir,'max_dist_dd_'+str(max_jitter))
        util.mkdir(out_dir_im)
  
        out_file_html = out_dir_im+'.html'
        ims_html = []
        captions_html = []

        for au_num,au_curr in enumerate(au_map):
            out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
            test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')
            out_file = os.path.join(out_dir_caps,'min_idx_'+str(min_mag)+'.txt')
            im_files = util.readLinesFromFile(test_file)
            idx_rel = [int(val) for val in util.readLinesFromFile(out_file)]
            ims_curr = [im_files[idx_curr] for idx_curr in idx_rel]
            
            ims_html_row = []
            captions_html_row = []
            
            for idx_im, im_curr in enumerate(ims_curr):
                file_str = '_'.join([str(val) for val in [au_curr,idx_im]])+'.jpg'
                out_file = os.path.join(out_dir_im,file_str)
                out_im = dreamer.dream_fc_caps(model,im_curr, octave_n = octave_n, control =au_num ,color = True,num_iterations = num_iterations, learning_rate = learning_rate, sigma= sigma, max_jitter = max_jitter)[:,:,::-1]
                scipy.misc.imsave(out_file, out_im)
                
                ims_html_row.append(util.getRelPath(im_curr.replace(str_replace[0],str_replace[1]), dir_server))
                captions_html_row.append('org')
                
                ims_html_row.append(util.getRelPath(out_file.replace(str_replace_viz[0],str_replace_viz[1]), dir_server))
                captions_html_row.append(file_str)
            
            ims_html.append(ims_html_row)
            captions_html.append(captions_html_row)
            # break

        visualize.writeHTML(out_file_html, ims_html, captions_html, 256,256)
        print out_file_html.replace(str_replace_viz[0],str_replace_viz[1]).replace(dir_server,click_str)
    

def visualize_primary_html():
    dir_meta = '../scratch'
    eg_dirs = ['demo_primary','demo_primary_f19','demo_primary_m17']
    # for eg_dir in eg_dirs:
        # im_files = glob.glob(os.path.join(dir_meta,eg_dir,'*.jpg.jpg'))
        # for im_file in im_files:
        #     out_file = im_file[:im_file.rindex('.')]
        #     # print im_file, out_file
        #     # raw_input()
        #     shutil.move(im_file, out_file)
        # visualize.writeHTMLForFolder(os.path.join(dir_meta, eg_dir))
    out_file_html = os.path.join(dir_meta, 'all_demos.html')
    im_html = []
    captions_html = []

    for filter_num,x,y in itertools.product(range(32),range(2,5),range(2,5)):
        im_row = []
        caption_row = []
        for eg_dir in eg_dirs:
            caption_curr = '_'.join([str(val) for val in [filter_num,x,y]])
            in_file = os.path.join(dir_meta, eg_dir, caption_curr+'.jpg')
            in_file = util.getRelPath(in_file.replace(str_replace_viz[0],str_replace_viz[1]),dir_server)
            im_row.append(in_file)
            caption_row.append(caption_curr)
        im_html.append(im_row)
        captions_html.append(caption_row)

    visualize.writeHTML(out_file_html,im_html, captions_html,224,224)

#(2,3) 23,28
#(4,3) 5,6,9,21
#(3,3) 0,11,22




def main(args):

    # get_correct_for_au()
    # save_au_caps()
    # get_direction_differences()
    # visualize_max_direction_im()
    # deep_dream_au_max_dist()

    # return
    # run_for_caps_exp()
    # # script_explore_lr_steps()
    # visualize_primary_html()
    return
   

    parser = argparse.ArgumentParser(description='Deep Dream AUs')
    parser.add_argument('--in_file', metavar='in_file', type=str,  default = '../scratch/rand_im_224.jpg',  help='input image')
    parser.add_argument('--out_dir_im', metavar='out_dir_im', default = '../scratch/demo', type=str, help='out directory')
    parser.add_argument('--octave_n', metavar='octave_n', default = 2, type=int, help='number of octaves')
    
    parser.add_argument('--num_iterations', metavar='num_iterations', default = 200, type=int, help='number of iterations')
    parser.add_argument('--learning_rate', metavar='learning_rate', default = 5e-1, type=float, help='learning rate')
    parser.add_argument('--sigma', metavar='sigma', nargs = '+',default = [0.6,0.5], type=float, help='sigma for gaussian')
    parser.add_argument('--return_caps', dest='return_caps', default = False, action = 'store_true', help='return caps')
    parser.add_argument('--primary', dest='primary', default = False, action = 'store_true', help='primary caps')

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
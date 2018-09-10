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

import script_deep_dream as sdd
dir_server = '/disk3'
str_replace = ['..',os.path.join(dir_server,'maheen_data/eccv_18')]
str_replace_viz = ['..',os.path.join(dir_server,'maheen_data/viz_project')]
click_str = 'http://vision3.idav.ucdavis.edu:1000'
au_map = [1,2,4,6,7,10,12,14,15,17,23,24]   

def most_ext_per_idx():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')

    
    for au_num, au_curr in enumerate(au_map):
        out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
        test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')
        
        out_file = os.path.join(out_dir,'au_'+str(au_num)+'_max_idx.txt')

        im_files = util.readLinesFromFile(test_file)
        ids_all = np.array([file_curr.split('/')[-3] for file_curr in im_files])
        ids_uni = np.unique(ids_all)
        im_files = np.array(im_files)

        num_files = glob.glob(os.path.join(out_dir_caps,'*.npy'))
        
        num_files = [int(file_curr[file_curr.rindex('/')+1:file_curr.rindex('.')]) for file_curr in num_files]
        num_files.sort()
        
        caps_all = []

        for num_curr in num_files:
            caps_all.append(np.load(os.path.join(out_dir_caps, str(num_curr)+'.npy')))

        caps_all = np.concatenate(caps_all,axis = 0)
        mags_all = np.linalg.norm(caps_all, axis = 1)

        high_mags = []
        for id_uni in ids_uni:
            bin_rel = ids_all==id_uni
            idx_max = np.argmax(mags_all[bin_rel])
            im_max = im_files[bin_rel][idx_max]
            print im_max
            high_mags.append(im_max)
            
        print high_mags
        print out_file
        util.writeFile(out_file, high_mags)


def visualize_max_au_per_id():
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'

    model_file = os.path.join('../../eccv_18/experiments',model_name,model_file_name)

    out_dir = os.path.join('../experiments',model_name,'au_info')

    out_file_html = os.path.join(out_dir,'max_idx.html')
    
    ims_html = []
    captions_html = []
    for au_num, au_curr in enumerate(au_map):
        out_dir_caps = os.path.join(out_dir,'au_'+str(au_num)+'_caps')
        test_file = os.path.join(out_dir,'correct_'+str(au_num)+'.txt')
        out_file = os.path.join(out_dir,'au_'+str(au_num)+'_max_idx.txt')
        ims_row = util.readLinesFromFile(out_file)
        ims_row = [util.getRelPath(file_curr.replace(str_replace[0],str_replace[1]),dir_server) for file_curr in ims_row]
        captions_row = [str(au_curr)]*len(ims_row)
        ims_html.append(ims_row)
        captions_html.append(captions_row)
        # print captions_row
        # raw_input()

        

    visualize.writeHTML(out_file_html, ims_html, captions_html, 224,224)



def dream_primary_aus_specific(vgg_ft=False):

    ids_chosen = ['F001', 'F019', 'M008', 'M014']
    # ids_chosen = ['F004', 'F007', 'F010', 'F013', 'F016', 'F022', 'M005', 'M011', 'M017']+['F001', 'F019', 'M008', 'M014']

    
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])
    out_dir_in = out_dir

    if vgg_ft:
        model_name = 'vgg_face_finetune/bp4d_256_train_test_files_256_color_align_0_False_MultiLabelSoftMarginLoss_10_step_5_0.1_0_0.0001_0.001_0.001_False'
        out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])

    

    au_map = [1,2,4,6,7,10,12,14,15,17,23,24]
    # range_x_map = {1:[3],
    #                 2:[3],
    #                 4:[3],
    #                 }
    range_x_map = {1:[3],
                    2:[3],
                    4:[3],
                    6:[3],
                    7:[3],
                    10:[3],
                    12:[3],
                    14:[3],
                    15:[3],
                    17:[3],
                    23:[3],
                    24:[3],
                    }
    range_y_map = {1:[1,2],
                    2:[1,2],
                    4:[1,2],
                    6:[1,2],
                    7:[1,2],
                    10:[3,4],
                    12:[3,4],
                    14:[3,4],
                    15:[3,4],
                    17:[3,4],
                    23:[3,4],
                    24:[3,4],
                    }
    filter_range = [i for i in range(32) if i not in [0,5,6,9,21] ]
    # range(5,6)

    for au_num, au_curr in enumerate(au_map):
        if au_num<0:
            continue

    # (au_curr, range_x, range_y) in enumerate(au_params):
        # if vgg_ft:
        #     range_x = range(1,6)
        #     range_y = range(1,6)
        # else:
        range_x = range_x_map[au_curr]
        range_y = range_y_map[au_curr]

        out_dir_im = os.path.join(out_dir, 'max_viz_primary_'+str(au_curr))
        util.mkdir(out_dir_im)
        
        out_file_html = os.path.join(out_dir_im,'viz_all.html')
        ims_html = []
        captions_html = []


        max_au_file = os.path.join(out_dir_in,'au_'+str(au_num)+'_max_idx.txt')
        im_files = util.readLinesFromFile(max_au_file)
        ids = [file_curr.split('/')[-3] for file_curr in im_files]
        rel_idx = np.in1d(ids,ids_chosen)
        print rel_idx
        rel_files = np.array(im_files)[rel_idx] 
        print rel_files

        for rel_file in rel_files:
            id_curr = rel_file.split('/')[-3]
            out_dir_im_curr = os.path.join(out_dir_im, id_curr)
            util.mkdir(out_dir_im_curr)
            
            if vgg_ft:
                tups = sdd.deep_dream_vgg_ft(in_file=rel_file, out_dir_im=out_dir_im_curr, octave_n=2, num_iterations=200, learning_rate=3e-2, sigma = [0.6,0.5], return_caps = False, primary = True, filter_range = filter_range, x_range = range_y, y_range = range_x)
                str_all = ['_'.join([str(val) for val in tup_curr])+'.jpg' for tup_curr in tups]
            else:
                sdd.deep_dream(in_file=rel_file, out_dir_im=out_dir_im_curr, octave_n=2, num_iterations=200, learning_rate=5e-1, sigma = [0.6,0.5], return_caps = False, primary = True, filter_range = filter_range, x_range = range_y, y_range = range_x)
                str_all = ['_'.join([str(val) for val in [filt,y,x]])+'.jpg' for filt,y,x in itertools.product(filter_range, range_y,range_x)]

            visualize.writeHTMLForFolder(out_dir_im_curr)
            ims_row = [util.getRelPath(os.path.join(out_dir_im_curr,str_curr),dir_server) for str_curr in str_all]
            ims_html.append(ims_row)
            captions_html.append(str_all)

        ims_html= np.array(ims_html).T
        captions_html = np.array(captions_html).T

        visualize.writeHTML(out_file_html, ims_html,captions_html,224,224)

        # raw_input()



            # raw_input()




    # in_file = 


    # deep_dream(in_file=None, out_dir_im=None, octave_n=None, num_iterations=None, learning_rate=None, sigma = None, return_caps = False, primary = False, filter_range = range(32), x_range = range(6), y_range = range(6))

def make_primary_au_specific_comparative_html(vgg_ft=False):
    ids_chosen = ['F001', 'F019', 'M008', 'M014']
    ids_chosen = ['F004', 'F007', 'F010', 'F013', 'F016', 'F022', 'M005', 'M011', 'M017']

    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])
    
    out_dir_in = out_dir

    if vgg_ft:
        model_name = 'vgg_face_finetune/bp4d_256_train_test_files_256_color_align_0_False_MultiLabelSoftMarginLoss_10_step_5_0.1_0_0.0001_0.001_0.001_False'
        out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])

    
    
    out_dir_html = os.path.join(out_dir,'max_viz_primary_comparison_htmls')
    util.mkdir(out_dir_html)

    au_map = [1,2,4,6,7,10,12,14,15,17,23,24]
    # id_curr = 'M008'
    
    filter_range = [4507,339,498]

    aus_chosen = [1,2,4]
    pos_x = 3
    pos_y = 2
    
    # aus_chosen = [12,14,15,23,24]
    # pos_x = 3
    # pos_y = 3
    

    for id_curr in ids_chosen:
        out_file_html = '_'.join([str(val) for val in [id_curr,'aus']+aus_chosen])
        out_file_html = os.path.join(out_dir_html,out_file_html+'.html')
        ims_html = []
        captions_html =[]
    
        for au_curr in aus_chosen:
            au_num = au_map.index(au_curr)
            im_row = []
            caption_row = []

            out_dir_im = os.path.join(out_dir, 'max_viz_primary_'+str(au_curr))
            util.mkdir(out_dir_im)
            
            # out_file_html = os.path.join(out_dir_im,'viz_all.html')
            # ims_html = []
            # captions_html = []


            max_au_file = os.path.join(out_dir_in,'au_'+str(au_num)+'_max_idx.txt')
            im_files = util.readLinesFromFile(max_au_file)
            ids = [file_curr.split('/')[-3] for file_curr in im_files]
            rel_idx = np.in1d(ids,ids_chosen)
            print rel_idx
            rel_file = np.array(im_files)[rel_idx] 
            rel_file = [file_curr for file_curr in rel_file if id_curr in file_curr]
            print rel_file

            assert len(rel_file)<=1
            if len(rel_file)==0:
                continue
            rel_file = rel_file[0]
            id_curr = rel_file.split('/')[-3]
            out_dir_im_curr = os.path.join(out_dir_im, id_curr)
            util.mkdir(out_dir_im_curr)
            
            str_all = ['_'.join([str(val) for val in [filt,y,x]])+'.jpg' for filt,y,x in itertools.product(filter_range, [pos_y],[pos_x])]
            ims_row = [util.getRelPath(os.path.join(out_dir_im_curr,str_curr),dir_server) for str_curr in str_all]
            ims_row = [util.getRelPath(rel_file.replace(str_replace[0],str_replace[1]),dir_server)]+ims_row
            str_all = ['AU '+str(au_curr)]+str_all

            ims_html.append(ims_row)
            captions_html.append(str_all)

        # ims_html= np.array(ims_html).T
        # captions_html = np.array(captions_html).T

        visualize.writeHTML(out_file_html, ims_html,captions_html,224,224)
        print out_file_html



        # raw_input()

    # # ,6,7,10,12,14,15,17,23,24]

    # # range_x_map = {1:[3],
    # #                 2:[3],
    # #                 4:[3],
    # #                 }
    # range_x_map = {1:[3],
    #                 2:[3],
    #                 4:[3],
    #                 6:[3],
    #                 7:[3],
    #                 10:[3],
    #                 12:[3],
    #                 14:[3],
    #                 15:[3],
    #                 17:[3],
    #                 23:[3],
    #                 24:[3],
    #                 }
    # range_y_map = {1:[2],
    #                 2:[2],
    #                 4:[2],
    #                 6:[2],
    #                 7:[2],
    #                 10:[3],
    #                 12:[3],
    #                 14:[3],
    #                 15:[3],
    #                 17:[3],
    #                 23:[3],
    #                 24:[3],
    #                 }
    # filter_range = [0,5,6,9,21]
    # # range(5,6)

    # for au_num, au_curr in enumerate(au_map):
    #     if au_num<6:
    #         continue

    # # (au_curr, range_x, range_y) in enumerate(au_params):
    #     range_x = range_x_map[au_curr]
    #     range_y = range_y_map[au_curr]

    #     out_dir_im = os.path.join(out_dir, 'max_viz_primary_'+str(au_curr))
    #     util.mkdir(out_dir_im)
        
    #     out_file_html = os.path.join(out_dir_im,'viz_all.html')
    #     ims_html = []
    #     captions_html = []


    #     max_au_file = os.path.join(out_dir,'au_'+str(au_num)+'_max_idx.txt')
    #     im_files = util.readLinesFromFile(max_au_file)
    #     ids = [file_curr.split('/')[-3] for file_curr in im_files]
    #     rel_idx = np.in1d(ids,ids_chosen)
    #     print rel_idx
    #     rel_files = np.array(im_files)[rel_idx] 
    #     print rel_files

    #     for rel_file in rel_files:
    #         id_curr = rel_file.split('/')[-3]
    #         out_dir_im_curr = os.path.join(out_dir_im, id_curr)
    #         util.mkdir(out_dir_im_curr)
    #         sdd.deep_dream(in_file=rel_file, out_dir_im=out_dir_im_curr, octave_n=2, num_iterations=200, learning_rate=5e-1, sigma = [0.6,0.5], return_caps = False, primary = True, filter_range = filter_range, x_range = range_y, y_range = range_x)
    #         visualize.writeHTMLForFolder(out_dir_im_curr)

    #         str_all = ['_'.join([str(val) for val in [filt,y,x]])+'.jpg' for filt,y,x in itertools.product(filter_range, range_y,range_x)]
    #         ims_row = [util.getRelPath(os.path.join(out_dir_im_curr,str_curr),dir_server) for str_curr in str_all]
    #         ims_html.append(ims_row)
    #         captions_html.append(str_all)

    #     ims_html= np.array(ims_html).T
    #     captions_html = np.array(captions_html).T

    #     visualize.writeHTML(out_file_html, ims_html,captions_html,224,224)

def get_vgg_ft_stats():

    ids_chosen = ['F001', 'F019', 'M008', 'M014']
    # ids_chosen = ['F004', 'F007', 'F010', 'F013', 'F016', 'F022', 'M005', 'M011', 'M017']+['F001', 'F019', 'M008', 'M014']

    
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])
    out_dir_in = out_dir

    # if vgg_ft:
    model_name = 'vgg_face_finetune/bp4d_256_train_test_files_256_color_align_0_False_MultiLabelSoftMarginLoss_10_step_5_0.1_0_0.0001_0.001_0.001_False'
    out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])

    

    au_map = [1,2,4,6,7,10,12,14,15,17,23,24]
    # range_x_map = {1:[3],
    #                 2:[3],
    #                 4:[3],
    #                 }
    range_x_map = {1:[3],
                    2:[3],
                    4:[3],
                    6:[3],
                    7:[3],
                    10:[3],
                    12:[3],
                    14:[3],
                    15:[3],
                    17:[3],
                    23:[3],
                    24:[3],
                    }
    range_y_map = {1:[1,2],
                    2:[1,2],
                    4:[1,2],
                    6:[1,2],
                    7:[1,2],
                    10:[3,4],
                    12:[3,4],
                    14:[3,4],
                    15:[3,4],
                    17:[3,4],
                    23:[3,4],
                    24:[3,4],
                    }
    # range(5,6)

    filter_dict = {}

    for au_num, au_curr in enumerate(au_map):
        
        range_x = range_x_map[au_curr]
        range_y = range_y_map[au_curr]

        out_dir_im = os.path.join(out_dir, 'max_viz_primary_'+str(au_curr))
        util.mkdir(out_dir_im)
        
        out_file_html = os.path.join(out_dir_im,'viz_all.html')
        ims_html = []
        captions_html = []


        max_au_file = os.path.join(out_dir_in,'au_'+str(au_num)+'_max_idx.txt')
        im_files = util.readLinesFromFile(max_au_file)
        ids = [file_curr.split('/')[-3] for file_curr in im_files]
        rel_idx = np.in1d(ids,ids_chosen)
        # print rel_idx
        rel_files = np.array(im_files)[rel_idx] 
        # print rel_files

        filters_rel = []
        for rel_file in rel_files:
            id_curr = rel_file.split('/')[-3]
            out_dir_im_curr = os.path.join(out_dir_im, id_curr)
            util.mkdir(out_dir_im_curr)

            im_files = glob.glob(os.path.join(out_dir_im_curr,'*.jpg'))
            filters_rel += [int(os.path.split(im_file_curr)[1].split('_')[0]) for im_file_curr in im_files]
        
        filter_dict[au_curr] = filters_rel

    for au_curr in filter_dict.keys():
        out_file_au_count = os.path.join(out_dir,'au_count_'+str(au_curr)+'.txt')
        filters = filter_dict[au_curr]
        lines = []
        for val in set(filters):
            count = filters.count(val)
            lines.append(' '.join([str(val_curr) for val_curr in [val, count]]))
        util.writeFile(out_file_au_count,lines)


def pick_aus_to_keep():
    model_name = 'vgg_face_finetune/bp4d_256_train_test_files_256_color_align_0_False_MultiLabelSoftMarginLoss_10_step_5_0.1_0_0.0001_0.001_0.001_False'
    out_dir = os.path.join('../experiments',model_name,'au_info').replace(str_replace_viz[0],str_replace_viz[1])
    au_map = [1,2,4,6,7,10,12,14,15,17,23,24]
    for au_num, au_curr in enumerate(au_map):

        au_file = os.path.join(out_dir, 'au_count_'+str(au_curr)+'.txt')
        lines = util.readLinesFromFile(au_file)
        print 'AU ',au_curr 
        for line_curr in lines:
            print line_curr
        print ''
        # raw_input()


def main(args):
    # pick_aus_to_keep()
    # get_vgg_ft_stats()

    make_primary_au_specific_comparative_html(True)

    # F001, F019, M008, M014
    # dream_primary_aus_specific(True)
    # visualize_max_au_per_id()
    # most_ext_per_idx()
   
if __name__=='__main__':
    main(sys.argv)
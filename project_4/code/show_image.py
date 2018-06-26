import os
import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream_exp import Deep_Dream
# dream, dream_fc
import scipy.misc
import numpy as np
from helpers import util,visualize
# from models import 
def for_imagenet():

    import torchvision
    from torchvision import models
    import torch.utils.model_zoo as model_zoo
    from torchvision.models.vgg import model_urls

    out_dir = '../scratch/deep_dream_new_code'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # input_img = scipy.misc.imread('./sky.jpg')
    in_file = os.path.join(out_dir,'rand_im_224.jpg')
    cat = 161
    out_file = in_file[:in_file.rindex('.')]+'_'+str(cat)+'_fc_all_vgg16.jpg'
    # os.path.join(out_dir,'try.jpg')
    # input_img = Image.fromarray(np.uint8(np.random.randint(256,size = (224,224,3))))

    # scipy.misc.imsave(in_file,input_img)

    # return
    input_img = Image.open(in_file)
    input_tensor = img_transform(input_img).unsqueeze(0)
    input_np = input_tensor.numpy()

    # model = resnet50(pretrained=True)

    model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
    model = torchvision.models.vgg16(pretrained=True)

    # vgg = models.vgg16(pretrained=True)
    model = model.cuda()
    
    # print model.fc
    # return
    print model
    # return


    if torch.cuda.is_available():
        model = model.cuda()
    for param in model.parameters():
        param.requires_grad = False

    output_im = dream_fc(model, input_np,control = cat)
    print output_im.shape
    
    scipy.misc.imsave(out_file,output_im)

def save_rand_im(im_size,out_file):
    background_color = 128.
    input_img = Image.fromarray(np.uint8(np.random.normal(background_color, 8, size = im_size)))
    scipy.misc.imsave(out_file,input_img)

def run_for_conv_exp():
    in_file = os.path.join('../scratch','rand_im_96.jpg')
    # save_rand_im((224,224,3),in_file)


    # return 
    print 'hello'
    model_name = 'bl_khorrami_ck_96_nobn_pixel_augment_255_range_trans_fix/split_0_400_300_0.01_0.01'
    model_file_name =  'model_399.pt'
    out_dir = os.path.join('../experiments/visualizing',model_name)
    util.makedirs(out_dir)
    out_file = os.path.join(out_dir,'try.jpg')
    model_file = os.path.join('../../eccv_18/experiments', model_name, model_file_name)

    type_data = 'train_test_files'; n_classes = 8;
    train_pre = os.path.join('../data/ck_96',type_data)
    test_pre =  os.path.join('../data/ck_96',type_data)

    split_num = 0
    train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
    test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

    test_im = [line.split(' ')[0] for line in util.readLinesFromFile(test_file)]
    # in_file = test_im[0]

    # bl_khorrami_ck_96/split_0_100_100_0.01_0.01/model_99.pt';
    model = torch.load(model_file)
    print model

    dreamer = Deep_Dream(mean_file,std_file)
    out_im = dreamer.dream_fc(model,in_file)
    print 'in_file',in_file
        # in_file) 
    print out_im.shape
    scipy.misc.imsave(out_file, out_im)

    # params reasonable
    # octave_n = 3
    # sigma = np.linspace(0.6,0.4,octave_n)
    # learning_rate = 5e-3
    # max_jitter = 10
    # num_iterations = 20

def run_for_caps_exp():
    in_file = os.path.join('../scratch','rand_im_96.jpg')
    # save_rand_im((224,224,3),in_file)


    # return 
    print 'hello'

    model_name = 'khorrami_capsule_7_33/ck_96_4_reconstruct_True_True_all_aug_margin_False_wdecay_0_600_step_600_0.1_0.001_0.001_0.001'
    model_file_name =  'model_599.pt'

    out_dir = os.path.join('../experiments/visualizing',model_name)
    util.makedirs(out_dir)
    

    model_file = os.path.join('../../eccv_18/experiments', model_name, model_file_name)

    type_data = 'train_test_files'; n_classes = 8;
    train_pre = os.path.join('../data/ck_96',type_data)
    test_pre =  os.path.join('../data/ck_96',type_data)
    
    out_file = os.path.join(out_dir,'blur_less.jpg')

    split_num = 4
    train_file = os.path.join(train_pre,'train_'+str(split_num)+'.txt')
    test_file = os.path.join(test_pre,'test_'+str(split_num)+'.txt')
    mean_file = os.path.join(train_pre,'train_'+str(split_num)+'_mean.png')
    std_file = os.path.join(train_pre,'train_'+str(split_num)+'_std.png')

    test_im = [line.split(' ')[0] for line in util.readLinesFromFile(test_file)]
    # in_file = test_im[0]

    # bl_khorrami_ck_96/split_0_100_100_0.01_0.01/model_99.pt';
    model = torch.load(model_file)
    print model

    dreamer = Deep_Dream(mean_file,std_file)
    for control in range(1):
        out_file = os.path.join(out_dir, str(control)+'_blur_less.jpg')
        out_im = dreamer.dream_fc_caps(model,in_file, octave_n = 2, control =control ,learning_rate = 5e-2, num_iterations = 80, sigma = [0.1,0.1])

    # print 'in_file',in_file
    #     # in_file) 
    # print out_im.shape
        scipy.misc.imsave(out_file, out_im)
    visualize.writeHTMLForFolder(out_dir)

    # params reasonable
    # octave_n = 2
    # sigma = np.linspace(0.6,0.4,octave_n)
    # learning_rate = 5e-2
    # max_jitter = 14
    # num_iterations = 80


def script_explore_lr_steps():

    in_file = os.path.join('../scratch','rand_im_224.jpg')
    
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'


    out_dir = os.path.join('../experiments/visualizing',model_name)
    util.makedirs(out_dir)
    

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

    au_list = [1]
    # ,2,4,6,7,10,12,14,15,17,23,24]
    
    out_dir_im = os.path.join(out_dir,'au_color_lr_iter_'+str(au_list[0]))
    util.mkdir(out_dir_im)


    iter_range = [10,200]
    lr_range = [1e-1]
    # ,5e-2,5e-3]

    # for control in range(len(au_list)):
    control = 0
    for lr in lr_range:
        for n_iter in iter_range:
            au = au_list[control]
            out_file = os.path.join(out_dir_im, '_'.join([str(val) for val in [au,lr,n_iter]])+'.jpg')

            out_im = dreamer.dream_fc_caps(model,in_file, octave_n = 2, control =control ,color = True,
                num_iterations = n_iter, learning_rate = lr)[:,:,::-1]
            scipy.misc.imsave(out_file, out_im)

    visualize.writeHTMLForFolder(out_dir_im)






def main():

    run_for_caps_exp()
    # script_explore_lr_steps()

    return
    in_file = os.path.join('../scratch','rand_im_224.jpg')
        # 'deep_dream_new_code','rand_im_224.jpg')
    # scratch/deep_dream_new_code/rand_im_224.jpg

#     background_color = np.float32([200.0, 200.0, 200.0])
# # generate initial random image
#     input_img = np.random.normal(background_color, 8, (224, 224, 3))
#     scipy.misc.imsave(in_file,input_img)


#     return
    model_name = 'vgg_capsule_7_33/bp4d_256_train_test_files_256_color_align_0_reconstruct_True_True_all_aug_marginmulti_False_wdecay_0_1_exp_0.96_350_1e-06_0.0001_0.001_0.001_lossweights_1.0_0.1_True'
    model_file_name = 'model_0.pt'


    out_dir = os.path.join('../experiments/visualizing',model_name)
    util.makedirs(out_dir)
    

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
    
    out_dir_im = os.path.join(out_dir,'au_color_gauss_5e-1_200')
    util.mkdir(out_dir_im)

    for control in range(n_classes):
        au = au_list[control]
        out_file = os.path.join(out_dir_im, str(au))

        out_im = dreamer.dream_fc_caps(model,in_file, octave_n = 2, control =control ,color = True,num_iterations = 200, learning_rate = 5e-1)[:,:,::-1]
        scipy.misc.imsave(out_file+'.jpg', out_im)

    visualize.writeHTMLForFolder(out_dir_im)

    # 

    # params reasonable
    # octave_n = 2
    # sigma = np.linspace(0.6,0.4,octave_n)
    # learning_rate = 5e-2
    # max_jitter = 14
    # num_iterations = 80


if __name__=='__main__':
    main()
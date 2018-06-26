import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
from torchvision import transforms, utils
import numpy as np
from helpers import util, visualize
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops
import os
import torchvision
from torchvision.models.vgg import model_urls
import scipy.misc


def mkdir(dir_curr):
    if not os.path.exists(dir_curr):
        os.mkdir(dir_curr);

def load_image(path):
    image = Image.open(path)
    # plt.imshow(image)
    # plt.title("Image loaded successfully")
    return image

def deprocess(image):
    return image * torch.Tensor([0.229, 0.224, 0.225]).cuda()  + torch.Tensor([0.485, 0.456, 0.406]).cuda()

def dd_helper(image, layer, iterations, lr):        
    
    input = Variable(preprocess(image).unsqueeze(0).cuda(), requires_grad=True)
    loss_arr = torch.zeros(1,1000)
    loss_arr[0,281]=1
    loss = Variable(loss_arr.cuda())
    print loss_arr

    vgg.zero_grad()
    for i in range(iterations):
#         print('Iteration: ', i)
        out = input
        out = vgg(out)
        
        
        # print loss.size()
        # print loss[0,281]
        # print loss[0,:10]
        out.backward(loss)
        input.data = input.data + lr * input.grad.data
    
    input = input.data.squeeze()
    input.transpose_(0,1)
    input.transpose_(1,2)
    input = np.clip(deprocess(input), 0, 1)
    im = Image.fromarray(np.uint8(input*255))
    return im

def deep_dream_vgg(image, layer, iterations, lr, octave_scale, num_octaves,out_file):
    # print image.size
    if num_octaves>0:
        image1 = image.filter(ImageFilter.GaussianBlur(2))
        if(image1.size[0]/octave_scale < 1 or image1.size[1]/octave_scale<1):
            size = image1.size
        else:
            size = (int(image1.size[0]/octave_scale), int(image1.size[1]/octave_scale))
            
        image1 = image1.resize(size,Image.ANTIALIAS)
        image1 = deep_dream_vgg(image1, layer, iterations, lr, octave_scale, num_octaves-1,out_file)
        size = (image.size[0], image.size[1])
        image1 = image1.resize(size,Image.ANTIALIAS)
        image = ImageChops.blend(image, image1, 0.6)
    # print("-------------- Recursive level: ", num_octaves, '--------------')
    img_result = dd_helper(image, layer, iterations, lr)
    img_result = img_result.resize(image.size)
    # img_result = Image.fromarray(img_result, mode='RGB')
    
    if num_octaves == 20:
        img_result = np.array(img_result)
        scipy.misc.imsave(out_file,img_result)
        
    return img_result


def deep_dream_cat(image, iterations, lr, out_file, cat = 281):
    image = Image.fromarray(np.uint8(np.random.randint(256, size = (224,224,3))))

    # input = Variable(torch.zeros(1,3,224,224).cuda(), requires_grad=True)
    
    # print torch.min(input),torch.max(input)
    # print preprocess(image).unsqueeze(0).size()
    vgg.zero_grad()
    start_sigma = 0.78*1.2
    end_sigma = 0.4
    sigma_arr = np.linspace(start_sigma, end_sigma, iterations)
    for i in range(iterations):
        image = image.filter(ImageFilter.GaussianBlur(sigma_arr[i]))
        input = Variable(preprocess(image).unsqueeze(0).cuda(), requires_grad=True)
#         print('Iteration: ', i)
        # out = input
        # for j in range(layer):
        out = vgg(input)
        # print out.size()
        # raw_input()
        # print 'out.size()',out.size()
        loss = out[0,cat]
        # print 'loss.size()',loss.size()
        loss.backward()
        input.data = input.data + lr * input.grad.data

        image = input.data.squeeze()
        image.transpose_(0,1)
        image.transpose_(1,2)
        image = np.clip(deprocess(image), 0, 1)
        image = Image.fromarray(np.uint8(image*255))
        scipy.misc.imsave(out_file[:out_file.rindex('.')]+'_'+str(i)+'.jpg',np. array(image))    

    
    input = input.data.squeeze()
    input.transpose_(0,1)
    input.transpose_(1,2)
    input = np.clip(deprocess(input), 0, 1)
    im = Image.fromarray(np.uint8(input*255))
    im = np.array(im)
    scipy.misc.imsave(out_file,im)
    # return im
    util.writeHTMLForFolder(os.path.split(out_file)[0])


out_dir = '../scratch/deep_dream_on_nb'
# if not os.path.exists(out_dir):
mkdir(out_dir)

normalise = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
    )

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalise
    ])

model_urls['vgg16'] = model_urls['vgg16'].replace('https://', 'http://')
vgg = torchvision.models.vgg16(pretrained=True)

# vgg = models.vgg16(pretrained=True)
vgg = vgg.cuda()
print(vgg)
modulelist = list(vgg.features.modules())

out_dir = '../scratch/rand'
# visualize.writeHTMLForFolder(out_dir)


# in_file = os.path.join(out_dir,'sky_eye.jpg')
in_file = os.path.join(out_dir,'rand.jpg')

# rand_noise_im = np.uint8(np.random.randint(256,size = (224,224,3)))
# scipy.misc.imsave(in_file, rand_noise_im)

# sky-dd.jpeg'
sky = load_image(in_file)
# print sky.shape
num_iter = 10
lr = 0.05
cat = 281

# deep_dream_vgg(image, layer, iterations, lr, octave_scale, num_octaves,out_file)
for num_iter in range(25,50,5):
    out_file = os.path.join(out_dir,'_'.join([str(val) for val in [num_iter, lr, cat]])+'.jpg')
    deep_dream_vgg(sky, 28, num_iter, 0.3, 2, 20, out_file)
    print 'done with ',out_file

visualize.writeHTMLForFolder(out_dir)

    # os.path.join(out_dir,'sky_5.jpg'))
# deep_dream_cat(sky, num_iter, lr, out_file, cat = cat)

# # sigma = 2.5
# # out_file = os.path.join(out_dir,'blurred_'+str(sigma)+'.jpg')
# # blur_and_save(sky,out_file,sigma)


# print 'DONEEEEE'

# deep_dream_cat(sky, 20, 0.3, os.path.join(out_dir,'sky_as_cat_20.jpg'))
# deep_dream_cat(sky, 30, 0.3, os.path.join(out_dir,'sky_as_cat_30.jpg'))
# deep_dream_cat(sky, 5, 0.3, os.path.join(out_dir,'sky_as_cat_5.jpg'))


# sky_5 = deep_dream_vgg(sky, 5, 5, 0.3, 2, 20, os.path.join(out_dir,'sky_5.jpg'))
# sky_7 = deep_dream_vgg(sky, 7, 4, 0.3, 2, 20, os.path.join(out_dir,'sky_7.jpg'))
# sky_10 = deep_dream_vgg(sky, 10, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_10.jpg'))
# sky_12 = deep_dream_vgg(sky, 12, 2, 0.3, 2, 20, os.path.join(out_dir,'sky_10.jpg'))
# sky_14 = deep_dream_vgg(sky, 14, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_14.jpg'))
# sky_17 = deep_dream_vgg(sky, 17, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_17.jpg'))
# sky_19 = deep_dream_vgg(sky, 19, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_19.jpg'))
# sky_21 = deep_dream_vgg(sky, 21, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_21.jpg'))
# sky_24 = deep_dream_vgg(sky, 24, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_24.jpg'))
# sky_26 = deep_dream_vgg(sky, 26, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_26.jpg'))
# sky_28 = deep_dream_vgg(sky, 28, 3, 0.3, 2, 20, os.path.join(out_dir,'sky_28.jpg'))




import numpy as np
import torch
from util import showtensor
import scipy.ndimage as nd
from torch.autograd import Variable
import scipy.misc

def objective_L2(dst, guide_features):
    grad = dst.data.clone()
    # print guide_features
    grad[0,:guide_features]=0
    grad[0,guide_features+1:]=0
    return grad


def make_step(img, model, control=None, distance=objective_L2):
    mean = np.array([0.485, 0.456, 0.406]).reshape([3, 1, 1])
    std = np.array([0.229, 0.224, 0.225]).reshape([3, 1, 1])

    learning_rate = 5e-2
    max_jitter = 0
    num_iterations = 100
    show_every = 10
    end_layer = 5
    guide_features = control

    for i in range(num_iterations):
        shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
        img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
        # apply jitter shift
        model.zero_grad()
        img_tensor = torch.Tensor(img)
        if torch.cuda.is_available():
            img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        else:
            img_variable = Variable(img_tensor, requires_grad=True)

        # print img_variable.size()
        act_value = model.forward(img_variable, end_layer)
        # print act_value.size()
        # raw_input()
        diff_out = distance(act_value, guide_features)
        act_value.backward(diff_out)
        ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
        learning_rate_use = learning_rate / ratio
        img_variable.data.add_(img_variable.grad.data * learning_rate_use)
        img = img_variable.data.cpu().numpy()  # b, c, h, w
        img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
        img[0, :, :, :] = np.clip(img[0, :, :, :], -mean / std,
                                  (1 - mean) / std)
        # if i == 0 or (i + 1) % show_every == 0:
        #     showtensor(img)


        # src.data[:] += step_size/np.abs(g).mean() * g
    return img

def dream_fc(model,
          base_img,
          octave_n=1,
          octave_scale=1.4,
          control=281,
          distance=objective_L2,
          out_file = None):
    # octaves = [base_img]
    sigma = np.linspace(2,0.4,6)
    # print sigma
    # print type(base_img)
    # print base_img.shape
    
    # torch.nn.functional.conv2d(base_img, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
    # base_img_t = base_img.squeeze().transpose(1,2,0)
    # print base_img_t.shape
    # raw_input()
    octaves = []
    for i in range(octave_n ):
        # img_curr = nd.filters.gaussian_filter(base_img_t,sigma[i])
        # img_curr = img_curr.transpose(2,0,1)[np.newaxis,:,:,:]
        # octaves.append(img_curr)
        octaves.append(base_img)

        # img_curr = showtensor(img_curr)
        # out_file_curr = out_file[:out_file.rindex('.')]+'_'+str(sigma[i])+'.jpg'
        # scipy.misc.imsave(out_file_curr,img_curr)
        # print out_file_curr
        # octaves.append(Image.fromabase_img)
            # nd.zoom(
            #     octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),
            #     order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        # if octave > 0:
        #     h1, w1 = detail.shape[-2:]
        #     detail = nd.zoom(
        #         detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail

        # input_oct = input_oct.squeeze().transpose(1,2,0)
        # input_oct = nd.filters.gaussian_filter(input_oct,sigma[octave])
        # input_oct = input_oct.transpose(2,0,1)[np.newaxis,:,:,:]
        

        print(input_oct.shape)
        out = make_step(input_oct, model, control, distance=distance)
        detail = out - octave_base
    return showtensor(out)

def dream(model,
          base_img,
          octave_n=6,
          octave_scale=1.4,
          control=None,
          distance=objective_L2):
    octaves = [base_img]
    for i in range(octave_n - 1):
        octaves.append(nd.zoom(
                octaves[-1], (1, 1, 1.0 / octave_scale, 1.0 / octave_scale),
                order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(
                detail, (1, 1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        input_oct = octave_base + detail
        print(input_oct.shape)
        out = make_step(input_oct, model, control, distance=distance)
        detail = out - octave_base
    return showtensor(out)

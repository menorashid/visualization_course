import numpy as np
import torch
# from util import showtensor
import scipy.ndimage as nd
from torch.autograd import Variable
import scipy.misc



# import PIL.Image
from PIL import Image
from io import BytesIO
import numpy as np

class Deep_Dream():
    def __init__(self,mean_file,std_file,color = False,list_regularizers = 'all'):
        if mean_file is 'vgg':
            self.mean = np.array([93.5940,104.7624,129.1863])
            self.std = np.array([1.,1.,1.])
            self.mean = self.mean[np.newaxis,np.newaxis,:]
            self.std = self.std[np.newaxis,np.newaxis,:]

        else:
            self.mean = scipy.misc.imread(mean_file).astype(np.float32)
            self.std = scipy.misc.imread(std_file).astype(np.float32)
        self.color = color
        self.list_regularizers = list_regularizers



    def showtensor(self,a):
        mean = self.mean
        # np.array([0.485, 0.456, 0.406]).reshape([1, 1, 3])
        std = self.std
        # np.array([0.229, 0.224, 0.225]).reshape([1, 1, 3])
        if a.shape[1]==1:
            inp = a[0, 0, :, :]
        else:
            inp = a[0,:,:,:].transpose(1,2,0)
        # inp = inp.transpose(1, 2, 0)
        inp = std * inp + mean
        # print np.min(inp[:,:,0]), np.max(inp[:,:,0])
        # print np.min(inp[:,:,1]), np.max(inp[:,:,1])
        # print np.min(inp[:,:,2]), np.max(inp[:,:,2])
        # inp *= 255
        # inp = np.uint8(np.clip(inp, 0, 255))
        return inp



    def objective_L2(dst, guide_features):
        grad = dst.data.clone()
        # print guide_features
        # print grad.size()
        if guide_features>0:
            grad[0,:guide_features]=0

        if guide_features < grad.size(1)-1:
            grad[0,guide_features+1:]=0
        # print grad
        # raw_input()
        return grad

    def objective_L2_caps_direction(self,dst, caps, dir_idx,guide_features):
        grad = dst.data.clone()
        # # print guide_features
        # # print grad.size()
        if guide_features>0:
            grad[0,:guide_features]=0

        if guide_features < grad.size(1)-1:
            grad[0,guide_features+1:]=0
        grad[0,guide_features]=1
        # print grad
        # raw_input()
        grad_caps = caps.data.clone()
        grad_caps.fill_(0)
        grad_caps[0,guide_features,dir_idx] = 1
        # print grad_caps
        return grad,grad_caps


    def primary_objective(self,dst, guide_features):
        # print guide_features
        # print filt_vals.shape
        # print filt_vals
        # print np.sum(filt_vals==0)
        # print np.argmax(filt_vals)
        # raw_input()
        grad = dst.data.clone()
        
        grad.fill_(0)
        # print grad.size()
        # print guide_features
        grad[0,guide_features[0],guide_features[1],guide_features[2]] = 1
        # print grad

        return grad

    
    def make_step(self, img, model, control=None, distance=objective_L2,sigma=None,caps = False,color = False,num_iterations = 80, learning_rate = 5e-2, return_caps = False, max_jitter = 14):
        mean = self.mean
        std = self.std
        # learning_rate = 5e-2
        # max_jitter = 5
        # num_iterations = 80
        show_every = 10
        end_layer = 5
        iter_mul = 1
        guide_features = control
        print 'sigma',sigma
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
            if caps and not return_caps:
                act_value = model.forward_for_viz(img_variable)
                diff_out = distance(act_value, guide_features)
                act_value.backward(diff_out)
            elif caps and return_caps:
                act_value, caps_vecs = model.forward_for_viz(img_variable,return_caps = return_caps)
                print act_value.size()
                print caps_vecs.size()
                # raw_input()
                # diff_out = distance(act_value, guide_features)
                diff_out,grad_caps = self.objective_L2_caps_direction(act_value, caps_vecs, 0,guide_features)
                # diff_out.backward()
                act_value.backward(diff_out,retain_graph=True)
                caps_vecs.backward(grad_caps)
                # raw_input()
            else:
                act_value = model.forward(img_variable)
                diff_out = distance(act_value, guide_features)
                act_value.backward(diff_out)

            ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
            learning_rate_use = learning_rate / ratio
            img_variable.data.add_(img_variable.grad.data * learning_rate_use)
            img = img_variable.data.cpu().numpy()  # b, c, h, w
            img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
            # print np.min(img),np.max(img),
            min_clip = -mean/std
            max_clip = (255-mean)/std
            # print np.min(min_clip), np.max(min_clip)
            # print np.min(max_clip), np.max(max_clip)
            if color:
                min_clip = min_clip.transpose(2,1,0)
                max_clip = max_clip.transpose(2,1,0)
                img[0, :, :, :] = np.clip(img[0 ,:, :, :], min_clip,
                                          max_clip)

            if sigma is not None and num_iterations%iter_mul==0:
                if not color:
                    img = img.squeeze()
                else:
                    # print img.shape
                    img = img.squeeze().transpose(1,2,0)
                    # print img.shape
                    


                
                if not color:
                    img = nd.filters.gaussian_filter(img,sigma)
                    img = img[np.newaxis,np.newaxis,:,:]
                else:
                    # print img.shape
                    img[:,:,0] = nd.filters.gaussian_filter(img[:,:,0],sigma)
                    img[:,:,1] = nd.filters.gaussian_filter(img[:,:,1],sigma)
                    img[:,:,2] = nd.filters.gaussian_filter(img[:,:,2],sigma)
                    img = img.transpose(2,0,1)[np.newaxis,:,:,:]
                    # print img.shape
                # raw_input()

            # if i == 0 or (i + 1) % show_every == 0:
            #     showtensor(img)


            # src.data[:] += step_size/np.abs(g).mean() * g
        return img

    def make_step_primary(self, img, model, control=None, distance=objective_L2,sigma=None,caps = False,color = False,num_iterations = 80, learning_rate = 5e-2, return_caps = False, max_jitter = 14):
        
        mean = self.mean
        std = self.std
        # learning_rate = 5e-2
        # max_jitter = 5
        # num_iterations = 80
        show_every = 10
        end_layer = 5
        iter_mul = 1
        guide_features = control
        print 'sigma',sigma
        for i in range(num_iterations):
            # print guide_features
            
            shift_x, shift_y = np.random.randint(-max_jitter, max_jitter + 1, 2)
            img = np.roll(np.roll(img, shift_x, -1), shift_y, -2)
            
            model.zero_grad()
            img_tensor = torch.Tensor(img)
            if torch.cuda.is_available():
                img_variable = Variable(img_tensor.cuda(), requires_grad=True)
            else:
                img_variable = Variable(img_tensor, requires_grad=True)

            # print img_variable.size()
            act_value = model.forward_for_viz_primary(img_variable)
            if guide_features[0] is None and i==0:
                filt_vals = act_value.data.cpu().numpy()[0,:,guide_features[1],guide_features[2]]
                if np.max(filt_vals)==0:
                    img = None
                    break

                # print filt_vals.shape, np.min(filt_vals),np.max(filt_vals),np.argmax(filt_vals),np.max(filt_vals)
                guide_features[0]=np.argmax(filt_vals)

            
            diff_out = self.primary_objective(act_value, guide_features)
            act_value.backward(diff_out)

            
            ratio = np.abs(img_variable.grad.data.cpu().numpy()).mean()
            # print ratio
            ratio = max(ratio,1e-9)
            learning_rate_use = learning_rate / ratio

            img_variable.data.add_(img_variable.grad.data * learning_rate_use)
            img = img_variable.data.cpu().numpy()  # b, c, h, w
            img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
            min_clip = -mean/std
            max_clip = (255-mean)/std
            if color:
                min_clip = min_clip.transpose(2,1,0)
                max_clip = max_clip.transpose(2,1,0)
                img[0, :, :, :] = np.clip(img[0 ,:, :, :], min_clip,
                                          max_clip)

            if sigma is not None and num_iterations%iter_mul==0:
                if not color:
                    img = img.squeeze()
                else:
                    # print img.shape
                    img = img.squeeze().transpose(1,2,0)
                    # print img.shape
                    


                
                if not color:
                    img = nd.filters.gaussian_filter(img,sigma)
                    img = img[np.newaxis,np.newaxis,:,:]
                else:
                    # print img.shape
                    img[:,:,0] = nd.filters.gaussian_filter(img[:,:,0],sigma)
                    img[:,:,1] = nd.filters.gaussian_filter(img[:,:,1],sigma)
                    img[:,:,2] = nd.filters.gaussian_filter(img[:,:,2],sigma)
                    img = img.transpose(2,0,1)[np.newaxis,:,:,:]
                    
        return img,guide_features


    
    def dream_primary_caps(self,model,
              in_file,
              octave_n=3,
              octave_scale=1.4,
              control=7,
              distance=objective_L2,
              out_file = None,
              num_iterations = 80,
              learning_rate = 5e-2,
              color= False,
              sigma = np.linspace(0.6,0.4,2),
              return_caps = False, max_jitter = 14):
        
        base_img = scipy.misc.imread(in_file)
        if base_img.shape[0]>224:
            base_img = scipy.misc.imresize(base_img,(224,224))[:,:,::-1]

        base_img = base_img.astype(np.float32)
        # print base_img.shape
        # print self.mean.shape
        # raw_input()
        base_img = (base_img - self.mean)/self.std
        

        if not color:
            base_img = base_img[np.newaxis,np.newaxis,:,:]
        else:
            base_img = base_img[np.newaxis,:,:,:]
            print base_img.shape
            base_img = base_img.transpose(0,3,1,2)
        # base_img = np.concatenate([base_img,base_img],0)
        print base_img.shape, np.min(base_img),np.max(base_img)

        # sigma = np.linspace(0.6,0.4,octave_n)
        # [None]*octave_n
        # np.linspace(0.6,0.4,octave_n)

        # img_tensor = torch.Tensor(base_img)        
        # img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        # act_value = model.forward_for_viz(img_variable)
        # print act_value.size()
        # print act_value
        # ,act_value[1].size()

        octaves = []
        for i in range(octave_n ):
            octaves.append(base_img)

        detail = np.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            input_oct = octave_base + detail
            print octave, sigma
            out,control = self.make_step_primary(input_oct, model, control, distance=distance,sigma=sigma[octave], caps = True,color = color,num_iterations = num_iterations,learning_rate = learning_rate, return_caps = return_caps, max_jitter = max_jitter)
            
            if out is None:
                return None, None

            detail = out - octave_base
        return self.showtensor(out),control


    def dream_fc_caps(self,model,
              in_file,
              octave_n=3,
              octave_scale=1.4,
              control=7,
              distance=objective_L2,
              out_file = None,
              num_iterations = 80,
              learning_rate = 5e-2,
              color= False,
              sigma = np.linspace(0.6,0.4,2),
              return_caps = False, max_jitter = 14):
        
        base_img = scipy.misc.imread(in_file)
        if base_img.shape[0]>224:
            base_img = scipy.misc.imresize(base_img,(224,224))[:,:,::-1]

        base_img = base_img.astype(np.float32)
        # print base_img.shape
        # print self.mean.shape
        # raw_input()
        base_img = (base_img - self.mean)/self.std
        

        if not color:
            base_img = base_img[np.newaxis,np.newaxis,:,:]
        else:
            base_img = base_img[np.newaxis,:,:,:]
            print base_img.shape
            base_img = base_img.transpose(0,3,1,2)
        # base_img = np.concatenate([base_img,base_img],0)
        print base_img.shape, np.min(base_img),np.max(base_img)

        # sigma = np.linspace(0.6,0.4,octave_n)
        # [None]*octave_n
        # np.linspace(0.6,0.4,octave_n)

        # img_tensor = torch.Tensor(base_img)        
        # img_variable = Variable(img_tensor.cuda(), requires_grad=True)
        # act_value = model.forward_for_viz(img_variable)
        # print act_value.size()
        # print act_value
        # ,act_value[1].size()

        octaves = []
        for i in range(octave_n ):
            octaves.append(base_img)

        detail = np.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            input_oct = octave_base + detail
            print octave, sigma
            out = self.make_step(input_oct, model, control, distance=distance,sigma=sigma[octave], caps = True,color = color,num_iterations = num_iterations,learning_rate = learning_rate, return_caps = return_caps, max_jitter = max_jitter)
            detail = out - octave_base
        return self.showtensor(out)


    def dream_fc(self,model,
              in_file,
              octave_n=3,
              octave_scale=1.4,
              control=7,
              distance=objective_L2,
              out_file = None,
              sigma = np.linspace(0.6,0.4,2)):
        # octaves = [base_img]

        # transformer = transforms.Compose([
        #         transforms.ToTensor(),
        #         lambda x: x*255.
        #     ])

        # base_img = Image.open(in_file)
        # base_img = transformer(base_img).unsqueeze(0)
        base_img = scipy.misc.imread(in_file).astype(np.float32)
        base_img = (base_img - self.mean)/self.std
        
        base_img = base_img[np.newaxis,np.newaxis,:,:]
        

        print base_img.shape, np.min(base_img),np.max(base_img)

        # sigma = np.linspace(0.6,0.4,octave_n)
        # sigma = [None]*octave_n
        # np.linspace(0.6,0.4,octave_n)
        # [None]*octave_n
        # np.linspace(2,0.1,6)

        octaves = []
        for i in range(octave_n ):
            octaves.append(base_img)

        detail = np.zeros_like(octaves[-1])
        for octave, octave_base in enumerate(octaves[::-1]):
            h, w = octave_base.shape[-2:]
            input_oct = octave_base + detail
            print octave, sigma
            out = self.make_step(input_oct, model, control, distance=distance,sigma=sigma[octave])
            detail = out - octave_base
        return self.showtensor(out)

    def dream(self,model,
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
            out = self.make_step(input_oct, model, control, distance=distance)
            detail = out - octave_base
        return showtensor(out)

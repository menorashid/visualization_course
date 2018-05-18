import torch.nn as nn

import numpy as np
import scipy.misc
import torch
import math
from capsules import Primary_Caps,Conv_Caps

class Matrix_Capsule_Model(nn.Module):

    def __init__(self,n_classes,conv_layers,caps_layers,r, coord_add, pose_mat_size=4):
        super(Matrix_Capsule_Model, self).__init__()
        out_size = 3

        # coord_add = torch.autograd.Variable(torch.Tensor(np.array([[[8., 8.], [12., 8.], [16., 8.]],
        #                   [[8., 12.], [12., 12.], [16., 12.]],
        #                   [[8., 16.], [12., 16.], [16., 16.]]]))

        self.pose_mat_size = pose_mat_size
        self.features = []
        for conv_param in conv_layers:
            self.features.append(nn.Conv2d(in_channels=1, out_channels=conv_param[0],
                                   kernel_size=conv_param[1], stride=conv_param[2]))
            self.features.append(nn.ReLU(True))

        
        for idx_caps_param,caps_param in enumerate(caps_layers):
            if idx_caps_param==0:
                self.features.append(Primary_Caps(conv_layers[-1][0], caps_layers[0][0]))
            else:
                self.features.append( Conv_Caps(caps_layers[idx_caps_param-1][0], caps_param[0], r, caps_param[1], caps_param[2], self.pose_mat_size))
        self.features.append( Conv_Caps(caps_layers[-1][0], n_classes, r, 1, 1, self.pose_mat_size, class_it=True, coord_add=coord_add))

        self.features = nn.Sequential(*self.features)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.primary_caps(x)
        # x = self.conv_caps1(x)
        # x = self.conv_caps2(x)
        # x = self.class_caps(x)
        
        temp = self.pose_mat_size*self.pose_mat_size+1
        output = x.view(x.size(0),x.size(1)/temp,temp)
        activation = output[:,:,-1]
        
        return activation

    def spread_loss(self,x,target,m):
        use_cuda = next(self.parameters()).is_cuda

        b = x.size(0)
        target_t = target.type(torch.LongTensor)
        
        if use_cuda:
            target_t = target_t.cuda()
        
        rows = torch.LongTensor(np.array(range(b)))
        
        if use_cuda:
            rows = rows.cuda()

        a_t = x[rows,target_t]
        a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10
        u = m-(a_t_stack-x) #b,10
        u = nn.functional.relu(u)**2
        u[rows,target_t]=0
        loss = torch.sum(u)/b
        
        return loss

class Network:
    def __init__(self,n_classes=8,r=3,input_size=96):
        conv_layers = [[32,5,2]]
        caps_layers = [[8,1,1],[16,3,2],[16,3,2],[16,3,2]]

        layers_all = conv_layers+caps_layers
        currentLayer = [input_size,1,1,0.5]
        for i in range(len(layers_all)):
            currentLayer = self.outFromIn(layers_all[i][1:], currentLayer)
            # print currentLayer
        start_in = math.floor(currentLayer[-1])
        jump = currentLayer[1]
        out_size = int(currentLayer[0])
        coord_add = np.zeros((out_size,out_size,2))
        for i in range(coord_add.shape[0]):
            for j in range(coord_add.shape[1]):
                coord_add[i,j,1] = start_in+i*jump
                coord_add[i,j,0] = start_in+j*jump

        print coord_add.shape
        coord_add = torch.autograd.Variable(torch.Tensor(coord_add))

        model = Matrix_Capsule_Model(n_classes,conv_layers,caps_layers,r,coord_add)

        
        for idx_m,m in enumerate(model.modules()):
            if isinstance(m, nn.Conv2d) or isinstance(m,nn.Linear):
                nn.init.xavier_normal(m.weight.data)
                nn.init.constant(m.bias.data,0.)
            elif isinstance(m, nn.BatchNorm2d):
                
                nn.init.constant(m.weight.data,1.)
                nn.init.constant(m.bias.data,0.)
                
        self.model = model
        
    def outFromIn(self, conv, layerIn):
        n_in = layerIn[0]
        j_in = layerIn[1]
        r_in = layerIn[2]
        start_in = layerIn[3]
        k = conv[0]
        s = conv[1]
        p = 0

        n_out = math.floor((n_in - k + 2*p)/s) + 1
        actualP = (n_out-1)*s - n_in + k 
        pR = max(0,math.ceil(actualP/2))
        pL = max(0,math.floor(actualP/2))

        j_out = j_in * s
        r_out = r_in + (k - 1)*j_in
        start_out = start_in + ((k-1)/2 - pL)*j_in
        return n_out, j_out, r_out, start_out
    # def get_lr_list(self, lr):
    #     lr_list= [{'params': self.model.features.parameters(), 'lr': lr[0]}]\
    #             +[{'params': self.model.classifier.parameters(), 'lr': lr[1]}]
    #     return lr_list



    


def main():
    import numpy as np
    import torch
    from torch.autograd import Variable

    options = {'mnist': ([[[8., 8.], [12., 8.], [16., 8.]],
                          [[8., 12.], [12., 12.], [16., 12.]],
                          [[8., 16.], [12., 16.], [16., 16.]]], 28.),
               'smallNORB': ([[[8., 8.], [12., 8.], [16., 8.], [24., 8.]],
                              [[8., 12.], [12., 12.], [16., 12.], [24., 12.]],
                              [[8., 16.], [12., 16.], [16., 16.], [24., 16.]],
                              [[8., 24.], [12., 24.], [16., 24.], [24., 24.]]], 32.)
               }
    mnist_coord_add = np.array(options['mnist'][0])
    print mnist_coord_add.shape

    smallNORB_coord_add = np.array(options['smallNORB'][0])
    print smallNORB_coord_add.shape
    
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(2)

    # print np.array(options['mnist'][0])

    

    net = Network()
    net.model = net.model.cuda()
    print net.model
    input = np.zeros((10,1,96,96))
    labels = np.array(range(8)+[0,1],dtype=np.int)
    input = torch.Tensor(input).cuda()
    print input.shape
    input = Variable(input)
    labels = Variable(torch.Tensor(labels).cuda())
    print labels.size()
    output = net.model(input)
    # print output
    loss = net.model.spread_loss(output,labels,0.2)
    print loss

    # loss.backward()

    # print output.size()




    

if __name__=='__main__':
    main()
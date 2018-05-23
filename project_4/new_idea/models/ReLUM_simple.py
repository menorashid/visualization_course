from torchvision import models
import torch.nn as nn
import torch

class ReLUM(nn.Module):

    def __init__(self,num_in,num_out,conv_kernel,conv_stride,conv_pad):
        super(ReLUM, self).__init__()
        self.num_out = num_out
        self.num_in = num_in
        self.thresh = nn.Parameter(torch.zeros(self.num_out,1,self.num_in,1,1))
        nn.init.xavier_normal(self.thresh.data)
        # self.pool = nn.MaxPool2d(pool_kernel,pool_stride)
        self.conv = nn.ModuleList([nn.Conv2d(num_in, 1, conv_kernel, stride = conv_stride, padding = conv_pad) for idx in range(num_out)])

        
    def forward(self, x):
        # print self.thresh
        # idx_out = 1
        # print self.thresh.size()
        # print self.thresh[idx_out].size()
        # out = x+ self.thresh[idx_out]
        # .view(1,self.num_in,1,1)

        # out = []
        # for idx_out in range(self.num_out):

        out = torch.cat([self.conv[idx_out](nn.functional.relu(x+ self.thresh[idx_out], inplace=False)) for idx_out in range(self.num_out)], dim=1)
        #     out.append(self.conv[idx_out](out_curr))

        # out = torch.cat(out,1)
        # print out.size()
        # raw_input()
            # print out_curr.size()                
            # print out[idx_out].size()
            #     # print in_rel.size()
            #     # print out_curr.size()
            # raw_input()


            # out.append(self.conv[idx_out].(self.pool(nn.functional.threshold(x[:,idx,:,:], self.thresh.data[idx,idx_out], 0, inplace=True)) for idx in range(self.num_in)            

        # x = [[self.pool(nn.functional.threshold(x, self.thresh.data[idx,idx_out], 0, inplace=True)) for idx in range(self.num_in)] for idx_out in range(self.num_out)]
        # x = [self.conv[idx](x_curr) for idx,x_curr in enumerate(x)]

        return out

def main():
    from torch.autograd import Variable

    r = ReLUM(16,32,5,1,2)
    input = Variable(-1*torch.ones(10,16,32,32))
    out = r.forward(input)
    print out.size()
    # print len(out[0])
    # print out[0][0].size()


if __name__=='__main__':
    main()
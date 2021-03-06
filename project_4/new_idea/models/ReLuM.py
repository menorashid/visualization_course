from torchvision import models
import torch.nn as nn
import torch

class ReLUM(nn.Module):

    def __init__(self,num_in,num_out,pool_kernel,pool_stride,conv_kernel,conv_stride,conv_pad):
        super(ReLUM, self).__init__()
        self.num_out = num_out
        self.num_in = num_in
        self.thresh = nn.Parameter(torch.zeros(self.num_in,self.num_out))
        self.pool = nn.MaxPool2d(pool_kernel,pool_stride)
        self.conv = nn.ModuleList([nn.Conv2d(num_in, 1, conv_kernel, stride = conv_stride, padding = conv_pad) for idx in range(num_out)])

        
    def forward(self, x):
        # print self.thresh
        out = []
        for idx_out in range(self.num_out):

            out_curr = torch.cat([self.pool(nn.functional.threshold(x[:,idx_in:idx_in+1,:,:], self.thresh.data[idx_in,idx_out], 0, inplace=False)) for idx_in in range(self.num_in)], dim=1)
            out.append(self.conv[idx_out](out_curr))

        out = torch.cat(out,1)
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

    r = ReLUM(16,32,3,2,5,1,2)
    input = Variable(-1*torch.ones(10,16,32,32))
    out = r.forward(input)
    print out.size()
    # print len(out[0])
    # print out[0][0].size()


if __name__=='__main__':
    main()
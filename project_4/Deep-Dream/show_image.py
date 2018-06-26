import os
import torch
from torchvision import transforms
from PIL import Image
from resnet import resnet50
from deepdream import dream, dream_fc
import scipy.misc
import numpy as np
def main():
	out_dir = '../scratch/deep_dream_new_code'
	if not os.path.exists(out_dir):
		os.mkdir(out_dir)

	img_transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	# input_img = scipy.misc.imread('./sky.jpg')
	in_file = os.path.join(out_dir,'rand_im_224.jpg')
	cat = 281
	out_file = in_file[:in_file.rindex('.')]+'_'+str(cat)+'_fc_all.jpg'
	# os.path.join(out_dir,'try.jpg')
	# input_img = Image.fromarray(np.uint8(np.random.randint(256,size = (224,224,3))))

	# scipy.misc.imsave(in_file,input_img)

	# return
	input_img = Image.open(in_file)
	input_tensor = img_transform(input_img).unsqueeze(0)
	input_np = input_tensor.numpy()

	model = resnet50(pretrained=True)
	# print model.fc
	# return
	# print model


	if torch.cuda.is_available():
	    model = model.cuda()
	for param in model.parameters():
	    param.requires_grad = False

	output_im = dream_fc(model, input_np,control = cat)
	print output_im.shape
	
	scipy.misc.imsave(out_file,output_im)


if __name__=='__main__':
	main()
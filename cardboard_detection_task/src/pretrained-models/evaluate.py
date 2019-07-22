import torch
import pretrainedmodels
import pretrainedmodels.utils as utils

# inceptionresnetv2
model_name = 'inceptionresnetv2'
model = pretrainedmodels.__dict__[model_name](num_classes=1001, pretrained='imagenet+background')

# # nasnetalarge
# model_name = 'nasnetalarge'
# model = pretrainedmodels.__dict__[model_name](num_classes=1001, pretrained='imagenet+background')



## Predict on single image

# switch to eval mode?
model.eval()

load_img = utils.LoadImage()

# transformations depending on the model
# rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
tf_img = utils.TransformImage(model) 

path_img = 'data/cat.jpg'

input_img = load_img(path_img)
input_tensor = tf_img(input_img)         # 3x400x225 -> 3x299x299 size may differ
input_tensor = input_tensor.unsqueeze(0) # 3x299x299 -> 1x3x299x299
input = torch.autograd.Variable(input_tensor,
    requires_grad=False)


output_features = model.features(input) # 1x14x14x2048 size may differ
output_logits = model.logits(output_features) # 1x1000

print (output_features.shape)
print (output_logits[0].shape)
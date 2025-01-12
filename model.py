import torchvision.transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import copy
from PIL import Image

def gram_matrix(input):
    batch_size, h, w, f_map_num = input.size()

    features = input.view(batch_size * h, w * f_map_num)

    G = torch.mm(features, features.t())
    return G.div(batch_size * h * w * f_map_num)

class ContentLoss(nn.Module):

    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = F.mse_loss(self.target, self.target)

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class StyleTransferModel:
    def __init__(self, imsize=256, model_name='vgg_19'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if model_name == 'vgg_19':
            self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
            self.content_layers = ['conv_4']
            self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5', 'conv_6', 'conv_7']
        elif model_name == 'alexnet':
            self.cnn = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).to(self.device).eval()
            self.content_layers = ['conv_4']
            self.style_layers = ['conv_3', 'conv_4', 'conv_5']
        elif model_name == 'vgg_fast':
            self.cnn = models.vgg19(pretrained=True).features.to(self.device).eval()
            self.content_layers = ['conv_1']
            self.style_layers = ['conv_1', 'conv_3']
        else:
            raise RuntimeError()

        self.imsize = imsize

    def _process_image(self, img):
        loader = transforms.Compose([
            transforms.Resize(self.imsize),
            transforms.CenterCrop(self.imsize),
            transforms.ToTensor()])

        #image = Image.open(img)
        image = loader(img).unsqueeze(0)
        return image.to(self.device, torch.float)


    def _get_style_model_and_losses(self, normalization_mean, normalization_std,
                                   style_img, content_img,):
        cnn = copy.deepcopy(self.cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(self.device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0
        for layer in cnn.modules():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'mpool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            elif isinstance(layer, nn.Hardswish):
                name = 'hs_{}'.format(i)
                layer = nn.Hardswish(inplace=False)
            elif isinstance(layer, nn.AdaptiveAvgPool2d):
                name = 'apool_{}'.format(i)
            elif isinstance(layer, nn.Hardsigmoid):
                name = 'hsig_{}'.format(i)
                layer = nn.Hardsigmoid(inplace=False)
            else:
                continue

            model.add_module(name, layer)

            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    def _get_input_optimizer(self, input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    def transfer_style(self, content_img, style_img, num_steps=500, style_weight=100000, content_weight=1):
        normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(self.device)
#        normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(self.device),

        content_img = self._process_image(content_img)
        style_img = self._process_image(style_img)
        input_img = content_img.clone()

        model, style_losses, content_losses = self._get_style_model_and_losses(normalization_mean, torch.tensor([0.229, 0.224, 0.225]).to(self.device), style_img, content_img)
        optimizer = self._get_input_optimizer(input_img)
        run = [0]
        while run[0] <= num_steps:

            def closure():

                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return torchvision.transforms.ToPILImage()(input_img.detach()[0])


def process(content_image_path, style_image_path, model_name='vgg19', size=256, result_path=None):

    model = StyleTransferModel(size, model_name=model_name)

    content_image = Image.open(content_image_path)
    style_image = Image.open(style_image_path)

    output = model.transfer_style(content_img=content_image, style_img=style_image)

    if result_path is not None:
        output.save(result_path, format=content_image.format)

    return output

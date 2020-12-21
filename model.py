import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from torch.nn import Parameter
from torch.autograd import Variable
from timm.models import create_model

'''
 weight_decay = 1e-4
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)

    # model architecture is inspired humpback comp's solution.
    # I prepared cosface head and dense head for each outputs.
    
    # root
    x1 = Dense(512, kernel_initializer='he_normal', kernel_regularizer=regularizers.l2(weight_decay))(x)
    x1 = BatchNormalization()(x1)
    x1 = tf.nn.l2_normalize(x1, axis=1)
    root = CosFace(168, regularizer=regularizers.l2(weight_decay), name='root')([x1, r_label])
    x1 = Dense(168, use_bias=False)(x1)
    root2 = Lambda(lambda x: K.softmax(x), name='root2')(x1)
'''

class MyModel(nn.Module):
  def __init__(self, input_size, backbone, classes_number, pretrained=True, dropout=0.2):
    super(MyModel, self).__init__()
    self.input_size = input_size
    self.backbone = backbone
    self.classes_number = classes_number
    self.pretrained = pretrained
    self.dropout=dropout

    self.backbone = create_model(self.backbone, self.pretrained, num_classes=0)

    intermid = []
    feature_size = self.backbone.state_dict()['bn2.weight'].shape[0]
    intermid.append(nn.BatchNorm1d(feature_size))
    intermid.append(nn.Dropout(self.dropout))
    intermid.append(nn.Linear(feature_size, 512))
    intermid.append(nn.BatchNorm1d(512))

    self.intermid = nn.ModuleList(intermid)
    torch.nn.init.kaiming_normal_(self.intermid[2].weight)
    self.arc_face = Arcface(512, self.classes_number)
    self.head = nn.Linear(512, self.classes_number, bias=False)


  def forward(self, input, label):
    x = self.backbone(input)
    for inter in self.intermid:
      x = inter(x)

    x = F.normalize(x, p=2, dim=1)
    output = self.arc_face(x, label)
    output2 = self.head(x)

    return output, output2

class MultiHeadModel(nn.Module):
  
  def __init__(self, input_size, backbone, pretrained=True, dropout=0.2):
    super(MultiHeadModel, self).__init__()
    self.input_size = input_size
    self.backbone = backbone
    self.pretrained = pretrained
    self.dropout=dropout

    self.backbone = create_model(self.backbone, self.pretrained, num_classes=0)

    intermid = []
    feature_size = self.backbone.state_dict()['bn2.weight'].shape[0]
    intermid.append(nn.BatchNorm1d(feature_size))
    intermid.append(nn.Dropout(self.dropout))
    intermid.append(nn.Linear(feature_size, 512))
    intermid.append(nn.BatchNorm1d(512))

    self.intermid_root = nn.ModuleList(intermid)
    torch.nn.init.kaiming_normal_(self.intermid_root[2].weight)
    self.intermid_consonant = nn.ModuleList(intermid)
    torch.nn.init.kaiming_normal_(self.intermid_consonant[2].weight)
    self.intermid_vowel = nn.ModuleList(intermid)
    torch.nn.init.kaiming_normal_(self.intermid_vowel[2].weight)
    self.intermid_unique = nn.ModuleList(intermid)
    torch.nn.init.kaiming_normal_(self.intermid_unique[2].weight)
    
    self.arc_face_root = Arcface(512, 168)
    self.arc_face_consonant = Arcface(512, 11)
    self.arc_face_vowel = Arcface(512, 18)
    self.arc_face_unique = Arcface(512, 1295)

    self.head_root = nn.Linear(512, 168, bias=False)
    self.head_consonant = nn.Linear(512, 11, bias=False)
    self.head_vowel = nn.Linear(512, 18, bias=False)
    self.head_unique = nn.Linear(512, 1295, bias=False)

  def multi_head(self, input, root, consonant, vowel, unique):
    input = self.backbone(input)

    x1 = self.intermid_root[0](input)
    for inter in self.intermid_root[1:]:
      x1 = inter(x1)
    x1 = F.normalize(x1, p=2, dim=1)
    root = self.arc_face_root(x1, root)
    root2 = self.head_root(x1)
    
    x2 = self.intermid_consonant[0](input)
    for inter in self.intermid_consonant[1:]:
      x2 = inter(x2)
    x2 = F.normalize(x2, p=2, dim=1)
    consonant = self.arc_face_consonant(x2, consonant)
    consonant2 = self.head_consonant(x2)

    x3 = self.intermid_vowel[0](input)
    for inter in self.intermid_vowel[1:]:
      x3 = inter(x3)
    x3 = F.normalize(x3, p=2, dim=1)
    vowel = self.arc_face_vowel(x3, vowel)
    vowel2 = self.head_vowel(x3)

    x4 = self.intermid_unique[0](input)
    for inter in self.intermid_unique[1:]:
      x4 = inter(x4)
    x4 = F.normalize(x4, p=2, dim=1)
    unique = self.arc_face_unique(x4, unique)
    unique2 = self.head_unique(x4)

    return root, consonant, vowel, unique, root2, consonant2, vowel2, unique2


  def forward(self, input, root, consonant, vowel, unique):
    multi_head_outputs = self.multi_head(input, root, consonant, vowel, unique)

    return multi_head_outputs

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output

# class Arcface(nn.Module):
#     # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
#     def __init__(self, embedding_size=512, classnum=51332,  s=30., m=0.35):
#         super(Arcface, self).__init__()
#         self.classnum = classnum
#         self.weight = Parameter(torch.Tensor(embedding_size,classnum))
#         # initial kernel
#         self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
#         self.m = m # the margin value, default is 0.5
#         self.s = s # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
#         self.cos_m = math.cos(m)
#         self.sin_m = math.sin(m)
#         self.mm = self.sin_m * m  # issue 1
#         self.threshold = math.cos(math.pi - m)
#     def forward(self, embbedings, label):
#         # weights norm
#         nB = len(embbedings)
#         kernel_norm = l2_norm(self.weight,axis=0)
#         # cos(theta+m)
#         cos_theta = torch.mm(embbedings,kernel_norm)
# #         output = torch.mm(embbedings,kernel_norm)
#         cos_theta = cos_theta.clamp(-1,1) # for numerical stability
#         cos_theta_2 = torch.pow(cos_theta, 2)
#         sin_theta_2 = 1 - cos_theta_2
#         sin_theta = torch.sqrt(sin_theta_2)
#         cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
#         # this condition controls the theta+m should in range [0, pi]
#         #      0<=theta+m<=pi
#         #     -m<=theta<=pi-m
#         cond_v = cos_theta - self.threshold
#         cond_mask = cond_v <= 0
#         keep_val = (cos_theta - self.mm) # when theta not in [0,pi], use cosface instead
#         cos_theta_m[cond_mask] = keep_val[cond_mask]
#         output = cos_theta * 1.0 # a little bit hacky way to prevent in_place operation on cos_theta
#         idx_ = torch.arange(0, nB, dtype=torch.long)
#         output[idx_, label] = cos_theta_m[idx_, label]
#         output *= self.s # scale up in order to make softmax work, first introduced in normface
#         return output


class Arcface(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.35, easy_margin=False):
        super(Arcface, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output
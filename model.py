import torch
import torch.nn.functional as F
import math
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
  def __init__(self, input_size, backbone, pretrained=True, dropout=0.2):
    self.input_size = input_size
    self.backbone = backbone
    self.pretrained = pretrained
    self.dropout=dropout

    self.backbone = create_model(self.backbone, self.pretrained, num_classes=0)

    intermid = []
    feature_size = self.bachbone.state_dict()['bn2.weight'].shape[0]
    intermid.append(nn.BatchNorm(feature_size))
    intermid.append(nn.Dropout(self.dropout))
    intermid.append(nn.Linear(feature_size, 512))
    intermid.append(nn.BatchNorm(512))

    self.intermid_root = nn.ModuleList(intermid)
    self.intermid_consonant = nn.ModuleList(intermid)
    self.intermid_vowel = nn.ModuleList(intermid)
    self.intermid_unique = nn.ModuleList(intermid)
    
    self.arc_face_root = ArcMarginProduct(512, 168)
    self.arc_face_consonant = ArcMarginProduct(512, 11)
    self.arc_face_vowel = ArcMarginProduct(512, 18)
    self.arc_face_unique = ArcMarginProduct(512, 1292)

    self.head_root = nn.Linear(512, 168)
    self.head_consonant = nn.Linear(512, 11)
    self.head_vowel = nn.Linear(512, 18)
    self.head_unique = nn.Linear(512, 1292)

  def multi_head(self, input, label):
    input = self.backbone(input)

    x1 = intermid_root(input)
    x1 = F.normalize(x1, p=2, dim=1)
    root = self.arc_face_root(x1, label)
    root2 = self.head_root(x1)

    x2 = intermid_consonant(input)
    x2 = F.normalize(x2, p=2, dim=1)
    consonant = self.arc_face_consonant(x2, label)
    consonant2 = self.head_consonant(x2)

    x3 = intermid_vowel(input)
    x3 = F.normalize(x3, p=2, dim=1)
    vowel = self.arc_face_vowel(x3, label)
    vowel2 = self.head_vowel(x3)

    x4 = intermid_unique(input)
    x4 = F.normalize(x4, p=2, dim=1)
    unique = self.arc_face_unique(x4, label)
    unique2 = self.head_unique(x4)

    return root, consonant, vowel, unique, roo2, consonant2, vowel2, unique2



  def forward(self, input, label):


class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
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

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
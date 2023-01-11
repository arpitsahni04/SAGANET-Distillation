import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
from pointnet_util import get_graph_feature
from Attention.Cluster_Attention import MultiHeadAttention as MHA
from Attention.positionwiseFeedForward import PositionwiseFeedForward


class Attention(nn.Module):

	def __init__(self, Fea, q=1, v=1, h=1, dropout = 0.3):
		super(Attention, self).__init__()
		# attention
		# eight heads for now
		self.q, self.v, self.h = q, v, h
		self.dropout = dropout
		# input should be (batch, cluster, feature)
		# (b, N, feature)
		self.Fea = Fea

		self.skipAttention = MHA(self.Fea, self.q, self.v, self.h)
		self.feedForward = PositionwiseFeedForward(self.Fea)
		self.layerNorm1 = nn.LayerNorm(self.Fea)
		self.layerNorm2 = nn.LayerNorm(self.Fea)
		self.dropout = nn.Dropout(p=self.dropout)

	def forward(self, x):
		# print('here-------------------------')
		# print(x.shape)
		x = x.permute(0, 2, 1)
		residual = x
		x = self.skipAttention(query = x, key = x, value = x)
		x = self.dropout(x)
		x = self.layerNorm1(x + residual)

		# Feed forward
		residual = x
		x = self.feedForward(x)
		x = self.dropout(x)
		x = self.layerNorm2(x + residual)
		x = x.permute(0, 2, 1)
		return x


class Feature_Extractor_Student(nn.Module):
    # can probably refine more
    def __init__ (self,k, emb_dims, scale_encoder):
        super(Feature_Extractor_Student, self).__init__()
        self.k = k # ? for nearest neighbhor?
        self.emb_dims = emb_dims # output of Feature Extractor 
         # adjust intermediate
        self.conv1 = nn.Sequential(nn.Conv2d(6, int(256 * scale_encoder), kernel_size=1, bias=False),
                            nn.BatchNorm2d(int(256 * scale_encoder)),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.conv2 = nn.Sequential(nn.Conv2d(int(128 * 4 * scale_encoder),int(512 * scale_encoder),  kernel_size=1, bias=False),
                            nn.BatchNorm2d(int(scale_encoder*128*4)),
                            nn.LeakyReLU(negative_slope=0.2))
        
        self.conv3 =  nn.Sequential(nn.Conv2d(int(512 * 2 * scale_encoder), self.emb_dims, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.emb_dims),
                                   nn.LeakyReLU(negative_slope=0.2))
        self._attention6 = Attention(1024) # Attention Layer Unmodified
        self.maxpool = nn.MaxPool2d((1, 2048), 1)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x3 = self._attention6(x3)
        x3 = torch.squeeze(self.maxpool(x3), 2)
        
        output = x3.view(batch_size, -1, 1)
        return output, x1, x2 # x1, x2 are for Interchannel and output is for CLS


class Encoder_Student(nn.Module):
    def __init__(self, num_points,scale_encoder):
        super(Encoder_Student, self).__init__()
        self.fe1 = Feature_Extractor_Student(4, 1024,scale_encoder)
        # self.out_layer = nn.MaxPool2d((1, 2), 1)
    
    def forward(self, x):
        out_1, conv11, conv12 = self.fe1(x)  # (batch_size, 512, 3) || (batch_size, 1920)
        # out = torch.cat((out_1), 2)  # (batch_size, 1920, 2)
        # print(out.shape)
        out = out_1.view(-1, 1024)  # (batch_size, 1024)
        return out, conv11, conv12

class Student_SAGANET(nn.Module):
    def __init__ (self,num_points,crop_point_num,scale_decoder = 0.5,scale_encoder = 0.5):
        super(Student_SAGANET,self).__init__()
        self.crop_point_num = crop_point_num 
        # 3 Graph conv, self attention and Pool 
        self.latent_features = Encoder_Student(num_points,scale_encoder)
        self.scale_decoder = scale_decoder
        self.scale_encoder = scale_encoder
        self.latent_vector = None
        # Coarse Layers
        self.fc1 = nn.Linear(1024,int( 128 * self.crop_point_num*scale_decoder)) #x,de/2
        self.conv1_1 = torch.nn.Conv1d(int(self.crop_point_num*scale_decoder),int( 512 * scale_decoder), 1)
        self.conv1_2 = torch.nn.Conv1d(int(512 * scale_decoder), int((self.crop_point_num * 3) / 128), 1) # (512,48)
        
        # Fine Layers
        self.fc2 = nn.Linear(int( 128 * self.crop_point_num*scale_decoder), int(64 * 128*scale_decoder)) #x_2,de/4
        # self.fc2_1 = nn.Linear(512, 64 * 128)
        self.conv2_1 = torch.nn.Conv1d(int(128*scale_decoder), 6, 1)

    def forward(self,x):
        # get latent features from encoder
        x, conv11, conv12= self.latent_features(x)
        self.latent_vector = x
        x = F.relu(self.fc1(x))  # 1024
        x_2 = F.relu(self.fc2(x))# 512	v	# 2nd Channel coarse
        # x_2 = self.fc2_1(x_2)
        x_2 = x_2.reshape(-1, int(128*self.scale_decoder), 64)
        x_2 = self.conv2_1(x_2)
        
        #1st Channel fine
        x = x.reshape(-1, int(self.crop_point_num*self.scale_decoder), 128)
        x = F.relu(self.conv1_1(x)) # in-1024, out- 512*sc
        x = self.conv1_2(x)  # 12x128
        x = x.reshape(-1, 128, int(self.crop_point_num / 128), 3)
        print("Student x",x.shape)
        x_2 = x_2.reshape(-1, 128, 1, 3)
        print("Student x_2",x_2.shape)
        x = x + x_2  # 128x4x3
        x = x.reshape(-1, self.crop_point_num, 3)  
        print("Student Decoder Channel Shape",x_2.squeeze().shape, x.shape)
        return x_2.squeeze(), x, conv11, conv12,self.latent_vector
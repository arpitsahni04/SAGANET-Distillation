import _init_paths
from dataset.dataset import FullDataset
import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import *
import numpy as np
import data_utils as d_utils
from config import Configuration
from Encoder1024 import Decoder
from Student import Student_SAGANET
from D_net import D_net
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Distil:
    def __init__ (self,opt,teacher_net_trained):
        
        self.teacher_gen =  Decoder(opt.point_scales_list[0], opt.crop_point_num) # set teacher gen to eval mode
        state_dict  = torch.load(teacher_net_trained,map_location="cpu")
        self.teacher_gen.load_state_dict(state_dict, strict=False)
        self.teacher_gen

        self.Student_gen =  Student_SAGANET(opt.point_scales_list[0], opt.crop_point_num)
        self.Discriminator = D_net(opt.crop_point_num)
        self.alpha = opt.alpha_kd
        self.Student_Loss = PointLoss()
        self.DistillerLoss =  LatentDistiller_Loss() # Cosine Distance
        self.opt = opt

            


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)     

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

if __name__=="__main__":
    torch.cuda.empty_cache()
    
    config = Configuration('train')
    torch.backends.cudnn.enabled = False
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
    parser.add_argument('--trainingplots',
                        default='',
                        help='path to training plots')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batchSize', type=int, default=6, help='input batch size')
    parser.add_argument('--pnum', type=int, default=config.prior_num, help='the point number of a sample')
    parser.add_argument('--crop_point_num', type=int, default=config.crop_point_num, help='0 means do not use else use with this weight')
    parser.add_argument('--niter', type=int, default=120, help='number of epochs to train for')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--cuda', type=bool, default=True, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--num_scales', type=int, default=2, help='number of scales')
    parser.add_argument('--point_scales_list', type=list, default=[2048, 1024], help='number of points in each scales')
    parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
    parser.add_argument('--wtl2', type=float, default=0.99, help='0 means do not use else use with this weight')
    parser.add_argument('--wtemd', type=int, default=10, help='EMD weights')
    parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
    parser.add_argument('--netG', default='', help="put in gen_net.pth location to continue training)")
    parser.add_argument('--netD', default='', help="put in dis_net.pth location to continue training)")
    parser.add_argument('--cloud_size', type=int, default=config.partial_pcd_num, help='0 means do not use else use with this weight')
    parser.add_argument('--class_choice', default='Table', help='choice of class')    # [Car, Airplane, Bag, Cap, Chair, Guitar, Lamp, Laptop, Motorbike, Mug, Pistol, Skateboard, Table]
    parser.add_argument('--alpha_kd', default=0.3, help='Alpha to weigh distillation loss')
    parser.add_argument('--save_model_dir', default="Trained_Model_1", help='Path to Trained Teacher Network')

    writer = SummaryWriter("runs")
    MIN_dic = {'Car': 0.3,
            'Airplane': 0.16,
            'Bag': 0.5,
            'Cap': 1.5,
            'Chair': 0.25,
            'Guitar': 0.1,
            'Lamp': 1.5,
            'Laptop': 0.17,
            'Motorbike': 0.35,
            'Mug': 0.35,
            'Pistol': 0.4,
            'Skateboard': 0.4,
            'Table': 0.4}


    opt = parser.parse_args()
    MIN =  MIN_dic[opt.class_choice]

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    USE_CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load trained teacher model
    teacher_path = os.path.join(opt.save_model_dir, 'gen_net_Table_Attention110.pth') # find and save model in checkpoints
    
    Distiller = Distil(opt,teacher_path)
    # sys.exit()
    cudnn.benchmark = True  # faster runtime
    resume_epoch = 0

    count_student = count_parameters(Distiller.Student_gen) 
    count_teacher = count_parameters(Distiller.teacher_gen) 
    
    print('number of parameters in teacher ', count_teacher)
    print('number of parameters in student', count_student)
    print("Percentage Reduction in Parameters",(count_teacher-count_student)*100/count_teacher)
 
    if USE_CUDA:
        print("Using", torch.cuda.device_count(), "GPUs")
        Student_gen = torch.nn.DataParallel(Distiller.Student_gen)
        Student_gen.to(device)
        Student_gen.apply(weights_init_normal)
        dis_net = torch.nn.DataParallel(Distiller.Discriminator)
        dis_net.to(device)
        dis_net.apply(weights_init_normal)
        Distiller.teacher_gen.to(device).eval()
        
    if opt.netG != '':
        Student_gen.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netG)['epoch']
    if opt.netD != '':
        dis_net.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
        resume_epoch = torch.load(opt.netD)['epoch']

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    # define transforms for point cloud data
    transforms = transforms.Compose([d_utils.PointcloudToTensor(), ])

    # define transforms for point cloud data
    train_set = FullDataset("train")
    assert train_set
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=opt.batchSize, 
    shuffle=True, num_workers=int(opt.workers), drop_last = True)

    print("Trainloader length: ", len(train_loader))
    test_set = FullDataset('test')
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False,
                                              num_workers=int(opt.workers), drop_last = True)
    print("Test loader len: ", len(test_loader))
    writer = SummaryWriter("runs/")
    for i, data in enumerate(train_loader):
        resampled_data, real_data = data
        writer.add_graph(Distiller.teacher_gen, torch.randn(6, 2048, 3).to(device))
        #writer.add_graph(Student_gen, torch.randn(6, 2048, 3).to(device))

        break
    sys.exti()
    # criteria
    criterion = torch.nn.BCEWithLogitsLoss().to(device)  # discriminator loss
    criterion_PointLoss = PointLoss().to(device)
    criterion_PointLoss_test = PointLoss_test().to(device)
    # criterion_layer1 = nn.MSELoss().to(device)
    # criterion_layer2 = nn.MSELoss().to(device)
    # criterion_layer3 = nn.MSELoss().to(device)
    # criterion_layer4 = nn.MSELoss().to(device)
    # criterion_expansion = expansion.expansionPenaltyModule()
    real_label = 1
    fake_label = 0

    # setup optimizer
    optimizerD = torch.optim.Adam(dis_net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                weight_decay=opt.weight_decay)
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    optimizerG = torch.optim.Adam(Student_gen.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05,
                                weight_decay=opt.weight_decay)
    schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

    crop_point_num = int(opt.crop_point_num) #1024
    input_cropped1 = torch.FloatTensor(opt.batchSize, opt.pnum, 3) #(6,2048,3)
    label = torch.FloatTensor(opt.batchSize)

    num_batch = len(train_set) / opt.batchSize

    LOSS_pg, LOSS_gp = [], []
    EPOCH = []
    dis_loss = []
    gen_loss = []

for epoch in tqdm(resume_epoch, opt.niter):
    if epoch < 30:
        lam1 = 0.01
        lam2 = 0.02
    elif epoch < 80:
        lam1 = 0.05
        lam2 = 0.1
    else:
        lam1 = 0.1
        lam2 = 0.2
    
    if epoch < 100:
        wtl_mse = 1
        wtl_exp = 0.1
    else:
        wtl_mse = 0
        wtl_exp = 0.1
    for i, data in enumerate(train_loader):
        resampled_data, real_data = data
        #cloud(2048), model_points(1024) # both are labels 
          # 50% of the points are changed real center is the resampled points 
        batch_size = resampled_data.size()[0]  
        real_data = real_data.float()
        input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)

        input_cropped1 = input_cropped1.data.copy_(resampled_data)
        resampled_data = torch.unsqueeze(resampled_data, 1)
        input_cropped1 = torch.unsqueeze(input_cropped1, 1)  # input_cropped1.shape = [24, 1, 2024, 3]

        label.resize_([batch_size, 1]).fill_(real_label)
        if resampled_data.size()[0] < opt.batchSize: continue # only 6 batch size 
        # resampled_data = resampled_data.to(device)  # resampled_data.shape = [6, 1, 2048, 3]
        real_data = real_data.to(device)  # real_data.shape = [6, 1, 512, 3]
        input_cropped1 = input_cropped1.to(device)  # input_cropped1.shape = [6, 1, 2048, 3]
        label = label.to(device)  # real label construction done
        
        # obtain data for the two channels
        real_data = Variable(real_data, requires_grad=True) # real_data with fine
        real_data = torch.squeeze(real_data, 1)  # [24, 512, 3]
        real_data_key1_idx = utils.farthest_point_sample(real_data, 128, RAN=False) # key_1 for coarse
        real_data_key1 = utils.index_points(real_data, real_data_key1_idx)
        real_data_key1 = Variable(real_data_key1, requires_grad=True)  # [24, 128, 3]		
        input_cropped1 = torch.squeeze(input_cropped1, 1)
        input_cropped1 = Variable(input_cropped1, requires_grad=True)  # [24, 2048, 3]
  
        
        # Start Distillation()
        gen_net = gen_net.train()
        dis_net = dis_net.train()
        
        # update discriminator by passing real data
        dis_net.zero_grad()
        real_data = torch.unsqueeze(real_data, 1)
        real_out = dis_net(real_data)
        dis_err_real = criterion(real_out, label)
        dis_err_real.backward()
        
        # generate Fake pointclouds or current genrator prediction
        fake_center1, fake_fine, conv11, conv12= gen_net(input_cropped1)
        # feature_loss = criterion_layer1(conv11, conv21) + criterion_layer2(conv12, conv22) \
        
        # for teacher generator
        gen_fake_t = Distiller.teacher_gen()
        fake_out_t = dis_net(gen_fake_t)
        

        # pass fake data through discriminator
        fake_fine = torch.unsqueeze(fake_fine, 1)
        label.data.fill_(fake_label)
        fake_out = dis_net(fake_fine.detach())
        dis_err_fake = criterion(fake_out, label)
        dis_err_fake.backward()
        dis_err = dis_err_real + dis_err_fake
        
        if epoch % 4 == 0:    # Update discriminator weights
            optimizerD.step()
        
        # update generator objective max(log(D(G(z))))
        # Try to fool the discriminator
        gen_net.zero_grad()
        label.data.fill_(real_label)
        fake_out = dis_net(fake_fine)
        errG_D = criterion(fake_out, label)  # discriminator loss of fake points
        errG_l2 = 0
        
        # dist, _, mean_mst_dis = criterion_expansion(torch.squeeze(fake_fine), opt.crop_point_num//16, 1.0)
        # expansion_loss = torch.mean(dist)
        # double check these dimensions
        errG_l2 = criterion_PointLoss(torch.squeeze(fake_fine, 1), torch.squeeze(real_data, 1)) \
                  + lam1 * criterion_PointLoss(fake_center1, # coarse channel
                                               real_data_key1) #+ wtl_mse * feature_loss #+ wtl_exp * expansion_loss
        
        errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2  # original # need to be readjusted. for 
        errG.backward()
        optimizerG.step()
        print('Epoch[%d/%d] Batch[%d/%d] D_loss: %.4f G_loss: %.4f errG: %.4f errG_D: %.4f errG_l2: %.4f'
              % (epoch, opt.niter, i, len(train_loader),
                 dis_err.data, errG, errG, errG_D.data, errG_l2))
        f = open(opt.class_choice + '_loss.txt', 'a+')

    # start of testing
    MEAN_FLAG = False
    losses1, losses2 = [], []
    with torch.no_grad():
        print('After, ', epoch, '-th batch')
        for i, data in enumerate(test_loader):
            real_data, target = data
            batch_size = real_data.size()[0]
            if batch_size < opt.batchSize: continue
            real_data = real_data.float()
            input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            input_cropped1 = input_cropped1.data.copy_(real_data)
            
            real_data = torch.unsqueeze(real_data, 1)
            input_cropped1 = torch.unsqueeze(input_cropped1, 1)

            real_data = real_data.to(device)
            target = target.to(device)  
            real_data = torch.squeeze(real_data, 1)
            input_cropped1 = input_cropped1.to(device)
            input_cropped1 = torch.squeeze(input_cropped1, 1)
            input_cropped1 = Variable(input_cropped1, requires_grad=False)
            
            gen_net.eval()
            _, fake_fine, conv11, conv12= gen_net(input_cropped1)
            
            CD_loss = criterion_PointLoss(torch.squeeze(fake_fine, 1), torch.squeeze(target, 1))
            
            print('test CD loss: %.4f' % (CD_loss))
            f.write('\n' + 'test result:  %.4f' % (CD_loss))
            
            if CD_loss.item() > MIN and i == 0:
                break
            
            _, dist1, dist2 = criterion_PointLoss_test(torch.squeeze(fake_fine, 1), torch.squeeze(target, 1))
            losses2.append(dist1.item())
            losses1.append(dist2.item())
            
            MEAN_FLAG = True
    
    if MEAN_FLAG:
        
        loss_pg, loss_gp = np.mean(losses1) * 10, np.mean(losses2) * 10
        print('mean CD loss pred->GT|GT->pred:', loss_pg, loss_gp)
        f.write('mean CD loss pred->GT|GT->pred: %.5f, %.5f' % (loss_pg, loss_gp))
        
        LOSS_pg.append(loss_pg)
        LOSS_gp.append(loss_gp)
        EPOCH.append(epoch)
        
        first_min = min(LOSS_pg)
        first_idx = LOSS_pg.index(first_min)
        second_min = min(LOSS_gp)
        second_idx = LOSS_gp.index(second_min)
        draw_result_pggp(EPOCH, LOSS_pg, LOSS_gp, opt.trainingplots + str(epoch), opt.class_choice)
        
        if loss_pg == first_min:
            torch.save({'epoch': epoch + 1,
                        'state_dict': gen_net.state_dict()},
                       'Trained_Model_1/gen_net_' + opt.class_choice + '_Attention' + str(epoch) + '.pth')
            torch.save({'epoch': epoch + 1,
                        'state_dict': dis_net.state_dict()},
                       'Trained_Model_1/dis_net_' + opt.class_choice + '_Attention' + str(epoch) + '.pth')
        elif loss_gp == second_min:
            torch.save({'epoch': epoch + 1,
                        'state_dict': gen_net.state_dict()},
                       'Trained_Model_1/gen_net_' + opt.class_choice + '_Attention' + str(epoch) + '.pth')
            torch.save({'epoch': epoch + 1,
                        'state_dict': dis_net.state_dict()},
                       'Trained_Model_1/dis_net_' + opt.class_choice + '_Attention' + str(epoch) + '.pth')
        
        print('best so far (pg): ', first_min, LOSS_gp[first_idx])
        print('best so far (gp): ', LOSS_pg[second_idx], second_min)
        f.write('best so far (pg): %.5f, %.5f' % (first_min, LOSS_gp[first_idx]))
        f.write('best so far (gp): %.5f, %.5f' % (LOSS_pg[second_idx], second_min))
    
    f.close()
    schedulerD.step()
    schedulerG.step()

print('done')
print('Epochs: ', EPOCH)



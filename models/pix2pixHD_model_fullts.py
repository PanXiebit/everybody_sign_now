### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import pytorch_lightning as pl



class Pix2PixHDModel(pl.LightningModule):
    def name(self):
        return 'Pix2PixHDModel'

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        if opt.resize_or_crop != 'none': # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor

        ##### define networks        
        # Generator network
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1          
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 4*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss)

        # Face discriminator network
        if self.isTrain and opt.face_discrim:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 2*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netDface = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          1, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids, netD='face')

        #Face residual network
        if opt.face_generator:
            if opt.faceGtype == 'unet':
                self.faceGen = networks.define_G(opt.output_nc*2, opt.output_nc, 32, 'unet', 
                                          n_downsample_global=2, n_blocks_global=5, n_local_enhancers=0, 
                                          n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            elif opt.faceGtype == 'global':
                self.faceGen = networks.define_G(opt.output_nc*2, opt.output_nc, 64, 'global', 
                                      n_downsample_global=3, n_blocks_global=5, n_local_enhancers=0, 
                                      n_blocks_local=0, norm=opt.norm, gpu_ids=self.gpu_ids)
            else:
                raise('face generator not implemented!')
            
        print('---------- Networks initialized -------------')
        self.save_hyperparameters()
    

    def loss_func(self, opt):
        # set loss functions and optimizers
        if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
            raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
        self.fake_pool = ImagePool(opt.pool_size)
        self.old_lr = opt.lr

        # define loss functions
        self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
        self.criterionFeat = torch.nn.L1Loss()
        if not opt.no_vgg_loss:             
            self.criterionVGG = networks.VGGLoss()
        if opt.use_l1:
            self.criterionL1 = torch.nn.L1Loss()
    
        # Loss names
        self.loss_names = ['G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'G_GANface', 'D_realface', 'D_fakeface']


    def configure_optimizers(self, ):
        # initialize optimizers
        # optimizer G
        if self.opt.niter_fix_global > 0:
            print('------------- Only training the local enhancer network (for %d epochs) ------------' % self.opt.niter_fix_global)
            params_dict = dict(self.netG.named_parameters())
            params = []
            for key, value in params_dict.items():       
                if key.startswith('model' + str(self.opt.n_local_enhancers)):
                    params += [{'params':[value],'lr':self.opt.lr}]
                else:
                    params += [{'params':[value],'lr':0.0}]                            
        else:
            params = list(self.netG.parameters())

        if self.opt.face_generator:
            params = list(self.faceGen.parameters())
        else:
            if self.opt.niter_fix_main == 0:
                params += list(self.netG.parameters())

        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))                            

        # optimizer D
        if self.opt.niter_fix_main > 0:
            print('------------- Only training the face discriminator network (for %d epochs) ------------' % self.opt.niter_fix_main)
            params = list(self.netDface.parameters())                         
        else:
            if self.opt.face_discrim:
                params = list(self.netD.parameters()) + list(self.netDface.parameters())   
            else:
                params = list(self.netD.parameters())                  

        self.optimizer_D = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

        return [self.optimizer_G, self.optimizer_D]


    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminate_4(self, s0, s1, i0, i1, use_pool=False):
        input_concat = torch.cat((s0, s1, i0.detach(), i1.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def discriminateface(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netDface.forward(fake_query)
        else:
            return self.netDface.forward(input_concat)



    def forward(self, label, next_label, image, next_image, face_coords, zeroshere, infer=False):
        # Encode Inputs
        input_label, real_image, next_label, next_image, zeroshere = self.encode_input(label, image, \
                     next_label=next_label, next_image=next_image, zeroshere=zeroshere)
        

        if self.opt.face_discrim:
            miny = face_coords.data[0][0]
            maxy = face_coords.data[0][1]
            minx = face_coords.data[0][2]
            maxx = face_coords.data[0][3]

        initial_I_0 = 0

        # Fake Generation I_0
        input_concat = torch.cat((input_label, zeroshere), dim=1) 

        #face residual for I_0
        face_residual_0 = 0
        if self.opt.face_generator:
            initial_I_0 = self.netG.forward(input_concat)
            face_label_0 = input_label[:, :, miny:maxy, minx:maxx]
            face_residual_0 = self.faceGen.forward(torch.cat((face_label_0, initial_I_0[:, :, miny:maxy, minx:maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, miny:maxy, minx:maxx] = initial_I_0[:, :, miny:maxy, minx:maxx] + face_residual_0
        else:
            I_0 = self.netG.forward(input_concat)


        input_concat1 = torch.cat((next_label, I_0), dim=1)

        #face residual for I_1
        face_residual_1 = 0
        if self.opt.face_generator:
            initial_I_1 = self.netG.forward(input_concat1)
            face_label_1 = next_label[:, :, miny:maxy, minx:maxx]
            face_residual_1 = self.faceGen.forward(torch.cat((face_label_1, initial_I_1[:, :, miny:maxy, minx:maxx]), dim=1))
            I_1 = initial_I_1.clone()
            I_1[:, :, miny:maxy, minx:maxx] = initial_I_1[:, :, miny:maxy, minx:maxx] + face_residual_1
        else:
            I_1 = self.netG.forward(input_concat1)

        loss_D_fake_face = loss_D_real_face = loss_G_GAN_face = 0
        fake_face_0 = fake_face_1 = real_face_0 = real_face_1 = 0
        fake_face = real_face = face_residual = 0
        if self.opt.face_discrim:

            fake_face_0 = I_0[:, :, miny:maxy, minx:maxx]
            fake_face_1 = I_1[:, :, miny:maxy, minx:maxx]
            real_face_0 = real_image[:, :, miny:maxy, minx:maxx]
            real_face_1 = next_image[:, :, miny:maxy, minx:maxx]

            # Fake Detection and Loss
            pred_fake_pool_face = self.discriminateface(face_label_0, fake_face_0, use_pool=True)
            loss_D_fake_face += 0.5 * self.criterionGAN(pred_fake_pool_face, False)

            # Face Real Detection and Loss        
            pred_real_face = self.discriminateface(face_label_0, real_face_0)
            loss_D_real_face += 0.5 * self.criterionGAN(pred_real_face, True)

            # Face GAN loss (Fake Passability Loss)        
            pred_fake_face = self.netDface.forward(torch.cat((face_label_0, fake_face_0), dim=1))        
            loss_G_GAN_face += 0.5 * self.criterionGAN(pred_fake_face, True)

            pred_fake_pool_face = self.discriminateface(face_label_1, fake_face_1, use_pool=True)
            loss_D_fake_face += 0.5 * self.criterionGAN(pred_fake_pool_face, False)

            # Face Real Detection and Loss        
            pred_real_face = self.discriminateface(face_label_1, real_face_1)
            loss_D_real_face += 0.5 * self.criterionGAN(pred_real_face, True)

            # Face GAN loss (Fake Passability Loss)        
            pred_fake_face = self.netDface.forward(torch.cat((face_label_1, fake_face_1), dim=1))        
            loss_G_GAN_face += 0.5 * self.criterionGAN(pred_fake_face, True)

            fake_face = torch.cat((fake_face_0, fake_face_1), dim=3)
            real_face = torch.cat((real_face_0, real_face_1), dim=3)

            if self.opt.face_generator:
                face_residual = torch.cat((face_residual_0, face_residual_1), dim=3)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate_4(input_label, next_label, I_0, I_1, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate_4(input_label, next_label, real_image, next_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, next_label, I_0, I_1), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * \
                        self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG0 = self.criterionVGG(I_0, real_image) * self.opt.lambda_feat
            loss_G_VGG1 = self.criterionVGG(I_1, next_image) * self.opt.lambda_feat
            loss_G_VGG = loss_G_VGG0 + loss_G_VGG1 
            if self.opt.netG == 'global': #need 2x VGG for artifacts when training local
                loss_G_VGG *= 0.5
            if self.opt.face_discrim:
                loss_G_VGG += 0.5 * self.criterionVGG(fake_face_0, real_face_0) * self.opt.lambda_feat
                loss_G_VGG += 0.5 * self.criterionVGG(fake_face_1, real_face_1) * self.opt.lambda_feat

        if self.opt.use_l1:
            loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A
        
        # Only return the fake_B image if necessary to save BW
        return [[loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_G_GAN_face, loss_D_real_face,  loss_D_fake_face], 
            None if not infer else [torch.cat((I_0, I_1), dim=3), fake_face, face_residual, initial_I_0]]

    def training_step(self, batch, batch_idx, mode="train"):
        input_label = batch["label_1"]
        next_label = batch["label_2"]
        real_image = batch["rgb_1"]
        next_image = batch["rgb_2"]
        zeroshere = torch.zeros(input_label).float()
        

    def inference(self, label, prevouts, face_coords):

        # Encode Inputs        
        input_label, _, _, _, prevouts = self.encode_input(Variable(label), zeroshere=Variable(prevouts), infer=True)

        if self.opt.face_generator:
            miny = face_coords[0][0]
            maxy = face_coords[0][1]
            minx = face_coords[0][2]
            maxx = face_coords[0][3]

        """ new face """
        I_0 = 0
        # Fake Generation

        input_concat = torch.cat((input_label, prevouts), dim=1) 
        initial_I_0 = self.netG.forward(input_concat)

        if self.opt.face_generator:
            face_label_0 = input_label[:, :, miny:maxy, minx:maxx]
            face_residual_0 = self.faceGen.forward(torch.cat((face_label_0, initial_I_0[:, :, miny:maxy, minx:maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, miny:maxy, minx:maxx] = initial_I_0[:, :, miny:maxy, minx:maxx] + face_residual_0
            fake_face_0 = I_0[:, :, miny:maxy, minx:maxx]
            return I_0
        return initial_I_0

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.opt.face_discrim:
             self.save_network(self.netDface, 'Dface', which_epoch, self.gpu_ids)
        if self.opt.face_generator:
            self.save_network(self.faceGen, 'Gface', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.opt.face_generator:
            params += list(self.faceGen.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning global generator -----------')

    def update_fixed_params_netD(self):
        params = list(self.netD.parameters()) + list(self.netDface.parameters())         
        self.optimizer_D = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999)) 
        print('------------ Now also finetuning multiscale discriminator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def load_network(self, opt):
        # load networks
        if (not self.isTrain or opt.continue_train or opt.load_pretrain):
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
                if opt.face_discrim:
                    self.load_network(self.netDface, 'Dface', opt.which_epoch, pretrained_path)
            if opt.face_generator:
                self.load_network(self.faceGen, 'Gface', opt.which_epoch, pretrained_path)

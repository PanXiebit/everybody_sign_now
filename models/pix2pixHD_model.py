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
import torchvision



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
        print("####################")
        print("###   Generator  ###")
        print("####################")
        
        netG_input_nc = opt.label_nc
        if not opt.no_instance:
            netG_input_nc += 1          
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm)        

        # Discriminator network

        print("#####################")
        print("### Discriminator ###")
        print("#####################")
        use_sigmoid = opt.no_lsgan
        netD_input_nc = 4*opt.output_nc
        if not opt.no_instance:
            netD_input_nc += 1
        self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                        opt.num_D, not opt.no_ganFeat_loss)

        # Face discriminator network
        if self.isTrain and opt.hand_discrim:
            print("##########################")
            print("### Face Discriminator ###")
            print("##########################")
            use_sigmoid = opt.no_lsgan
            netD_input_nc = 2*opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netDface = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 1, 
                not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids, netD='face')

        #Face residual network
        
        if opt.hand_generator:
            print("###############################")
            print("### Face Residual Generator ###")
            print("###############################")
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

        if opt.pool_size > 0:
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
        
        self.save_hyperparameters()


    def configure_optimizers(self):
        if self.isTrain:
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

            if self.opt.hand_generator:
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
                if self.opt.hand_discrim:
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


    def training_step(self, batch, batch_idx, optimizer_idx):
        input_label = batch["label_0"]
        next_label = batch["label_1"]
        real_image = batch["rgb_0"]
        next_image = batch["rgb_1"]

        l_miny, l_maxy, l_minx, l_maxx = batch["lhand_0_coords"]
        r_miny, r_maxy, r_minx, r_maxx = batch["rhand_0_coords"]

        zerohere = torch.zeros(input_label.size()).float().to(input_label.device)
        input_concat_0 = torch.cat((input_label, zerohere), dim=1)
        
        # face residual for I_0
        if self.opt.hand_generator:
            initial_I_0 = self.netG.forward(input_concat_0)
            lhand_label_0 = input_label[:, :, l_miny:l_maxy, l_minx:l_maxx]
            rhand_label_0 = input_label[:, :, r_miny:r_maxy, r_minx:r_maxx]
            lhand_residual_0 = self.faceGen.forward(torch.cat((lhand_label_0, initial_I_0[:, :, l_miny:l_maxy, l_minx:l_maxx]), dim=1))
            rhand_residual_0 = self.faceGen.forward(torch.cat((rhand_label_0, initial_I_0[:, :, r_miny:r_maxy, r_minx:r_maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, l_miny:l_maxy, l_minx:l_maxx] = initial_I_0[:, :, l_miny:l_maxy, l_minx:l_maxx] + lhand_residual_0
            I_0[:, :, r_miny:r_maxy, r_minx:r_maxx] = initial_I_0[:, :, r_miny:r_maxy, r_minx:r_maxx] + rhand_residual_0
        else:
            I_0 = self.netG.forward(input_concat_0)

        input_concat_1 = torch.cat((next_label, I_0), dim=1)

        #face residual for I_1
        if self.opt.hand_generator:
            initial_I_1 = self.netG.forward(input_concat_1)
            lhand_label_1 = next_label[:, :, l_miny:l_maxy, l_minx:l_maxx]
            rhand_label_1 = next_label[:, :, r_miny:r_maxy, r_minx:r_maxx]
            lhand_residual_1 = self.faceGen.forward(torch.cat((lhand_label_1, initial_I_1[:, :, l_miny:l_maxy, l_minx:l_maxx]), dim=1))
            rhand_residual_1 = self.faceGen.forward(torch.cat((rhand_label_1, initial_I_1[:, :, r_miny:r_maxy, r_minx:r_maxx]), dim=1))
            I_1 = initial_I_1.clone()
            I_1[:, :, l_miny:l_maxy, l_minx:l_maxx] = initial_I_1[:, :, l_miny:l_maxy, l_minx:l_maxx] + lhand_residual_1
            I_1[:, :, r_miny:r_maxy, r_minx:r_maxx] = initial_I_1[:, :, r_miny:r_maxy, r_minx:r_maxx] + rhand_residual_1
        else:
            I_1 = self.netG.forward(input_concat_1)
        
        if self.opt.hand_discrim:
            fake_lhand_0 = I_0[:, :, l_miny:l_maxy, l_minx:l_maxx]
            fake_lhand_1 = I_1[:, :, l_miny:l_maxy, l_minx:l_maxx]
            real_lhand_0 = real_image[:, :, l_miny:r_maxy, l_minx:l_maxx]
            real_lhand_1 = next_image[:, :, l_miny:r_maxy, l_minx:l_maxx]

        pred_real = self.discriminate_4(input_label, next_label, real_image, next_image)
        pred_fake_pool = self.discriminate_4(input_label, next_label, I_0, I_1, use_pool=True)

        # visualize
        if batch_idx % 5 == 0:            
            input_label_vis = (torch.clamp(input_label, -1, 1.0) + 1.0) / 2
            input_label_vis = torchvision.utils.make_grid(input_label_vis[0])
            real_image_vis = (torch.clamp(real_image, -1, 1.0) + 1.0) / 2
            real_image_vis = torchvision.utils.make_grid(real_image_vis[0])

            pred_image_vis = (torch.clamp(I_0, -1, 1.0) + 1.0) / 2 
            pred_image_vis = torchvision.utils.make_grid(pred_image_vis[0])


            self.logger.experiment.add_image("input_label", input_label_vis, self.global_step)
            self.logger.experiment.add_image("real_image", real_image_vis, self.global_step)
            self.logger.experiment.add_image("pred_image", pred_image_vis, self.global_step)

        if optimizer_idx == 0:
            if self.opt.hand_discrim:
                # Generator loss for lhand_0 and lhand_1     
                pred_fake_lhand_0 = self.netDface.forward(torch.cat((lhand_label_0, fake_lhand_0), dim=1))        
                loss_G_GAN_lhand = 0.5 * self.criterionGAN(pred_fake_lhand_0, True)

                pred_fake_lhand_1 = self.netDface.forward(torch.cat((lhand_label_1, fake_lhand_1), dim=1))        
                loss_G_GAN_lhand += 0.5 * self.criterionGAN(pred_fake_lhand_1, True)

                fake_lhand = torch.cat((fake_lhand_0, fake_lhand_1), dim=3)
                real_lhand = torch.cat((real_lhand_0, real_lhand_1), dim=3)

                if self.opt.hand_generator:
                    lhand_residual = torch.cat((lhand_residual_0, lhand_residual_1), dim=3)

            # Generator loss 
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
                if self.opt.hand_discrim:
                    loss_G_VGG += 0.5 * self.criterionVGG(fake_lhand_0, real_lhand_0) * self.opt.lambda_feat
                    loss_G_VGG += 0.5 * self.criterionVGG(fake_lhand_1, real_lhand_1) * self.opt.lambda_feat

            if self.opt.use_l1:
                loss_G_VGG += (self.criterionL1(I_1, next_image)) * self.opt.lambda_A

            self.log('train/loss_G_GAN', loss_G_GAN, prog_bar=True)
            self.log('train/loss_G_GAN_Feat', loss_G_GAN_Feat, prog_bar=True)
            self.log('train/loss_G_VGG', loss_G_VGG, prog_bar=True)


            if self.opt.hand_discrim:
                self.log('train/loss_G_GAN_lhand', loss_G_GAN_lhand, prog_bar=True)
                G_loss = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG + loss_G_GAN_lhand
            else:
                G_loss = loss_G_GAN + loss_G_GAN_Feat + loss_G_VGG
            return G_loss

        if optimizer_idx == 1:
            if self.opt.hand_discrim:
                # Fake Detection and Loss
                pred_fake_pool_lhand = self.discriminateface(lhand_label_0, fake_lhand_0, use_pool=True)
                loss_D_fake_lhand = 0.5 * self.criterionGAN(pred_fake_pool_lhand, False)

                # Face Real Detection and Loss        
                pred_real_lhand = self.discriminateface(lhand_label_0, real_lhand_0)
                loss_D_real_lhand = 0.5 * self.criterionGAN(pred_real_lhand, True)

                pred_fake_pool_lhand = self.discriminateface(lhand_label_1, fake_lhand_1, use_pool=True)
                loss_D_fake_lhand += 0.5 * self.criterionGAN(pred_fake_pool_lhand, False)

                # Face Real Detection and Loss        
                pred_real_lhand = self.discriminateface(lhand_label_1, real_lhand_1)
                loss_D_real_lhand += 0.5 * self.criterionGAN(pred_real_lhand, True)

            # Discriminator loss
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)      

            # Real Detection and Loss        
            loss_D_real = self.criterionGAN(pred_real, True)

            D_loss = loss_D_real + loss_D_fake
            
            self.log('train/loss_D_real', loss_D_real, prog_bar=True)
            self.log('train/loss_D_fake', loss_D_fake, prog_bar=True)

            if self.opt.hand_discrim:
                self.log('train/loss_D_fake_lhand', loss_D_fake_lhand, prog_bar=True)
                return D_loss + loss_D_real_lhand + loss_D_fake_lhand
            return D_loss

    def validation_step(self, batch, batch_idx):
        if batch_idx > 5: return
        input_label = batch["label_0"]
        next_label = batch["label_1"]
        real_image = batch["rgb_0"]
        next_image = batch["rgb_1"]

        l_miny, l_maxy, l_minx, l_maxx = batch["lhand_0_coords"]
        r_miny, r_maxy, r_minx, r_maxx = batch["rhand_0_coords"]

        zerohere = torch.zeros(input_label.size()).float().to(input_label.device)

        """ new face """
        I_0 = 0
        # Fake Generation
        prevouts = zerohere

        input_concat = torch.cat((input_label, prevouts), dim=1) 
        initial_I_0 = self.netG.forward(input_concat)

        

        if self.opt.hand_generator:
            lhand_label_0 = input_label[:, :, l_miny:l_maxy, l_minx:l_maxx]
            lhand_residual_0 = self.faceGen.forward(torch.cat((lhand_label_0, initial_I_0[:, :, l_miny:l_maxy, l_minx:l_maxx]), dim=1))
            I_0 = initial_I_0.clone()
            I_0[:, :, l_miny:l_maxy, l_minx:l_maxx] = initial_I_0[:, :, l_miny:l_maxy, l_minx:l_maxx] + lhand_residual_0
            fake_lhand_0 = I_0[:, :, l_miny:l_maxy, l_minx:l_maxx]
        else:
            I_0 = initial_I_0
        
        input_label_vis = (torch.clamp(input_label, -1, 1.0) + 1.0) / 2
        input_label_vis = torchvision.utils.make_grid(input_label_vis[0])
        real_image_vis = (torch.clamp(real_image, -1, 1.0) + 1.0) / 2
        real_image_vis = torchvision.utils.make_grid(real_image_vis[0])

        pred_image_vis = (torch.clamp(initial_I_0, -1, 1.0) + 1.0) / 2 
        pred_image_vis = torchvision.utils.make_grid(pred_image_vis[0])


        self.logger.experiment.add_image("input_label", input_label_vis, self.current_epoch)
        self.logger.experiment.add_image("real_image", real_image_vis, self.current_epoch)
        self.logger.experiment.add_image("pred_image", pred_image_vis, self.current_epoch)

        return I_0

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        return edge.float()


    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vqperceptual import vanilla_d_loss, hinge_d_loss

from .lpips import LPIPS
from .discriminator import NLayerDiscriminator, NLayerDiscriminator3d, weights_init
from .multiscale_discriminator import MultiscaleDiscriminator, GANLoss


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class VQLPIPSWithDiscriminatorDance(nn.Module):
    def __init__(self, disc_start, codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="lsgan"):
        super().__init__()
        assert disc_loss in ["hinge", "vanilla", "lsgan"]
        self.codebook_weight = codebook_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight

        # self.discriminator = NLayerDiscriminator3d(input_nc=disc_in_channels,
        #                                            n_layers=disc_num_layers,
        #                                            use_actnorm=use_actnorm,
        #                                            ndf=disc_ndf
        #                                            ).apply(weights_init)
        self.discriminator = MultiscaleDiscriminator(input_nc=disc_in_channels,
                                                     n_layers=disc_num_layers,
                                                     ndf=disc_ndf,
                                                     num_D=3).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = vanilla_d_loss
        # print(f"VQLPIPSWithDiscriminator running with {disc_loss} loss.")
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def compute_loss(self, pred_logits):
        loss = 0
        for preds in pred_logits:
            pred = preds[-1]
            loss += torch.mean(pred)
        return loss

    def forward(self, codebook_loss, inputs, reconstructions, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train"):
        """inputs: [bs, 3, T, 128, 128], [bs, 3, T, 128, 128]
        """
        # reconstruction loss
        bs, c, t, h, w = inputs.size()
        inputs = inputs.permute(0, 2, 1, 3, 4).contiguous().flatten(0,1) # [bs*t, c, h, w]
        reconstructions = reconstructions.permute(0, 2, 1, 3, 4).contiguous().flatten(0,1)
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) # [bs*t, c, h, w]
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs, reconstructions)
            # p_loss = p_loss.view(bs, t, *p_loss.size()[1:]).permute(0, 2, 1, 3, 4).contiguous()
            rec_loss = rec_loss + self.perceptual_weight * p_loss
        else:
            p_loss = torch.tensor([0.0])
        
        nll_loss = rec_loss
        nll_loss = torch.mean(nll_loss)
        
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
                # logits_fake = self.disc_loss(logits_fake, target_is_real=True)
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
                # logits_fake = self.disc_loss(logits_fake, target_is_real=True)
            
            g_loss = - self.compute_loss(logits_fake)

            # try:
            #     d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            # except RuntimeError:
            #     assert not self.training
            #     d_weight = torch.tensor(0.0)

            # zero or one? for disc_factor.
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            # print("disc_factor: ", disc_factor, optimizer_idx, global_step)
            
            loss = nll_loss + disc_factor * g_loss + self.codebook_weight * codebook_loss.mean()

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(),
                   "{}/quant_loss".format(split): codebook_loss.detach().mean(),
                   "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/p_loss".format(split): p_loss.detach().mean(),
                #    "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
                # logits_real = self.disc_loss(logits_real, target_is_real=True)
                # logits_fake = self.disc_loss(logits_fake, target_is_real=False)
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))
                # logits_real = self.disc_loss(logits_real, target_is_real=True)
                # logits_fake = self.disc_loss(logits_fake, target_is_real=False)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)

            logits_real = self.compute_loss(logits_real)
            logits_fake = self.compute_loss(logits_fake)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


if __name__ == "__main__":
    vd_loss = VQLPIPSWithDiscriminatorDance(disc_start=0, 
                 codebook_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_ndf=64, disc_loss="lsgan")
    # print(vd_loss)

    inputs = torch.randn(2, 3, 32, 256, 256)
    recons = torch.randn(2, 3, 32, 256, 256)

    optimizer_idx = 0
    global_step = 0
    last_layer = None

    loss, _ = vd_loss(torch.tensor(0.23), inputs, recons, optimizer_idx, global_step)
    print(loss)



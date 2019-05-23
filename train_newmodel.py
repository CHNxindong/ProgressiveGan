from dataloader.DALI_tfrecordloader import Lsun_Loader
from model import G_paper,D_paper
import torch
from torch.autograd import Variable,grad
from torch.optim import Adam
from misc import propress_real
from math import floor
from misc import adjust_dynamic_range,upscale_phase
import os
from misc import save_image_grid
from tensorboardX import SummaryWriter
import logging

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
class Trainschedule:
    def __init__(self,
                 lod_transition_kimg=1000,
                 lod_stable_kimg=1000,
                 minibatch_base=128,
                 minibatch_dict={5:64,6:32,7:16,8:8},
                 save_kimg_tick={2:100,3:40,4:40,5:30,6:20,7:10,8:10},
                 initial_resolution=2,
                 max_resolution=8
                 ):
        self.initial_resolution=initial_resolution
        self.max_resolution=max_resolution

        self.save_kimg_tick=save_kimg_tick
        self.tick=0
        self.last_cur_phase_kimg=0

        self.minibatch_dict=minibatch_dict
        self.minibatch_base=minibatch_base
        self.batchsize=minibatch_dict.get(initial_resolution,minibatch_base)

        self.cur_img=0
        self.lod=initial_resolution
        self.phase=initial_resolution
        self.lod_transition_kimg=lod_transition_kimg
        self.lod_stable_kimg=lod_stable_kimg

        self.NEW_PHASE=False
        self.TRANSITION=False

    def update(self):
        cur_phase_kimg=self.cur_img/1000-(self.phase-self.initial_resolution)*(self.lod_stable_kimg+self.lod_transition_kimg)
        #save tick
        if(floor(cur_phase_kimg)!=self.last_cur_phase_kimg):
            self.last_cur_phase_kimg=floor(cur_phase_kimg)
            self.tick+=1

        if cur_phase_kimg>(self.lod_transition_kimg+self.lod_stable_kimg):
            if self.phase<self.max_resolution:
                self.phase+=1
                self.batchsize=self.minibatch_dict.get(self.phase,self.minibatch_base)
                self.NEW_PHASE=True
                self.TRANSITION=True
                print("update phase------------------------")
            else:
                print("lod:%.2f phase:%d stable cur_phase_kimg:%.2f BATCH:%d" % (
                self.lod, self.phase, cur_phase_kimg, self.batchsize))

        elif cur_phase_kimg<self.lod_transition_kimg:
            print("lod:%.2f phase:%d transition cur_phase_kimg:%.2f BATCH:%d" % (
                self.lod, self.phase, cur_phase_kimg, self.batchsize))
            self.lod = self.phase + (cur_phase_kimg / self.lod_transition_kimg)

        else:
            self.TRANSITION=False
            print("lod:%.2f phase:%d stable cur_phase_kimg:%.2f BATCH:%d"%(self.lod,self.phase,cur_phase_kimg,self.batchsize))

class Train:
    def __init__(self,
                 total_kimg,
                 D_repeat=2,
                 max_resolution=8):
        self.total_kimg=total_kimg
        self.D_repeat=D_repeat
        self.sche=Trainschedule()
        self.G=G_paper(max_resolution).cuda()
        #self.Gs=deepcopy(self.G)
        self.D=D_paper(max_resolution).cuda()

        self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=3e-3, betas=(0.0, 0.99))
        self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=3e-3, betas=(0.0, 0.99))
        self.dataloader = Lsun_Loader(device_idx=1)
        self.dataloader.update(self.sche.phase, self.sche.batchsize)
        self.z=torch.randn(24, 512)
        self.logger,self.writer=self.get_Logger_and_SummaryWriter()


    def get_Logger_and_SummaryWriter(self):
        for i in range(1000):
            self.tag_dir = 'checkpoints/tensorboard/try_{}'.format(i)
            if not os.path.exists(self.tag_dir):
                os.makedirs(self.tag_dir,exist_ok=True)
                logger = logging.getLogger("PGGAN")
                file_handler = logging.FileHandler(os.path.join(self.tag_dir, 'log.txt'), "w")
                stdout_handler = logging.StreamHandler()
                logger.addHandler(file_handler)
                logger.addHandler(stdout_handler)
                stdout_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                file_handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
                logger.setLevel(logging.INFO)
                return logger,SummaryWriter(self.tag_dir)

    def gradient_penalty(self,x, y, D):
        # interpolation
        shape = [x.size(0)] + [1] * (x.dim() - 1)
        alpha = torch.rand(shape).to(x.device)
        z = x + alpha * (y - x)
        # gradient penalty
        z = Variable(z, requires_grad=True).to(x.device)
        o = D(z,self.sche.lod,self.sche.phase,self.sche.TRANSITION)
        g = grad(o, z, grad_outputs=torch.ones(o.size()).to(z.device), create_graph=True)[0].view(z.size(0), -1)
        mixed_norm=g.norm(p=2, dim=1)
        gp = ((mixed_norm - 1) ** 2)
        self.writer.add_scalar('Loss/mixed_norm', float(mixed_norm.mean()), self.sche.cur_img)
        self.writer.add_scalar('Loss/mixed_scores', float(o.mean()), self.sche.cur_img)
        self.writer.add_scalar('Loss/Gradient_Penalty', float(gp.mean()), self.sche.cur_img)
        return gp

    def update_D(self,real_images,batch_size):
        self.G.zero_grad()
        # update discriminator.
        wgan_lambda = 1.0
        wgan_epsilon = 0.001
        fake_image_out = self.G(torch.randn(batch_size, 512),self.sche.lod,self.sche.phase,self.sche.TRANSITION)
        #print(fake_image_out[0].min(),fake_image_out[0].max())
        real_scores_out = self.D(real_images,self.sche.lod,self.sche.phase,self.sche.TRANSITION).squeeze()
        fake_scores_out = self.D(fake_image_out,self.sche.lod,self.sche.phase,self.sche.TRANSITION).squeeze()
        loss_d = fake_scores_out - real_scores_out
        loss_d += self.gradient_penalty(real_images, fake_image_out, self.D)*wgan_lambda
        epsilon_penalty = real_scores_out ** 2
        loss_d += epsilon_penalty * wgan_epsilon
        loss_d=loss_d.mean()
        self.opt_d.zero_grad()
        loss_d.backward()
        self.opt_d.step()
        self.writer.add_scalar('Loss/real_scores', real_scores_out.mean(), self.sche.cur_img)
        self.writer.add_scalar('Loss/epsilon_penalty', epsilon_penalty.mean(), self.sche.cur_img)
        return float(loss_d)

    def update_G(self,batch_size):
        # update generator.
        self.D.zero_grad()
        fake_scores_out = self.D(self.G(torch.randn(batch_size, 512),self.sche.lod,self.sche.phase,self.sche.TRANSITION),self.sche.lod,self.sche.phase,self.sche.TRANSITION)
        loss_g = -fake_scores_out.mean()
        self.writer.add_scalar('Loss/fake_scores', float(fake_scores_out.mean()), self.sche.cur_img)
        self.opt_g.zero_grad()
        loss_g.backward()
        self.opt_g.step()
        return float(loss_g)

    def renew(self):
        self.opt_g = Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=3e-3, betas=(0.0, 0.99))
        self.opt_d = Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=3e-3, betas=(0.0, 0.99))
        self.dataloader.update(self.sche.phase, self.sche.batchsize)
        self.sche.NEW_PHASE=False

    def save_model(self,phase):
        dir=os.path.join(self.tag_dir,"models")
        os.makedirs(dir,exist_ok=True)
        torch.save(self.G.state_dict(),os.path.join(dir,"phase_%d_G.pth"%phase))
        torch.save(self.D.state_dict(),os.path.join(dir,"phase_%d_D.pth"%phase))

    def train(self):
        while self.sche.cur_img<1000*self.total_kimg:
            real=self.dataloader.get_batch().cuda()
            real=propress_real(real,self.sche.lod,self.sche.phase)

            for repeat in range(self.D_repeat):
                loss_d=self.update_D(real,self.sche.batchsize)
                #self.Gs.update_Gs(self.G)
                self.sche.cur_img+=self.sche.batchsize
            loss_g=self.update_G(self.sche.batchsize)
            self.sche.cur_img+=self.sche.batchsize

            print("loss_d:%.2f loss_g:%.2f" % (loss_d, loss_g))

            self.writer.add_scalar('Loss/loss_d', loss_d, self.sche.cur_img)
            self.writer.add_scalar('Loss/loss_g', loss_g, self.sche.cur_img)


            if self.sche.tick%self.sche.save_kimg_tick.get(self.sche.phase,1)==0:
                with torch.no_grad():
                    img=self.G(self.z,self.sche.lod,self.sche.phase,self.sche.TRANSITION)
                    img=upscale_phase(img,self.sche.phase,self.sche.max_resolution)
                    os.makedirs(os.path.join(self.tag_dir,"images"),exist_ok=True)
                    save_image_grid(adjust_dynamic_range(img,[-1,1],[0,1]).clamp(0,1), os.path.join(self.tag_dir,"images", "phase%d_cur%d.jpg"%(self.sche.phase,self.sche.last_cur_phase_kimg)))
                    save_image_grid(adjust_dynamic_range(upscale_phase(real[:24],self.sche.phase,self.sche.max_resolution),[-1,1],[0,1]).clamp(0,1), os.path.join(self.tag_dir,"images", "phase%d_cur%d_real.jpg"%(self.sche.phase,self.sche.last_cur_phase_kimg)))


            self.sche.update()
            if self.sche.NEW_PHASE is True:
                self.save_model(self.sche.phase)
                self.G.Add_To_RGB_Layer(self.sche.phase)
                self.D.Add_From_RGB_Layer(self.sche.phase)
                self.renew()




if __name__=="__main__":
    torch.backends.cudnn.benchmark = True
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.manual_seed(1000)
    with torch.cuda.device(1):
        t=Train(total_kimg=18000)
        t.train()


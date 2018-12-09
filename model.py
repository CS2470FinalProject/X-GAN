from __future__ import division
import os
from glob import glob
import tensorflow as tf
import numpy as np
from collections import namedtuple

from module import *
from utils import *


class xgan:
    def __init__(self, sess, args):
        self.sess = sess
        self.batch_size = args.batch_size
        self.image_size = args.fine_size
        self.input_c_dim = args.input_nc
        self.output_c_dim = args.output_nc
        self.dann_weight = args.dann_weight
        self.sem_weight = args.sem_weight
        self.gan_weight = args.gan_weight
        self.dataset_dir = args.dataset_dir
        
        self.discriminator = discriminator
        self.encoder = encoder
        self.decoder = decoder
        self.cdann = cdann
        if args.use_lsgan:
            self.criterionGAN = mae_criterion
        else:
            self.criterionGAN = sce_criterion
        
        OPTIONS = namedtuple('OPTIONS', 'output_c_dim is_training ef_dim df_dim dcf_dim')
        self.options = OPTIONS._make((args.output_nc, args.phase == 'train', args.nef, args.ndf, args.ndcf))
        
        self._build_model()
        self.saver = tf.train.Saver()
        self.pool = ImagePool(args.max_size)
        
        
    def _build_model(self):
        self.real_data = tf.placeholder(tf.float32,
                                        [None, self.image_size, self.image_size,
                                         self.input_c_dim + self.output_c_dim],
                                         name='real_A_and_B_images')
        # Input: image of domain A and B
        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]
        
        # Generator
        # Encoder output
        self.embedding_A = self.encoder(self.real_A, self.options, domain_name="A")
        self.embedding_B = self.encoder(self.real_B, self.options, domain_name="B")
        
        # Reconstruction output
        # A->encoderA->decoderA
        # B->encoderB->decoderB
        self.reconstruct_A = self.decoder(self.embedding_A, self.options, domain_name="A")
        self.reconstruct_B = self.decoder(self.embedding_B, self.options, domain_name="B")
          
        # Cdann output
        self.cdann_A = self.cdann(self.embedding_A)
        self.cdann_B = self.cdann(self.embedding_B)
        
        # Generator output
        # B->encoderB->decoderA
        # A->encoderA->decoderB
        self.fake_A = self.decoder(self.embedding_B, self.options, domain_name="A")
        self.fake_B = self.decoder(self.embedding_A, self.options, domain_name="B")
        
        # Fake image encoder output
        self.embedding_fake_A = self.encoder(self.fake_A, self.options, domain_name="A")
        self.embedding_fake_B = self.encoder(self.fake_B, self.options, domain_name="B")
        
        # Discriminator output
        self.discriminate_fake_A = self.discriminator(self.fake_A, self.options, name="discriminator_A")
        self.discriminate_fake_B = self.discriminator(self.fake_B, self.options, name="discriminator_B")
        
        # Loss
        # Reconstruction loss
        self.rec_loss_A = euc_criterion(self.real_A, self.reconstruct_A)
        self.rec_loss_B = euc_criterion(self.real_B, self.reconstruct_B)
        self.rec_loss = self.rec_loss_A + self.rec_loss_B
        
        # Domain-adversarial loss
        self.dann_loss = sce_criterion(self.cdann_A, tf.zeros_like(self.cdann_A)) + sce_criterion(self.cdann_B, tf.ones_like(self.cdann_B))
        
        # Semantic consistency loss
        self.sem_loss_A = abs_criterion(self.embedding_A, self.embedding_fake_B)
        self.sem_loss_B = abs_criterion(self.embedding_B, self.embedding_fake_A)
        self.sem_loss = self.sem_loss_A + self.sem_loss_B
        
        # Gan loss-generator part
        self.gen_gan_loss_A = self.criterionGAN(self.discriminate_fake_A, tf.ones_like(self.discriminate_fake_A))
        self.gen_gan_loss_B = self.criterionGAN(self.discriminate_fake_B, tf.ones_like(self.discriminate_fake_B))
        self.gen_gan_loss =  self.gen_gan_loss_A + self.gen_gan_loss_B
        
        # Total loss
        self.gen_loss = self.rec_loss \
                        + self.dann_weight * self.dann_loss \
                        + self.sem_weight * self.sem_loss \
                        + self.gan_weight * self.gen_gan_loss
        
        # Discriminator
        self.fake_A_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.input_c_dim], name='fake_A_sample')
        self.fake_B_sample = tf.placeholder(tf.float32,
                                            [None, self.image_size, self.image_size,
                                             self.output_c_dim], name='fake_B_sample')
        # Discriminator output
        self.discriminate_real_A = self.discriminator(self.real_A, self.options, name="discriminator_A")
        self.discriminate_real_B = self.discriminator(self.real_B, self.options, name="discriminator_B")
        self.discriminate_fake_sample_A = self.discriminator(self.fake_A_sample, self.options, name="discriminator_A")
        self.discriminate_fake_sample_B = self.discriminator(self.fake_B_sample, self.options, name="discriminator_B")
        
        # Loss
        # Gan loss-discriminator part
        self.dis_gan_loss_real_A = self.criterionGAN(self.discriminate_real_A, tf.ones_like(self.discriminate_real_A))
        self.dis_gan_loss_fake_A = self.criterionGAN(self.discriminate_fake_sample_A, tf.zeros_like(self.discriminate_fake_sample_A))
        self.dis_gan_loss_A = (self.dis_gan_loss_real_A + self.dis_gan_loss_fake_A) / 2
        self.dis_gan_loss_real_B = self.criterionGAN(self.discriminate_real_B, tf.ones_like(self.discriminate_real_B))
        self.dis_gan_loss_fake_B = self.criterionGAN(self.discriminate_fake_sample_B, tf.zeros_like(self.discriminate_fake_sample_B))
        self.dis_gan_loss_B = (self.dis_gan_loss_real_B + self.dis_gan_loss_fake_B) / 2
        self.dis_gan_loss = self.dis_gan_loss_A + self.dis_gan_loss_B
        
        # Total loss
        self.dis_loss = self.gan_weight * self.dis_gan_loss
        
        self.rec_loss_sum = tf.summary.scalar("rec_loss", self.rec_loss)
        self.dann_loss_sum = tf.summary.scalar("dann_loss", self.dann_loss)
        self.sem_loss_sum = tf.summary.scalar("sem_loss", self.sem_loss)
        self.gen_gan_loss_sum = tf.summary.scalar("gen_gan_loss", self.gen_gan_loss)
        self.gen_loss_sum = tf.summary.scalar("gen_loss", self.gen_loss)
        self.gen_sum = tf.summary.merge([self.rec_loss_sum, self.dann_loss_sum, self.sem_loss_sum, self.gen_gan_loss_sum, self.gen_loss_sum])
        
        self.dis_gan_loss_sum = tf.summary.scalar("dis_gan_loss", self.dis_gan_loss)
        self.dis_loss_sum = tf.summary.scalar("dis_loss", self.dis_loss)
        self.dis_sum = tf.summary.merge([self.dis_gan_loss_sum, self.dis_loss_sum])
        
        self.test_A = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.input_c_dim], name='test_A')
        self.test_B = tf.placeholder(tf.float32,
                                     [None, self.image_size, self.image_size,
                                      self.output_c_dim], name='test_B')
        self.test_embedding_A = self.encoder(self.test_A, self.options, domain_name="A")
        self.test_embedding_B = self.encoder(self.test_B, self.options, domain_name="B")
        self.test_fake_A = self.decoder(self.test_embedding_B, self.options, domain_name="A")
        self.test_fake_B = self.decoder(self.test_embedding_A, self.options, domain_name="B")

        t_vars = tf.trainable_variables()
        self.gen_vars = [var for var in t_vars if 'encoder' in var.name or 'decoder' in var.name or 'cdann' in var.name]
        self.dis_vars = [var for var in t_vars if 'discriminator' in var.name]
        for var in t_vars: print(var.name)
            

    def train(self, args):
        self.lr = tf.placeholder(tf.float32, None, name='learning_rate')
        self.gen_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.gen_loss, var_list=self.gen_vars)
        self.dis_optim = tf.train.AdamOptimizer(self.lr, beta1=args.beta1).minimize(self.dis_loss, var_list=self.dis_vars)
        
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        
        counter = 1
        
        if args.continue_train:
            if self.load(args.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")
                
        for epoch in range(args.epoch):
            dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
            dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
            np.random.shuffle(dataA)
            np.random.shuffle(dataB)
            batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
            lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

            for idx in range(0, batch_idxs):
                batch_files = list(zip(dataA[idx * self.batch_size:(idx + 1) * self.batch_size],
                                       dataB[idx * self.batch_size:(idx + 1) * self.batch_size]))
                batch_images = [load_train_data(batch_file, args.load_size, args.fine_size) for batch_file in batch_files]
                batch_images = np.array(batch_images).astype(np.float32)

                # Update generator network and record fake outputs
                fake_A, fake_B, rec_loss, dann_loss, sem_loss, gen_gan_loss, gen_loss, _, summary_str = self.sess.run(
                    [self.fake_A, self.fake_B, self.rec_loss, self.dann_loss, self.sem_loss,
                     self.gen_gan_loss, self.gen_loss, self.gen_optim, self.gen_sum],
                    feed_dict={self.real_data: batch_images, self.lr: lr})
                self.writer.add_summary(summary_str, counter)
                [fake_A, fake_B] = self.pool([fake_A, fake_B])

                # Update discriminator network
                dis_loss, _, summary_str = self.sess.run(
                    [self.dis_loss, self.dis_optim, self.dis_sum],
                    feed_dict={self.real_data: batch_images,
                               self.fake_A_sample: fake_A,
                               self.fake_B_sample: fake_B,
                               self.lr: lr})
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] rec_loss: %4.3f|dann_loss: %4.3f|sem_loss: %4.3f|gen_gan_loss: %4.3f|gen_loss: %4.3f|dis_loss: %4.3f" % (
                    epoch, idx, batch_idxs, rec_loss, dann_loss, sem_loss, gen_gan_loss, gen_loss, dis_loss)))
                
                if np.mod(counter, args.print_freq) == 1:
                    self.sample_model(args.sample_dir, epoch, idx)

                if np.mod(counter, args.save_freq) == 2:
                    self.save(args.checkpoint_dir, counter)
     
    
    def save(self, checkpoint_dir, step):
        model_name = "xgan.model"
        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)
        
        
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        model_dir = "%s_%s" % (self.dataset_dir, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

        
    def sample_model(self, sample_dir, epoch, idx):
        dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)
        batch_files = list(zip(dataA[:self.batch_size], dataB[:self.batch_size]))
        sample_images = [load_train_data(batch_file, fine_size=64, is_testing=True) for batch_file in batch_files]
        sample_images = np.array(sample_images).astype(np.float32)

        fake_A, fake_B = self.sess.run(
            [self.fake_A, self.fake_B],
            feed_dict={self.real_data: sample_images}
        )
        save_images(fake_A, [self.batch_size, 1],
                    './{}/A_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))
        save_images(fake_B, [self.batch_size, 1],
                    './{}/B_{:02d}_{:04d}.jpg'.format(sample_dir, epoch, idx))

        
    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if args.which_direction == 'AtoB':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testA'))
        elif args.which_direction == 'BtoA':
            sample_files = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/testB'))
        else:
            raise Exception('--which_direction must be AtoB or BtoA')

        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # Write html for visual comparison
        index_path = os.path.join(args.test_dir, '{0}_index.html'.format(args.which_direction))
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        index.write("<th>name</th><th>input</th><th>output</th></tr>")

        out_var, in_var = (self.test_fake_B, self.test_A) if args.which_direction == 'AtoB' else (
            self.test_fake_A, self.test_B)

        for sample_file in sample_files:
            print('Processing image: ' + sample_file)
            sample_image = [load_test_data(sample_file, args.fine_size)]
            sample_image = np.array(sample_image).astype(np.float32)
            image_path = os.path.join(args.test_dir,
                                      '{0}_{1}'.format(args.which_direction, os.path.basename(sample_file)))
            fake_img = self.sess.run(out_var, feed_dict={in_var: sample_image})
            save_images(fake_img, [1, 1], image_path)
            index.write("<td>%s</td>" % os.path.basename(image_path))
            index.write("<td><img src='%s'></td>" % (sample_file if os.path.isabs(sample_file) else (
                '..' + os.path.sep + sample_file)))
            index.write("<td><img src='%s'></td>" % (image_path if os.path.isabs(image_path) else (
                '..' + os.path.sep + image_path)))
            index.write("</tr>")
        index.close()
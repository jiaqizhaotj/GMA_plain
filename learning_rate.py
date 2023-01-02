import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

class learning_rate():
    def __init__(self, boundaries, lr_values, lr_coefs, is_sgdr=False, type="", name = 'learning_rate'):
        self.name = name
        self.is_sgdr = is_sgdr
        self.type = type
        self.boundaries = boundaries
        self.lr_values = lr_values
        self.lr_coefs = lr_coefs
    
    def __call__(self, step, schedule_phase, initial_period_length=None, period_scale=None):
        #if(self.is_sgdr):
        lr = self.pytorch_compute_sgdr_lr(step, initial_period_length, period_scale, schedule_phase)
        # else:
        #     lr = self.compute_non_sgdr_lr(step)

        return lr

    # def compute_non_sgdr_lr(self, step, name = 'non_sgdr'):
    #     with tf.name_scope(name) as ns:
    #         #for only piecewise constant lr_coefs should be [[1]] in the config file.
    #         lr_coefs = tf.convert_to_tensor(self.lr_coefs, dtype=tf.float32, name = "convertion_to_tensor")
    #         learning_rate = tf.train.piecewise_constant(step, self.boundaries[:-1], self.lr_values, name = "PieceWise")
    #         tf.summary.scalar('initial_lr', learning_rate)
    #         coefs_len = tf.cast(tf.shape(self.lr_coefs, name = "reshape_line26")[0], tf.int64, name = "cast_line_26")
    #         residual = tf.math.mod(step, coefs_len, name= "modLine27")
    #         learning_rate =  lr_coefs[residual] * learning_rate
    #         tf.summary.scalar('coef', lr_coefs[residual])
            
    #         return learning_rate


    def pytorch_compute_sgdr_lr(self, step, initial_period_length, period_scale, schedule_phase):

        total_ite = self.boundaries[-1]
        initial_lr_max = self.lr_values[schedule_phase]
        lr_min = self.lr_coefs[-1] * self.lr_values[schedule_phase]
    
        if (period_scale == 1.0):
            relative_ite, curr_period_length = torch.tensor(torch.fmod(torch.tensor(step), initial_period_length)).float(), torch.tensor(initial_period_length).float()
        else:
            _step = step
            while(_step >= initial_period_length):
                _step = _step - initial_period_length
                initial_period_length = initial_period_length * period_scale
            relative_ite = _step
            curr_period_length = initial_period_length

        lr_max =  initial_lr_max 
        domain_scale = 1
        
        if self.type == 'reciprocal_scaled' or self.type == "rec_scaled":
            initial_domain = (lr_max // lr_min) - 1
            domain_scale = curr_period_length / torch.max(torch.tensor([initial_domain, 1]))
            learning_rate = (lr_max) * domain_scale * (1 / (relative_ite + 1 + (domain_scale - 1)))

        else:
            assert ValueError("sgdr type \"%s\" is unknown" % self.type)
        return learning_rate


    # def compute_sgdr_lr(self, step, initial_period_length, period_scale, lr_max_scale, schedule_phase, name = 'sgdr'):
    #     with tf.name_scope(name) as ns:

    #         total_ite = self.boundaries[-1]
    #         initial_lr_max = self.lr_values[schedule_phase]
    #         lr_min = self.lr_coefs[-1] * self.lr_values[schedule_phase]
    #         def compute_lr_scale_not1():
    #             nonlocal step, initial_period_length, period_scale
    #             def body(a, b):
    #                 return a - b, b * period_scale
    #             relative_ite, curr_period_length = tf.while_loop(lambda a, b: tf.math.greater_equal(a, b), body, [tf.cast(step, tf.float32), tf.cast(initial_period_length, tf.float32)])
    #             return relative_ite, curr_period_length

    #         def compute_lr_scale_1():
    #             nonlocal step, initial_period_length
    #             relative_ite, curr_period_length = tf.cast(tf.math.mod(step, initial_period_length), tf.float32), tf.cast(initial_period_length, tf.float32)
    #             return relative_ite, curr_period_length

    #         relative_ite, curr_period_length = tf.cond(tf.math.equal(period_scale, 1.0), compute_lr_scale_1, compute_lr_scale_not1)

    #         lr_max = ((1-(tf.cast(step, tf.float32) - relative_ite) / total_ite) * (lr_max_scale - 1) + 1) * (1.0 / lr_max_scale) * initial_lr_max 
    #         domain_scale = 1
    #         if self.type == 'cosine':
    #             pi = tf.constant(math.pi)
    #             learning_rate = lr_min + 0.5* (lr_max - lr_min) * (1 + tf.math.cos(relative_ite * pi / curr_period_length))
    #         elif self.type == 'reciprocal':
    #             learning_rate = lr_min + (lr_max - lr_min) * (1.0 /(relative_ite + 1))
    #         elif self.type == 'reciprocal_sqrt' or self.type == "sqrt":
    #             learning_rate = lr_min + (lr_max - lr_min) * tf.math.sqrt(1.0 /(relative_ite + 1))
    #         elif self.type == 'reciprocal_log':
    #             learning_rate = lr_min + (lr_max - lr_min) * (1.0 / (tf.math.log(relative_ite + 1) + 1))
    #         elif self.type == 'reciprocal_scaled' or self.type == "rec_scaled":
    #             initial_domain = lr_max // lr_min - 1
    #             domain_scale = curr_period_length / tf.math.maximum(initial_domain, 1)
    #             learning_rate = (lr_max) * domain_scale * (1 / (relative_ite + 1 + (domain_scale - 1)))

    #         else:
    #             print("""sgdr type "%s" is unknown""", self.type)
    #             print("the type of sgrd learning rate computation is unknown... Exiting....")
    #             exit(0)
    #         tf.summary.scalar('lr_max', initial_lr_max)
    #         tf.summary.scalar('lr_min', lr_min)
    #         tf.summary.scalar('scaled_lr_max', lr_max)
    #         tf.summary.scalar('relative_ite', relative_ite)
    #         tf.summary.scalar('current_period_length', curr_period_length)
    #         tf.summary.scalar('domain_scale', domain_scale)
    #         return learning_rate

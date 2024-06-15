from easydict import EasyDict as edict

options = edict()

options.target = 'psgn'
options.name = 'psgn-v4-mean5'
options.num_workers = 8

options.balance_gpu = False
options.gpu0_bs = 5
options.batch_list = [5, 8]

options.dataset = edict()
options.dataset.num_classes = 13

options.loss = edict()
options.loss.mean = 20.0
options.loss.emd = 1.0
options.loss.chamfer = 1.0
options.loss.chamfer_opposite = 0.55
options.loss.uniform = 0
options.loss.uniform_control = 4
options.loss.uniform_local = 1.0
options.loss.uniform_global = 1.0

options.optim = edict()
options.optim.adam_beta1 = 0.9
options.optim.lr = 3.0E-5
options.optim.wd = 1.0E-5
options.optim.lr_step = [10, 40]
options.optim.lr_factor = 0.1

options.train = edict()
options.train.num_epochs = 70
options.train.batch_size = 7
options.train.test_epochs = 5

options.test = edict()
options.test.view_fusion = True
options.test.batch_size = 7
options.test.weighted_mean = False

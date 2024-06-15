from easydict import EasyDict as edict

options = edict()

options.target = 'pcr'
options.name = 'pcr-v7.5'
options.num_workers = 8

options.balance_gpu = False
options.gpu0_bs = 5
options.batch_list = [2, 1, 1]
options.dataset = edict()
options.dataset.name = 'shapenet'
options.dataset.num_classes = 13

options.loss = edict()
options.loss.mean = 1.0
options.loss.emd = 1.0
options.loss.chamfer = 1.0
options.loss.chamfer_opposite = 0.55
options.loss.uniform = 0.1
options.loss.uniform_control = 4
options.loss.uniform_local = 1.0
options.loss.uniform_global = 1.0
options.loss.normal = 0.01
options.loss.normal_control = 4

options.optim = edict()
options.optim.adam_beta1 = 0.9
options.optim.lr = 5.0E-5
options.optim.wd = 1.0E-5
options.optim.lr_step = [5, 10, 15, 20]
options.optim.lr_factor = 0.3
options.optim.backward_steps = 4

options.train = edict()
options.train.num_epochs = 30
options.train.batch_size = 4
options.train.test_epochs = 5

options.test = edict()
options.test.view_fusion = True
options.test.batch_size = 1
options.test.weighted_mean = False

options.gcn = edict()
options.gcn.layer_num = 6
options.gcn.hidden_dim = 256
options.gcn.conv_k = 5
options.gcn.relative = False

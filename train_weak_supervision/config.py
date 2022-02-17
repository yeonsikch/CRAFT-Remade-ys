"""
Config containing all hardcoded parameters for training weak supervised model on datasets like icdar 2013
"""
from config import *

seed = 0

use_cuda = True
num_cuda = "0,1,2,3"

prob_synth = 1/6

check_path = 'Temporary/'

batch_size = {
	'train': 4*len(num_cuda.split(',')),
	'test': 8*len(num_cuda.split(',')),
}

num_workers = {
	'train': 16,
	'test': 16
}


lr = {
	0: 5e-5,
	1: 1e-5,
	2: 1e-5,
	3: 1e-5,
	4: 5e-6,
	5: 5e-6,
	6: 5e-6,
	7: 5e-6,
	8: 5e-6,
	9: 5e-6,
	10: 5e-7,
	11: 5e-7,
	12: 1e-7,
	13: 1e-7,
	14: 1e-7,
	15: 1e-7,
	16: 1e-7,
	17: 1e-7,
	18: 1e-7,
	19: 1e-7,
}

# lr = {
# 	0: 5e-5,
# 	1: 1e-5,
# 	2: 5e-6,
# 	3: 1e-6,
# 	4: 5e-7,
# 	5: 1e-7,
# 	6: 5e-8,
# 	7: 1e-8,
# 	8: 5e-9,
# 	9: 1e-9,
# }

optimizer_iterations = 1
iterations = batch_size['train']*12500*optimizer_iterations #12500
check_iterations = 5000
calc_f_score = 1000
change_lr = 1000
test_now = 1000

model_architecture = 'craft'

data_augmentation = {
	'crop_size': [[1024, 1024], [768, 768], [512, 512]],
	'crop_size_prob': [0.7, 0.2, 0.1],
}

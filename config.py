import math

seed = 0

THRESHOLD_POSITIVE = 0.1
THRESHOLD_NEGATIVE = 0

threshold_point = 25
window = 120

sigma = 18.5
sigma_aff = 20

boundary_character = math.exp(-1/2*(threshold_point**2)/(sigma**2))
boundary_affinity = math.exp(-1/2*(threshold_point**2)/(sigma_aff**2))

threshold_character = boundary_character + 0.03
threshold_affinity = boundary_affinity + 0.03

threshold_character_upper = boundary_character + 0.2
threshold_affinity_upper = boundary_affinity + 0.2

scale_character = math.sqrt(math.log(boundary_character)/math.log(threshold_character_upper))
scale_affinity = math.sqrt(math.log(boundary_affinity)/math.log(threshold_affinity_upper))

# dataset_name = 'hub_sign' #'ICDAR2013_ICDAR2017'
# test_dataset_name = 'hub_sign' #'ICDAR2013'
dataset_name = 'ICDAR13_17'
test_dataset_name = 'ICDAR13'
# dataset_name = 'ICDAR13'
# test_dataset_name = 'ICDAR13'

print(
	'Boundary character value = ', boundary_character,
	'| Threshold character value = ', threshold_character,
	'| Threshold character upper value = ', threshold_character_upper
)
print(
	'Boundary affinity value = ', boundary_affinity,
	'| Threshold affinity value = ', threshold_affinity,
	'| Threshold affinity upper value = ', threshold_affinity_upper
)
print('Scale character value = ', scale_character, '| Scale affinity value = ', scale_affinity)
print('Training Dataset = ', dataset_name, '| Testing Dataset = ', test_dataset_name)

DataLoaderSYNTH_base_path = '/data2/yeonsik/std/'+dataset_name
DataLoaderSYNTH_mat = '/data2/yeonsik/std/hub_sign/gt.mat'
DataLoaderSYNTH_Train_Synthesis = '/data2/yeonsik/std/SynthText'

DataLoader_Other_Synthesis = '/data2/yeonsik/std/'+dataset_name+'/Save/'
Other_Dataset_Path = '/data2/yeonsik/std/'+dataset_name # +'/Generated
save_path = '/data2/yeonsik/std/Models/WeakSupervision/'+dataset_name
images_path = '/data2/yeonsik/std/'+dataset_name+'/image'
target_path = '/data2/yeonsik/std/'+dataset_name+'/Generated'

Test_Dataset_Path = '/data2/yeonsik/std/'+test_dataset_name

threshold_word = 0.7
threshold_fscore = 0.5

dataset_pre_process = {
	'ICDAR13': {
		'train': {
			'target_json_path': '/data2/yeonsik/std/ICDAR13/train_gt',
			'target_folder_path': '/data2/yeonsik/std/ICDAR13/train',
		},
		'test': {
			'target_json_path': '/data2/yeonsik/std/ICDAR13/test_gt',
			'target_folder_path': '/data2/yeonsik/std/ICDAR13/test',
		}
	}
}

start_iteration = 0
skip_iterations = []

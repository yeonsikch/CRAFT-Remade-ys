import train_synth.config as config
from src.utils.data_manipulation import generate_target_others, denormalize_mean_variance
from train_synth.dataloader import DataLoaderEval
from src.utils.parallel import DataParallelModel
from src.utils.utils import generate_word_bbox, get_weighted_character_target, calculate_fscore, _init_fn

import cv2
import json
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = str(config.num_cuda)  # Specify which GPU you want to use


def synthesize(
		dataloader,
		model,save_txt,base_path_affinity, base_path_character, base_path_bbox, base_path_char, base_path_aff, base_path_json):

	"""

	Given a path to a set of images, and path to a pre-trained model, generate the character heatmap and affinity heatmap

	:param dataloader: A Pytorch dataloader for loading and resizing the images of the folder
	:param model: A pre-trained model
	:param base_path_affinity: Path where to store the predicted affinity heatmap
	:param base_path_character: Path where to store the predicted character heatmap
	:param base_path_bbox: Path where to store the word_bbox overlapped on images
	:param base_path_aff: Path where to store the predicted affinity bbox
	:param base_path_char: Path where to store the predicted character bbox
	:param base_path_json: Path where to store the predicted bbox in json format
	:return: None
	"""
	
	
	if save_txt:
		os.makedirs('./detection-results', exist_ok=True)
		os.makedirs('./normal_detection-results', exist_ok=True)
		os.makedirs('./tuned_detection-results', exist_ok=True)
	tuned_base_path_bbox = base_path_bbox[:-9]+'tuned_word_bbox'
	os.makedirs(tuned_base_path_bbox, exist_ok=True)
	print('Will generate the tuned word bbox at :', tuned_base_path_bbox)

	with torch.no_grad():
		
		model.eval()
		iterator = tqdm(dataloader)

		for no, (image, image_name, original_dim) in enumerate(iterator):

			if config.use_cuda:
				image = image.cuda()

			output = [x.cpu() for x in model(image)]

			if type(output) == list:

				# If using custom DataParallelModel this is necessary to convert the list to tensor
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			output[output < 0] = 0
			output[output > 1] = 1
			original_dim = original_dim.cpu().numpy()

			for i in range(output.shape[0]):

				# --------- Resizing it back to the original image size and saving it ----------- #

				image_i = denormalize_mean_variance(image[i].data.cpu().numpy().transpose(1, 2, 0))

				max_dim = original_dim[i].max()
				resizing_factor = 768/max_dim
				before_pad_dim = [int(original_dim[i][0]*resizing_factor), int(original_dim[i][1]*resizing_factor)]

				output[i, :, :, :] = np.uint8(output[i, :, :, :]*255)

				height_pad = (768 - before_pad_dim[0])//2
				width_pad = (768 - before_pad_dim[1])//2

				image_i = cv2.resize(
					image_i[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)

				character_bbox = cv2.resize(
					output[i, 0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)/255

				affinity_bbox = cv2.resize(
					output[i, 1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
					(original_dim[i][1], original_dim[i][0])
				)/255

				predicted_bbox = generate_word_bbox(
					character_bbox,
					affinity_bbox,
					character_threshold=config.threshold_character,
					affinity_threshold=config.threshold_affinity,
					word_threshold=config.threshold_word,
					character_threshold_upper=config.threshold_character_upper,
					affinity_threshold_upper=config.threshold_affinity_upper,
					scaling_character=config.scale_character,
					scaling_affinity=config.scale_affinity
				)

				predicted_bbox['tuned_word_bbox'] = tune_word_bbox(predicted_bbox['word_bbox'].tolist())
				tuned_word_bbox = predicted_bbox['tuned_word_bbox']
				word_bbox = predicted_bbox['word_bbox']
				char_bbox = np.concatenate(predicted_bbox['characters'], axis=0)
				aff_bbox = np.concatenate(predicted_bbox['affinity'], axis=0)

				tuned_word_image = image_i.copy()
				word_image = image_i.copy()
				char_image = image_i.copy()
				aff_image = image_i.copy()
				
				cv2.drawContours(tuned_word_image, tuned_word_bbox, -1, (0, 255, 0), 2)
				cv2.drawContours(word_image, word_bbox, -1, (0, 255, 0), 2)
				cv2.drawContours(char_image, char_bbox, -1, (0, 255, 0), 2)
				cv2.drawContours(aff_image, aff_bbox, -1, (0, 255, 0), 2)

				if save_txt:
					txt_name = os.path.join('./normal_detection-results', image_name[i][:-3]+'txt')
					if word_bbox.shape[0]==0:
						with open(txt_name, 'w') as f:
							f.write('')
					else:
						xmin = np.min(word_bbox[:,:,:,0], axis=1)
						xmax = np.max(word_bbox[:,:,:,0], axis=1)
						ymin = np.min(word_bbox[:,:,:,1], axis=1)
						ymax = np.max(word_bbox[:,:,:,1], axis=1)

						with open(txt_name, 'w') as f:
							for j in range(len(xmin)):
								f.write(' '.join(['###', '1', str(xmin[j][0]), str(ymin[j][0]), str(xmax[j][0]), str(ymax[j][0])])+'\n')

					txt_name = os.path.join('./tuned_detection-results', image_name[i][:-3]+'txt')
					if tuned_word_bbox.shape[0]==0:
						with open(txt_name, 'w') as f:
							f.write('')
					else:
						xmin = np.min(tuned_word_bbox[:,:,:,0], axis=1)
						xmax = np.max(tuned_word_bbox[:,:,:,0], axis=1)
						ymin = np.min(tuned_word_bbox[:,:,:,1], axis=1)
						ymax = np.max(tuned_word_bbox[:,:,:,1], axis=1)

						with open(txt_name, 'w') as f:
							for j in range(len(xmin)):
								f.write(' '.join(['###', '1', str(xmin[j][0]), str(ymin[j][0]), str(xmax[j][0]), str(ymax[j][0])])+'\n')

				plt.imsave(
					base_path_char + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					char_image)

				plt.imsave(
					base_path_aff + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					aff_image)

				plt.imsave(
					tuned_base_path_bbox + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					tuned_word_image)

				plt.imsave(
					base_path_bbox + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					word_image)

				plt.imsave(
					base_path_character + '/' + '.'.join(image_name[i].split('.')[:-1]) + '.png',
					np.float32(character_bbox > config.threshold_character),
					cmap='gray')

				plt.imsave(
					base_path_affinity+'/'+'.'.join(image_name[i].split('.')[:-1])+'.png',
					np.float32(affinity_bbox > config.threshold_affinity),
					cmap='gray')

				predicted_bbox['tuned_word_bbox'] = predicted_bbox['tuned_word_bbox'].tolist()
				predicted_bbox['word_bbox'] = predicted_bbox['word_bbox'].tolist()
				predicted_bbox['characters'] = [_.tolist() for _ in predicted_bbox['characters']]
				predicted_bbox['affinity'] = [_.tolist() for _ in predicted_bbox['affinity']]

				with open(base_path_json + '/' + '.'.join(image_name[i].split('.')[:-1])+'.json', 'w') as f:
					json.dump(predicted_bbox, f)

def remove_mini_box_under_big_box(word_bbox):
    remove_list = []
    for x in word_bbox:

        max_coord = np.max(np.array(x), axis=0)[0]
        min_coord = np.min(np.array(x), axis=0)[0]
        width, height = max_coord - min_coord

        for j, y in enumerate(word_bbox):
            if (min_coord[0] < y[0][0][0] and y[1][0][0] < max_coord[0]) and abs(y[0][0][1] - max_coord[1]) < height*0.2:
                remove_list.append(j)
                break
    word_bbox = np.array([word_bbox[i] for i, x in enumerate(word_bbox) if i not in remove_list])
    return word_bbox

def get_angle(p1, p2, direction="CW"):  #두점 사이의 각도: 시계 방향으로 계산한다. P1-(0,0)-P2의 각도를 시계방향으로
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    res = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    if direction == "CCW":    #반시계방향
        res = (360 - res) % 360
    return res

def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5

def remove_mini_box(word_bbox, threshold=0.5):
    area_list = []
    for x in word_bbox:

        max_coord = np.max(np.array(x), axis=0)[0]
        min_coord = np.min(np.array(x), axis=0)[0]
        width, height = max_coord - min_coord
        area_list.append(width*height)
    
    mean_area = np.mean(area_list)
    area_threshold = mean_area*threshold
    word_bbox = [word_bbox[i] for i, x in enumerate(area_list) if x > area_threshold]
    return word_bbox

def tune_word_bbox(word_bbox):
    remove_list = []
    for i, x in enumerate(word_bbox):
        if i in remove_list:
            continue

        right_top = x[1]
        right_bottom = x[2]

        max_coord = np.max(np.array(x), axis=0)[0]
        min_coord = np.min(np.array(x), axis=0)[0]
        width, height = max_coord - min_coord

        ratio = width/height
        if ratio < 1.3 and ratio > 0.7:

            remove_list.append(i)
            continue

        degree = get_angle(right_bottom[0], right_top[0])

        for j, y in enumerate(word_bbox):
            if (abs(right_top[0][0] - y[0][0][0]) < width*0.1 and abs(right_top[0][1] - y[0][0][1]) < height*0.2) \
                or ((abs(euclidean_distance(right_bottom[0], right_top[0]) - euclidean_distance(y[3][0], y[0][0])) < 10) and (euclidean_distance(right_top[0], y[0][0]) < 10) and degree>=5):

                xmin, ymin = np.min(np.array(x+y), axis=0).tolist()[0]
                xmax, ymax = np.max(np.array(x+y), axis=0).tolist()[0]

                new_left_top = [[xmin, ymin]]
                new_right_top = [[xmax, ymin]]
                new_right_bottom = [[xmax, ymax]]
                new_left_bottom = [[xmin, ymax]]
                
                new_box = [new_left_top, new_right_top, new_right_bottom, new_left_bottom]

                word_bbox[i] = new_box
                remove_list.append(j)
                break
	
    word_bbox = [x for i, x in enumerate(word_bbox) if i not in remove_list]
    word_bbox = remove_mini_box(word_bbox)
    word_bbox = np.array(remove_mini_box_under_big_box(word_bbox))
    
    return word_bbox

def generate_next_targets(original_dim, output, image, base_target_path, image_name, annots, dataloader, no):

	visualize = config.visualize_generated and no % config.visualize_freq == 0 and no != 0

	max_dim = original_dim.max()
	resizing_factor = 768 / max_dim
	before_pad_dim = [int(original_dim[0] * resizing_factor), int(original_dim[1] * resizing_factor)]

	output = np.uint8(output * 255)

	height_pad = (768 - before_pad_dim[0]) // 2
	width_pad = (768 - before_pad_dim[1]) // 2

	character_bbox = cv2.resize(
		output[0, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
		(original_dim[1]//2, original_dim[0]//2)) / 255

	affinity_bbox = cv2.resize(
		output[1, height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
		(original_dim[1]//2, original_dim[0]//2)) / 255

	# Generating word-bbox given character and affinity heatmap

	generated_targets = generate_word_bbox(
		character_bbox, affinity_bbox,
		character_threshold=config.threshold_character,
		affinity_threshold=config.threshold_affinity,
		word_threshold=config.threshold_word,
		character_threshold_upper=config.threshold_character_upper,
		affinity_threshold_upper=config.threshold_affinity_upper,
		scaling_character=config.scale_character,
		scaling_affinity=config.scale_affinity
	)

	generated_targets['word_bbox'] = generated_targets['word_bbox'] * 2
	generated_targets['characters'] = [i * 2 for i in generated_targets['characters']]
	generated_targets['affinity'] = [i * 2 for i in generated_targets['affinity']]

	if visualize:

		character_bbox = cv2.resize((character_bbox*255).astype(np.uint8), (original_dim[1], original_dim[0])) / 255

		affinity_bbox = cv2.resize((affinity_bbox*255).astype(np.uint8), (original_dim[1], original_dim[0])) / 255

		image_i = denormalize_mean_variance(image.data.cpu().numpy().transpose(1, 2, 0))

		image_i = cv2.resize(
			image_i[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
			(original_dim[1], original_dim[0])
		)

		# Saving affinity heat map
		plt.imsave(
			base_target_path + '_predicted/affinity/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			np.float32(affinity_bbox > config.threshold_affinity_upper),
			cmap='gray')

		# Saving character heat map
		plt.imsave(
			base_target_path + '_predicted/character/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			np.float32(character_bbox > config.threshold_character_upper), cmap='gray')

		cv2.drawContours(
			image_i,
			generated_targets['word_bbox'], -1,
			(0, 255, 0), 2)

		# Saving word bbox drawn on the original image
		plt.imsave(
			base_target_path + '_predicted/word_bbox/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			image_i)

	predicted_word_bbox = generated_targets['word_bbox'].copy()
	# --------------- PostProcessing for creating the targets for the next iteration ---------------- #

	generated_targets = get_weighted_character_target(
		generated_targets, {'bbox': annots['bbox'], 'text': annots['text']},
		dataloader.dataset.unknown,
		config.threshold_fscore,
		config.weight_threshold
	)

	target_word_bbox = generated_targets['word_bbox'].copy()

	f_score = calculate_fscore(
			predicted_word_bbox[:, :, 0, :],
			target_word_bbox[:, :, 0, :],
			text_target=annots['text'],
			unknown=dataloader.dataset.gt['unknown']
		)['f_score']

	if visualize:
		image_i = denormalize_mean_variance(image.data.cpu().numpy().transpose(1, 2, 0))
		image_i = cv2.resize(
			image_i[height_pad:height_pad + before_pad_dim[0], width_pad:width_pad + before_pad_dim[1]],
			(original_dim[1], original_dim[0])
		)

		# Generated word_bbox after postprocessing
		cv2.drawContours(
			image_i,
			generated_targets['word_bbox'], -1, (0, 255, 0), 2)

		# Saving word bbox after postprocessing
		plt.imsave(
			base_target_path + '_next_target/word_bbox/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			image_i)

		# Generate affinity heatmap after postprocessing
		affinity_target, affinity_weight_map = generate_target_others(
			(image_i.shape[0], image_i.shape[1]),
			generated_targets['affinity'].copy(),
			np.array(generated_targets['weights'])[:, 1])

		# Generate character heatmap after postprocessing
		character_target, characters_weight_map = generate_target_others(
			(image_i.shape[0], image_i.shape[1]),
			generated_targets['characters'].copy(),
			np.array(generated_targets['weights'])[:, 0])

		# Saving the affinity heatmap
		plt.imsave(
			base_target_path + '_next_target/affinity/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			affinity_target,
			cmap='gray')

		# Saving the character heatmap
		plt.imsave(
			base_target_path + '_next_target/character/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			character_target, cmap='gray')

		# Saving the affinity weight map
		plt.imsave(
			base_target_path + '_next_target/affinity_weight/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			affinity_weight_map,
			cmap='gray')

		# Saving the character weight map
		plt.imsave(
			base_target_path + '_next_target/character_weight/' + '.'.join(image_name.split('.')[:-1]) + '.png',
			characters_weight_map, cmap='gray')

	# Saving the target for next iteration in json format

	generated_targets['word_bbox'] = generated_targets['word_bbox'].tolist()
	generated_targets['characters'] = [word_i.tolist() for word_i in generated_targets['characters']]
	generated_targets['affinity'] = [word_i.tolist() for word_i in generated_targets['affinity']]

	with open(base_target_path + '/' + image_name + '.json', 'w') as f:
		json.dump(generated_targets, f)

	return f_score


def synthesize_with_score(dataloader, model, base_target_path):

	"""
	Given a path to a set of images(icdar 2013 dataset), and path to a pre-trained model, generate the character heatmap
	and affinity heatmap and a json of all the annotations
	:param dataloader: dataloader for icdar 2013 dataset
	:param model: pre-trained model
	:param base_target_path: path where to store the predictions
	:return:
	"""

	with torch.no_grad():

		model.eval()
		iterator = tqdm(dataloader)

		mean_f_score = []

		for no, (image, image_name, original_dim, item) in enumerate(iterator):

			annots = []

			for i in item:
				annot = dataloader.dataset.gt['annots'][dataloader.dataset.imnames[i]]
				annots.append(annot)

			if config.use_cuda:
				image = image.cuda()

			output = [x.cpu() for x in model(image)]

			if type(output) == list:
				# print([x.device for x in output])
				output = torch.cat(output, dim=0)

			output = output.data.cpu().numpy()
			output[output < 0] = 0
			output[output > 1] = 1
			original_dim = original_dim.cpu().numpy()

			f_score = []

			for i in range(output.shape[0]):

				f_score.append(
					generate_next_targets(
						original_dim[i],
						output[i],
						image[i],
						base_target_path,
						image_name[i],
						annots[i],
						dataloader,
						no
					)
				)

			mean_f_score.append(np.mean(f_score))

			iterator.set_description('F-score: ' + str(np.mean(mean_f_score)))


def main(
		folder_path,
		save_txt,
		base_path_character=None,
		base_path_affinity=None,
		base_path_bbox=None,
		base_path_char=None,
		base_path_aff=None,
		base_path_json=None,
		model_path=None,
		model=None,
):

	"""
	Entry function for synthesising character and affinity heatmap on images given in a folder using a pre-trained model
	:param folder_path: Path of folder where the images are
	:param base_path_character: Path where to store the character heatmap
	:param base_path_affinity: Path where to store the affinity heatmap
	:param base_path_char: Path where to store the image with character contours
	:param base_path_aff: Path where to store the image with affinity contours
	:param base_path_bbox: Path where to store the generated word_bbox overlapped on the image
	:param base_path_json: Path where to store the generated bbox in json format
	:param model_path: Path where the pre-trained model is stored
	:param model: If model is provided directly use it instead of loading it
	:return:
	"""

	if base_path_character is None:
		base_path_character = '/'.join(folder_path.split('/')[:-1])+'/character_heatmap'
	if base_path_affinity is None:
		base_path_affinity = '/'.join(folder_path.split('/')[:-1]) + '/affinity_heatmap'
	if base_path_bbox is None:
		base_path_bbox = '/'.join(folder_path.split('/')[:-1]) + '/word_bbox'
	if base_path_aff is None:
		base_path_aff = '/'.join(folder_path.split('/')[:-1])+'/affinity_bbox'
	if base_path_char is None:
		base_path_char = '/'.join(folder_path.split('/')[:-1]) + '/character_bbox'
	if base_path_json is None:
		base_path_json = '/'.join(folder_path.split('/')[:-1])+'/json_annotations'

	os.makedirs(base_path_affinity, exist_ok=True)
	os.makedirs(base_path_character, exist_ok=True)
	os.makedirs(base_path_aff, exist_ok=True)
	os.makedirs(base_path_char, exist_ok=True)
	os.makedirs(base_path_bbox, exist_ok=True)
	os.makedirs(base_path_json, exist_ok=True)

	# Dataloader to pre-process images given in the folder

	infer_dataloader = DataLoaderEval(folder_path)

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=config.batch_size['test'],
		shuffle=True, num_workers=config.num_workers['test'], worker_init_fn=_init_fn)

	if model is None:

		# If model has not been provided, loading it from the path provided

		if config.model_architecture == 'UNET_ResNet':
			from src.UNET_ResNet import UNetWithResnet50Encoder
			model = UNetWithResnet50Encoder()
		else:
			from src.craft_model import CRAFT
			model = CRAFT()
		model = DataParallelModel(model)

		if config.use_cuda:
			model = model.cuda()
			saved_model = torch.load(model_path)
		else:
			saved_model = torch.load(model_path, map_location='cpu')

		if 'state_dict' in saved_model.keys():
			model.load_state_dict(saved_model['state_dict'])
		else:
			model.load_state_dict(saved_model)

	synthesize(
		infer_dataloader,
		model, save_txt, base_path_affinity, base_path_character, base_path_bbox, base_path_char, base_path_aff, base_path_json)


def generator_(base_target_path, model_path=None, model=None):

	from train_weak_supervision.dataloader import DataLoaderEvalOther

	"""
	Generator function to generate weighted heat-maps for weak-supervision training
	:param base_target_path: Path where to store the generated annotations
	:param model_path: If model is not provided then load from model_path
	:param model: Pytorch Model can be directly provided ofr inference
	:return: None
	"""

	os.makedirs(base_target_path, exist_ok=True)

	# Storing Predicted

	os.makedirs(base_target_path + '_predicted/affinity', exist_ok=True)
	os.makedirs(base_target_path + '_predicted/character', exist_ok=True)
	os.makedirs(base_target_path + '_predicted/word_bbox', exist_ok=True)

	# Storing Targets for next iteration

	os.makedirs(base_target_path + '_next_target/affinity', exist_ok=True)
	os.makedirs(base_target_path + '_next_target/character', exist_ok=True)
	os.makedirs(base_target_path + '_next_target/affinity_weight', exist_ok=True)
	os.makedirs(base_target_path + '_next_target/character_weight', exist_ok=True)
	os.makedirs(base_target_path + '_next_target/word_bbox', exist_ok=True)

	# Dataloader to pre-process images given in the dataset and provide annotations to generate weight

	infer_dataloader = DataLoaderEvalOther('train')

	infer_dataloader = DataLoader(
		infer_dataloader, batch_size=config.batch_size['test'],
		shuffle=False, num_workers=config.num_workers['test'], worker_init_fn=_init_fn)

	if model is None:

		# If model has not been provided, loading it from the path provided

		if config.model_architecture == 'UNET_ResNet':
			from src.UNET_ResNet import UNetWithResnet50Encoder
			model = UNetWithResnet50Encoder()
		else:
			from src.craft_model import CRAFT
			model = CRAFT()

		model = DataParallelModel(model)

		if config.use_cuda:
			model = model.cuda()

		saved_model = torch.load(model_path)
		model.load_state_dict(saved_model['state_dict'])

	synthesize_with_score(infer_dataloader, model, base_target_path)

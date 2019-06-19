import string
import glob
import pickle as pickle
import common_functions as common_functions


def read_descriptions_file(document):
	image_description_mappings = dict()
	for line in document.split('\n'):
		tokens = line.split()
		if len(line) < 2:
			continue
		id, descriptions = tokens[0], tokens[1:]
		id = id.split('.')[0]
		descriptions = ' '.join(descriptions)
		if id not in image_description_mappings:
			image_description_mappings[id] = list()
		image_description_mappings[id].append(descriptions)
	return image_description_mappings


def clean_description_data(descriptions):
	# Replace all punctuations to empty character
	remove_punctuation = str.maketrans('', '', string.punctuation)
	for key, list_of_descriptions in descriptions.items():
		for i in range(len(list_of_descriptions)):
			description = list_of_descriptions[i]
			description = description.split()
			# convert all the words to lowercase
			description = [word.lower() for word in description]
			# Remove all punctuations
			description = [w.translate(remove_punctuation) for w in description]
			# Remove any word whose length is less than 1
			description = [word for word in description if len(word)>1]
			# Remove any word whose conetents are not alphabets
			description = [word for word in description if word.isalpha()]
			list_of_descriptions[i] = ' '.join(description)



# save the descriptions to a text file
def save_to_file(descriptions, filename):
	lines_in_file = list()
	for key, descriptions_list in descriptions.items():
		for desc in descriptions_list:
			lines_in_file.append(key + ' ' + desc)
	descriptions_data = '\n'.join(lines_in_file)
	with open(filename, 'w') as f:
		f.write(descriptions_data)


# load a pre-defined list of image file names
def load_set(filename):
	document = common_functions.read_document(filename)
	list_of_file_names = list()
	for line in document.split('\n'):
		if len(line) >= 1:
			# get the image name
			image_name = line.split('.')[0]
			list_of_file_names.append(image_name)
	return set(list_of_file_names)


# load the descriptions from file into memory
def load_clean_descriptions(filename, dataset):
	doc = common_functions.read_document(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		tokens_in_each_line = line.split()
		id, description = tokens_in_each_line[0], tokens_in_each_line[1:]
		if id in dataset:
			if id not in descriptions:
				descriptions[id] = list()
			description = 'captStrt ' + ' '.join(description) + ' captEnd'
			descriptions[id].append(description)
	return descriptions


# convert a dictionary of clean descriptions to a list of descriptions
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc


# calculate the length of the description with the most words
def get_max_length(descriptions):
	all_descriptions = list()
	for key in descriptions.keys():
		[all_descriptions.append(d) for d in descriptions[key]]
	return max(len(d.split()) for d in all_descriptions)


def pre_process_captions(class_name):
	if class_name == "general":
		filename = "Dataset/general_captions/Flickr8k.token.txt"
		document = common_functions.read_document(filename)
		print(document[:300])

		descriptions = read_descriptions_file(document)

		clean_description_data(descriptions)

		save_to_file(descriptions, 'Descriptions/' + class_name + '_' + 'descriptions.txt')

		filename = 'Dataset/general_captions/Flickr_8k.trainImages.txt'
		train = load_set(filename)
		print('Loading the training Dataset: %d' % len(train))

		images_path = 'Dataset/general_images/'
		list_of_images = glob.glob(images_path + '*.jpg')

		train_images_file = 'Dataset/general_captions/Flickr_8k.trainImages.txt'
		# Read the train image names in a set
		train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

		# this contains all the training images
		train_img = []

		for i in list_of_images:  # img is list of full path names of all images
			if i[len(images_path):] in train_images:  # Check if the image belongs to training set
				train_img.append(i)  # Add it to the list of train images

		# descriptions
		train_descriptions = load_clean_descriptions("Descriptions/"+class_name+'_descriptions.txt', train)
		print('Descriptions: train=%d' % len(train_descriptions))

		# Create a list of all the training captions
		training_captions = []
		for key, value in train_descriptions.items():
			for caption in value:
				training_captions.append(caption)
		len(training_captions)

		# Pick words that occur more than 10 times.
		word_count_threshold = 10
		word_counts = {}
		sentence_count = 0
		for sentence in training_captions:
			sentence_count += 1
			for w in sentence.split(' '):
				word_counts[w] = word_counts.get(w, 0) + 1

		vocabulary = [w for w in word_counts if word_counts[w] >= word_count_threshold]

		index_to_word = {}
		word_to_index = {}

		index = 1
		for w in vocabulary:
			word_to_index[w] = index
			index_to_word[index] = w
			index += 1

		vocab_size = len(index_to_word) + 1  # one for appended 0's

		max_length = get_max_length(train_descriptions)
		print('Max Description Length Length: ',  max_length)

		with open("Pickle/general_wordtoix.pkl", "wb") as encoded_pickle:
			pickle.dump(word_to_index, encoded_pickle)

		with open("Pickle/general_ixtoword.pkl", "wb") as encoded_pickle:
			pickle.dump(index_to_word, encoded_pickle)

		with open("Pickle/general_max_length.pkl", "wb") as encoded_pickle:
			pickle.dump(max_length, encoded_pickle)

		return max_length, vocab_size, train_descriptions, train_img, word_to_index






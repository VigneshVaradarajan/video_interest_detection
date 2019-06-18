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

# convert the loaded descriptions into a vocabulary of words
def to_vocabulary(descriptions):
	# build a list of all description strings
	all_desc = set()
	for key in descriptions.keys():
		[all_desc.update(d.split()) for d in descriptions[key]]
	return all_desc

# save descriptions to file, one per line
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

# load a pre-defined list of photo identifiers
def load_set(filename):
	doc = common_functions.read_document(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
	# load document
	doc = common_functions.read_document(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions




def pre_process_captions(class_name):
	if class_name=="general":
		filename = "Dataset/general_captions/Flickr8k.token.txt"
		doc = common_functions.read_document(filename)
		print(doc[:300])

		# parse descriptions
		descriptions = read_descriptions_file(doc)
		print('Loaded: %d ' % len(descriptions))# parse descriptions
		descriptions = read_descriptions_file(doc)
		print('Loaded: %d ' % len(descriptions))
		# clean descriptions
		clean_description_data(descriptions)

		# summarize vocabulary
		vocabulary = to_vocabulary(descriptions)
		print('Original Vocabulary Size: %d' % len(vocabulary))

		save_descriptions(descriptions, 'Descriptions/'+class_name+'_'+'descriptions.txt')

		filename = 'Dataset/general_captions/Flickr_8k.trainImages.txt'
		train = load_set(filename)
		print('Dataset: %d' % len(train))

		# Below path contains all the images
		images = 'Dataset/general_images/'
		# Create a list of all image names in the directory
		img = glob.glob(images + '*.jpg')

		# Below file conatains the names of images to be used in train data
		train_images_file = 'Dataset/general_captions/Flickr_8k.trainImages.txt'
		# Read the train image names in a set
		train_images = set(open(train_images_file, 'r').read().strip().split('\n'))

		# Create a list of all the training images with their full path names
		train_img = []

		for i in img:  # img is list of full path names of all images
			if i[len(images):] in train_images:  # Check if the image belongs to training set
				train_img.append(i)  # Add it to the list of train images

		# descriptions
		train_descriptions = load_clean_descriptions("Descriptions/"+class_name+'_descriptions.txt', train)
		print('Descriptions: train=%d' % len(train_descriptions))

		# Create a list of all the training captions
		all_train_captions = []
		for key, val in train_descriptions.items():
			for cap in val:
				all_train_captions.append(cap)
		len(all_train_captions)

		# Consider only words which occur at least 10 times in the corpus
		word_count_threshold = 10
		word_counts = {}
		nsents = 0
		for sent in all_train_captions:
			nsents += 1
			for w in sent.split(' '):
				word_counts[w] = word_counts.get(w, 0) + 1

		vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
		print('preprocessed words %d -> %d' % (len(word_counts), len(vocab)))

		ixtoword = {}
		wordtoix = {}

		ix = 1
		for w in vocab:
			wordtoix[w] = ix
			ixtoword[ix] = w
			ix += 1

		vocab_size = len(ixtoword) + 1  # one for appended 0's

		# convert a dictionary of clean descriptions to a list of descriptions
		def to_lines(descriptions):
			all_desc = list()
			for key in descriptions.keys():
				[all_desc.append(d) for d in descriptions[key]]
			return all_desc

		# calculate the length of the description with the most words
		def max_length(descriptions):
			lines = to_lines(descriptions)
			return max(len(d.split()) for d in lines)

		# determine the maximum sequence length
		max_length = max_length(train_descriptions)
		print('Description Length: %d' % max_length)

		with open("Pickle/general_wordtoix.pkl", "wb") as encoded_pickle:
			pickle.dump(wordtoix, encoded_pickle)

		with open("Pickle/general_ixtoword.pkl", "wb") as encoded_pickle:
			pickle.dump(ixtoword, encoded_pickle)

		with open("Pickle/general_max_length.pkl", "wb") as encoded_pickle:
			pickle.dump(max_length
						, encoded_pickle)


		return max_length, vocab_size, train_descriptions, train_img, wordtoix






import simplejson as json
from nltk.tokenize import word_tokenize
import unicodedata
from nltk.stem import WordNetLemmatizer
from collections import Counter
import glob,os
import re, math

lemmatizer = WordNetLemmatizer()

WORD = re.compile(r'\w+')

def get_cosine(vec1, vec2):  # Calculate cosine factor to find similarity
     intersection = set(vec1.keys()) & set(vec2.keys())
     numerator = sum([vec1[x] * vec2[x] for x in intersection])

     sum1 = sum([vec1[x]**2 for x in vec1.keys()])
     sum2 = sum([vec2[x]**2 for x in vec2.keys()])
     denominator = math.sqrt(sum1) * math.sqrt(sum2)

     if not denominator:
        return 0.0
     else:
        return float(numerator) / denominator

def text_to_vector(text):
     words = WORD.findall(text)
     return Counter(words)

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.

full_file_paths = get_filepaths("Candidate Profile Data")

profile={}


for f in full_file_paths:
	final=[]
	if f.endswith(".txt"):
		#print f
		json1_file = open(str(f))
		json1_str = json1_file.read()


		for json1_data in json.loads(json1_str):

			if isinstance(json1_data['Additional-Info'], str):
				temp = json1_data['Additional-Info']
			else:
				temp = unicodedata.normalize('NFKD', json1_data['Additional-Info']).encode('ascii','ignore')

			# print temp

			lexicon = []

			all_words = word_tokenize(temp)
			lexicon += list(all_words)

			# print lexicon

			temp=''
			li=[]
			# print lexicon
			for i in lexicon:
				# print i
				if i==',' or i=='.' or i=='\n' or i=='*' or i=='/' or i=='&' or i=='=========':
					li.append(temp)
					temp=''
				elif i==':' or i=='=':
					temp=''
				else:
					temp=temp+i

			if li==[]:
				lexicon = [lemmatizer.lemmatize(j) for j in lexicon]
			else:
				lexicon = [lemmatizer.lemmatize(j) for j in li]

			final.extend(lexicon)

		final = [i.lower() for i in final]
		profile[f]=Counter(final)

#print profile['Candidate Profile Data/System Architect.txt']


print "Enter your json formatted resume file path: ",
test_file = raw_input()

json1_file = open(test_file)
json1_str = json1_file.read()


for json1_data in json.loads(json1_str):

	# json1_data = json.loads(json1_str)[i]
	# print type(json1_data['Additional-Info'])
	if isinstance(json1_data['Additional-Info'], str):
		temp = json1_data['Additional-Info']
	else:
		temp = unicodedata.normalize('NFKD', json1_data['Additional-Info']).encode('ascii','ignore')

	# print temp

	lexicon = []

	all_words = word_tokenize(temp)
	lexicon += list(all_words)

	# print lexicon

	temp=''
	li=[]
	# print lexicon
	for i in lexicon:
		# print i
		if i==',' or i=='.' or i=='\n' or i=='*' or i=='/' or i=='&' or i=='=========':
			li.append(temp)
			temp=''
		elif i==':' or i=='=':
			temp=''
		else:
			temp=temp+i

	if li==[]:
		lexicon = [lemmatizer.lemmatize(j) for j in lexicon]
	else:
		lexicon = [lemmatizer.lemmatize(j) for j in li]


ans=[]
names=[]
for i in profile:
	cnt=Counter(profile[i])
	ans.append([0,i])
	# names.append(i)
	for j in lexicon:
		for k in profile[i]:
			if j.lower()==k:
				ans[-1][0]+=cnt[k]

ans.sort(key=lambda x:x[0], reverse=True)
print ans[:3]



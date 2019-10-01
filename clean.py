from collections import Counter

min_frequency = 3

orig_file = 'data/answers_raw.txt'
answers = [line.strip('\n').split() for line in open(orig_file) if line != ' ']

print('Number of answers: ' + str(len(answers)))

words = [word for line in answers for word in line]

old_counter = Counter(words)
new_counter = Counter()
ignore_words = set()

for key, val in old_counter.items():
	if val >= min_frequency:
		new_counter[key] = val
	else:
		ignore_words.add(key)

print('Number of distinct words: ' + str(len(old_counter)))
print('Number of distinct words after cutting those with less than '
	+str(min_frequency)+' appearances: ' + str(len(new_counter)))

answers = [line for line in answers if not any(word in ignore_words for word in line)]
print('Number of answers after cutting: ' + str(len(answers)))

word_to_num = {word : i for i,(word,_) in enumerate(new_counter.most_common())}

# token_file = 'data/tokens.txt'
# f = open(token_file, 'w+')
# for word, num in word_to_num.items():
# 	f.write(str(word) + ' ' + str(num) + ' \n')
# f.close()

tknzd_answers = []
for answer in answers:
	tknzd_answer = [word_to_num[word] for word in answer]
	tknzd_answers.append(tknzd_answer)

# target_file = 'data/answers.txt'
# f = open(target_file, 'w+')
# for tknzd_answer in tknzd_answers:
# 	for token in tknzd_answer: f.write(str(token)+' ')
# 	f.write('\n ')
# f.close()
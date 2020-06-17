import numpy as np
import re
'''This script processes the results of our model created in process_audio.py
    For possible future students looking to make their own model,
    you can input a trial number and avoid overwriting past results.
    Text files are of form "{Trial Num}_bla_bla.txt'''
# we'll plug in the argmax of the model's output into this dict to get our final guess'emotions = {}
emotions = {}
emotions[0] = 'sad'
emotions[1] = 'happy'
emotions[2] = 'neutral'
emotions[3] = 'angry'


predictions_file = open(f'3_Predictions.txt', 'r')
predicts = predictions_file.readlines()
predictions_file.close()
guesses = []
for line in predicts:
    #find argmax of each guess (0, 1, 2 or 3)
    line = line.strip(' ')
    line.replace("\n", '')
    line = line.split(' ')
    print(line)
    l = list(map(float, line))
    index, max_val = -1, -1
    for i in range(len(l)):
        if l[i] > max_val:
            index, max_val = i, l[i]
    guesses.append(emotions[index])

label_file = open(f'3_final_test_label.txt', 'r')
#final test labels are one string
big_string = label_file.readline()
l = re.split(r'\"|\'|,| |\n', big_string)
final_test_labels = []
for label in l:
    if label.isalpha() is True:
        final_test_labels.append(label)
label_file.close()

print(final_test_labels)
print(guesses)

text_for_excel = open(f'3_for_excel.txt', 'w+')
for i in range(len(guesses)):
    print(f'{final_test_labels[i]}\t{guesses[i]}', file=text_for_excel)
text_for_excel.close()

em  = {}
em['sad'] = []
em['happy'] = []
em['neutral'] = []
em['angry'] = []
foo = open('3_for_excel.txt', 'r')
i = 0
for line in foo:
    zed = line.split()
    em[zed[0]].append(zed[1])
for k in em.keys():
    print(k)
    happy, sad, angry, neutral = 0, 0, 0, 0
    length = len(em[k])
    for label in em[k]:
        if label == 'sad':
            sad += 1
        elif label == 'happy':
            happy += 1
        elif label == 'neutral':
            neutral += 1
        elif label == 'angry':
            angry += 1
    em[k] = sad, happy, neutral, angry
    print(em[k], '\n')


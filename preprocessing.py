import pickle
import csv
import numpy as np

csv.field_size_limit(1310720)

real_titles = []
with open('real.csv', encoding='utf8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[3] in row[2]:
            row[2] = row[2].replace(row[3], '')
        row[2] = row[2].replace("'", '')
        row[2] = row[2].replace(",", '')
        row[2] = row[2].replace('"', '')
        row[2] = row[2].replace(":", '')
        row[2] = row[2].replace("“", '')
        row[2] = row[2].replace("”", '')
        row[2] = row[2].replace("‘", '')
        row[2] = row[2].replace("’", '')
        row[2] = row[2].replace(".", '')
        row[2] = row[2].replace("(", '')
        row[2] = row[2].replace(")", '')
        row[2] = row[2].replace("-", ' ')
        real_titles.append(row[2])

fake_titles = []
with open('fake.csv', encoding='utf8') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        if row[6] == 'english':
            row[4] = row[4].replace("'", '')
            row[4] = row[4].replace(",", '')
            row[4] = row[4].replace('"', '')
            row[4] = row[4].replace(":", '')
            row[4] = row[4].replace("“", '')
            row[4] = row[4].replace("”", '')
            row[4] = row[4].replace("‘", '')
            row[4] = row[4].replace("’", '')
            row[4] = row[4].replace(".", '')
            row[4] = row[4].replace("(", '')
            row[4] = row[4].replace(")", '')
            row[4] = row[4].replace("-", ' ')
            fake_titles.append(row[4])

with open('real.pkl', 'wb') as f:
    pickle.dump(real_titles, f, protocol=pickle.HIGHEST_PROTOCOL)

with open('fake.pkl', 'wb') as f:
    pickle.dump(fake_titles, f, protocol=pickle.HIGHEST_PROTOCOL)

all = ''
for title in real_titles:
    split = title.split()
    for word in split:
        if word not in all:
            all += word + ' '
for title in fake_titles:
    split = title.split()
    for word in split:
        if word not in all:
            all += word + ' '

with open('glove100.txt', 'r', encoding='utf8') as f:
    embs = f.readlines()

embeddings = {}
for line in embs:
    split = line.split()
    word = split[0]
    if word in all:
        embedding = np.array([float(val) for val in split[1:]])
        embeddings[word] = embedding

with open('embs.pkl', 'wb') as f:
    pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)






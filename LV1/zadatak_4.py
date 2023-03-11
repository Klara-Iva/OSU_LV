fhand = open('song.txt')
dictionary_name = {}
for line in fhand:
    for word in line.split():
        if word in dictionary_name:
            dictionary_name[word] = dictionary_name[word]+1

        else:
            dictionary_name[word] = 1
counter = 0
for key in list(dictionary_name):
    if dictionary_name[key] == 1:
        counter = counter+1
        print(key, dictionary_name[key])
print('ukupno riječi koje se pojavljuju samo jednom: ', counter)

fhand.close()

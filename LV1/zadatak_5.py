file = open('SMSSpamCollection.txt', encoding="utf-8")
spam = 0
ham = 0
spamExclamation = 0
spamAvg = 0.0
hamAvg = 0.0
for line in file:
    line = line.rstrip()
    line = line.split()
    if line[0] == 'ham':
        ham += 1
        hamAvg += (len(line)-1)
    else:
        spam += 1
        spamAvg += (len(line)-1)
        if line[-1].endswith('!'):
            spamExclamation += 1
file.close()
print("Average number of ham SMS: ", hamAvg/ham)
print("Average number of spam SMS: ", spamAvg/spam)
print("Number of messages ending with exclamation: ", spamExclamation)

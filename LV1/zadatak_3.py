list = []


def prints(list):
    print()
    print('duzina liste: ', len(list))
    print('maximalna vrijednost liste: ', max(list))
    print('minimalna vrijednost liste: ', min(list))
    print('prosjek: ', sum(list)/len(list))
    print('sortirana lista: ', sorted(list))


def check_user_input():
    while True:
        inputs = input('unesi: ')
        try:
            val = int(inputs)
            list.append((float(inputs)))
        except ValueError:
            try:
                val = float(inputs)
                list.append(float(inputs))
            except ValueError:
                if (inputs == 'Done'):
                    prints(list)
                    break
                print("No.. input is not a number. It's a string. Try again....")


check_user_input()

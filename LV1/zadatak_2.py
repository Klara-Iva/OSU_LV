def Grade(number):
    if number >= 0.9:
        print('A')
    elif number >= 0.8:
        print('B')
    elif number >= 0.7:
        print('C')
    elif number >= 0.6:
        print('D')
    else:
        print('F')


def main():
    while True:
        try:
            number = float(input('Enter a number: '))
            if number > 1.0 or number < 0.0:
                print('Broj je izvan granica. Pokušajte ponovno...')

            else:
                Grade(number)
                break
        except ValueError:
            print("Oops! To nije broj.Pokušajte ponovno.... ")


main()

def calculatePay( hours, pay):
    return hours*pay
workingHours= int(input('Radni sati: '))
payPerHour= float(input('eura/h '))
print('Ukupno: ' ,calculatePay(workingHours,payPerHour),'eura' )

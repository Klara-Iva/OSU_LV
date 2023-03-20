import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

# a)
plt.figure()
data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
plt.title = 'vrijednost CO2 emisija'
plt.xlabel('CO2 emissions (g/km)')
plt.ylabel('Frequency')


# b)
plt . figure()
plt.xlabel('CO2 emissions (g/km)')
plt.ylabel('Fuel Consumption City (L/100km)')
# postoji .map funckija i mozes i nju iskoristit
color = ['g', 'r', 'b', 'y', 'k']
gas = ['X', 'Z', 'D', 'E', 'N']
for i in range(0, 5, 1):
    data1 = data[data['Fuel Type'] == gas[i]]
    x = np.array(data1['CO2 Emissions (g/km)'])
    y = np.array(data1['Fuel Consumption City (L/100km)'])
    plt.scatter(x, y, c=color[i], s=5)


plt.legend(["Regular gasoline", "Premium gasoline",
           "Diesel", "Ethanol (E85)", "Natural gas"])

# c)

grouped = data.groupby('Fuel Type')
data.boxplot(column=[
    'Fuel Consumption Hwy (L/100km)'], by='Fuel Type')


# d)
plt.figure()
data2 = data.groupby('Fuel Type').size()
data2.plot.bar(x='vrste goriva', y='broj auta', rot=0)

# e)
plt.figure()
data3 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
data3.plot.bar(x='broj cilindara', y='CO2 emisije', rot=0)
plt.legend()


plt.show()

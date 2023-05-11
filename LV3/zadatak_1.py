import pandas as pd
import numpy as np

data = pd.read_csv('data_C02_emission.csv')

# konvertiranje kategorickih velicina u tip category
data['Fuel Type'] = data['Fuel Type'].astype("category")
data['Transmission'] = data['Transmission'].astype("category")
data['Vehicle Class'] = data['Vehicle Class'].astype("category")
data['Model'] = data['Model'].astype("category")
data['Make'] = data['Make'].astype("category")
# a)
print('mjerenje sadrži: ', len(data))
print(data.info())
print(data.describe())
data.dropna(axis=0)
data.drop_duplicates()
data = data.reset_index(drop=True)

# b)
print()
print()

print(data[['Make', 'Model', 'Fuel Consumption City (L/100km)']
           ].iloc[data['Fuel Consumption City (L/100km)'].argsort().head(3)])
print(data[['Make', 'Model', 'Fuel Consumption City (L/100km)']
           ].iloc[data['Fuel Consumption City (L/100km)'].argsort().tail(3)])

# c)
print()
print()


data2 = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print('Motora izmedu 2.5 i 3.5: ', len(data2),
      'i njihova prosjecna CO2 emisija: ', data2['CO2 Emissions (g/km)'].mean())

# d)
print()
print()

data3 = data[(data['Make'] == 'Audi')]
print('Broj zapisa marke Audi: ', len(data3))
data4 = data3[data3['Cylinders'] == 4]
print('prosjecna emisija C02 plinova automobila proizvodaca Audi koji imaju 4 cilindara: ',
      data4['CO2 Emissions (g/km)'].mean())


print()
print()


# e)
cy = data[(data['Cylinders'] >= 4) & (data['Cylinders'] % 2 == 0)]
em_per_cy = cy.groupby(['Cylinders']).agg({'CO2 Emissions (g/km)': 'mean'})
print('CO2 emisija vozila s 2,4,6... cilindara...')
print(em_per_cy)


# f)
print()
print()
data6 = data[data['Fuel Type'] == 'D']
data7 = data[data['Fuel Type'] == 'X']
print('Diesel prosjecna grgadska potrosnja: prosjek je: ', data6['Fuel Consumption City (L/100km)'].mean(), 'median je: ',
      data6['Fuel Consumption City (L/100km)'].median())
print('Regular gasoline prosjecna gradska potrosnja: prosjek je:', data7['Fuel Consumption City (L/100km)'].mean(), 'median je: ',
      data7['Fuel Consumption City (L/100km)'].median())
print()
print()
# g)
data8 = data[(data['Fuel Type'] == 'D') & (data['Cylinders'] == 4)]
print('Vozilo koje ima dizel motor i 4 cilindra najvecom gradskom potrosnjom je: ')
print(data8.iloc[data8['Fuel Consumption City (L/100km)'].argsort().head(1)])

print()
print()

# h)
print('Broj vozila s ručnim mjenjacem: ', len(
    data[data['Transmission'].str.contains(pat='M[0-9]', regex=True)]))

print()
print()
# i) korelacija
print(data.corr(numeric_only=True))

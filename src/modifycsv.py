import pandas as pd

df = pd.read_csv('train/images.csv')
num_classes = len(pd.unique(df['label'])) 
numbers_to_classes = {}
classes_to_numbers = {}

for n, plankton_cat in enumerate(df['label'].unique()):
  numbers_to_classes[n] = plankton_cat
  classes_to_numbers[plankton_cat] = n

list = ['image']
for n, values in enumerate(numbers_to_classes.values()):
	list.append(values)

df2 = pd.read_csv('checkpoint0.csv', names=list)
df2.to_csv('checkpoint0_mod.csv', index=False)
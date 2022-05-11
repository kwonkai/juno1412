# test.py
from mod import to_str, dict1, dict2, json_ch, json_ch1, json_ch2, json_change
from doctest import DocFileSuite

from pandas import DataFrame
import pandas as pd

info = [ [ 'name', 1, 2, 3, 4  ], [ 'age', 10, 20, 30, 40 ], [ 'address', 100, 200, 300, 400 ] ]
df = pd.DataFrame( info, columns = ['type', 'list'], index=False )

f = [213, 512, 564, 214, 15, 48]
fr = pd.DataFrame(f, columns=['name'])
fr = fr.transpose()
# fr.to_csv('fr.csv')
# fr_data = pd.read_csv('C:/Users/kwonk/juno1412-1/juno1412/예제/fr.csv')
fr
for i in range(5): 
    i = i+1 
    fr[i]





df.to_csv('test_data.csv')

data = pd.read_csv('C:/Users/kwonk/juno1412-1/juno1412/예제/test_data.csv')
data.head(10)
data['type']
data['list']



# 2번째 코드

list = data['type', 'list']
json_change('name', list)

list = data['type', 'list']
json_change('age', list)

list = data['type', 'list']
json_change('address', list)






# 최종코드
list = pd.read_csv('C:/Users/kwonk/juno1412-1/juno1412/예제/test_data.csv')
for i in range(3):
    type = ['name', 'age', 'address']
    list = data['list']
    json_ch(type[i], list)
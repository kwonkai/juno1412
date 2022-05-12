# mod.py
import json
import test

## 1번째 코드
# list -> str
def to_str(type,list):
    if type == 'name':
        string = ','.join(map(list))
    elif type == 'age':
        string = ','.join(map(list))
    elif type == 'address':
        string = ','.join(map(list))
    return string

# str -> dictionary
def dict1(type,string):
    dict1 = {'type' : type, 'id' : string}
    return dict1

def dict2(type,string):
    dict2 = {'type' : type, 'id' : string}
    return dict2

# dictionary -> json
def json_ch1(dict1):
    json_ch1 = json.dumps([dict1], indent ='\n')
    return json_ch1

def json_ch2(dict2):
    json_ch2 = json.dumps([dict2], indent ='\n')
    return json_ch2

# json_merge
json_plus = json.dumps([json_ch1, json_ch2], indent='\n')



# 2번째 코드
def json_change(type, list):
    string = ','.join(map(str,list))
    tmp_dic = {'type' : type, 'id' : string}
    if type == 'name':
        tmp_dic = {'type' : type, 'id' : string}
    elif type == 'age':
        tmp_dic = {'type' : type, 'id' : string}
    elif type == 'address':
        tmp_dic = {'type' : type, 'id' : string}
    json_tmp = json.dump(tmp_dic)
    print(json_tmp)
    return json_tmp

# 3번째 최종 코드 list -> json
def json_ch(type, list):
    string = ','.join(map(list))
    tmp_dic = {'type' : type, 'id' : string}
    json_ch = json.dumps(tmp_dic, indent='/n')
    return json_ch



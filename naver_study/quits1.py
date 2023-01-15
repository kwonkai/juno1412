# 주어진 리스트
num_list = [1, 5, 7, 15, 16, 22, 28, 29]

def get_odd_num(num_list):
    #TODO
    odd_num =[]
    for i in range(len(num_list)):
        if num_list[i] % 2 == 1:
            odd_num.append(num_list[i])
            
    return odd_num

print(get_odd_num(num_list))
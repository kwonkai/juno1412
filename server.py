def change(type, content_id_list, id):
        import json
        type == 'title'
        content_id_list = getRecommandedContents('title', id) # title id 가져오기
        string = ','.join(map(str,content_id_list)) # int -> str 변환
        dic1 = { 'type' : 'title', 'id' : [string] } # key : values

        type == 'learningArea'
        content_id_list = getRecommandedContents(type, id)
        string = ','.join(map(str,content_id_list))
        dic2 = { 'type' : type, 'id' : [string] }

        type == 'level'
        content_id_list = getRecommandedContents(type, id)
        string = ','.join(map(str,content_id_list))
        dic3 = { 'type' : type, 'id' : [string] }

        type == 'category'
        content_id_list = getRecommandedContents(type, id)
        string= ','.join(map(str,content_id_list))
        dic4 = { 'type' : type, 'id' : [string]}

        type == 'genre'
        content_id_list = getRecommandedContents(type, id)
        string = ','.join(map(str,content_id_list))
        dic5 = { 'type' : type, 'id' : [string] }

        type == 'cast'
        content_id_list = getRecommandedContents(type, id)
        string = ','.join(map(str,content_id_list))
        dic6 = { 'type' : type,  'id' : [string] }
    
        json.dumps([dic1,dic2,dic3,dic4,dic5,dic6], indent="\t")
        # dic 1~6 밖으로 뽑아내
        # 외부 dic 1~6 설정 = dic 저장

def change(type, content_id_list, id):
        import json
        dic = {key: value for key, value in dict.formkeys(type).item(id)}
        type = ['title', 'learningArea', 'level', 'category','genre','cast']
        if type == 'title':
            content_id_list == getRecommandedContents('title', id): # title id 가져오기
            string1 = ','.join(map(str,content_id_list)) # int -> str 변환
            dic1 = { 'type' : 'title', 'id' : [string1] } # key : values
            # json_str = json.dumps(dic).copy() # json 형식으로 변환


        elif type == 'learningArea':
            content_id_list = getRecommandedContents(type, id)
            string2 = ','.join(map(str,content_id_list))
            dic2 = { 'type' : type, 'id' : [string2] }
            # json_str = json.dumps(dic).copy()

        
        # dic 1~6 밖으로 뽑아내
        # 외부 dic 1~6 설정 = dic 저장

        
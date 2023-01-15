# 주어진 리스트
score = [(100,100), (95, 90), (55, 60), (75, 80), (70, 70)]

def get_avg(score):
    # score list 내부 데이터 개수 i만큼 반복
    for i in range(len(score)):
        # 빈 list 생성
        score_sum=[]
        # 내부 튜플의 데이터 개수 j만큼 반복
        # 빈 list = score_sum에 추가
        for j in range(len(score[i])):
            score_sum.append(score[i][j])
        
        # i번 평균, 출력
        print(i+1,"번, 평균 :",  sum(score_sum) / len(score_sum))
    return

print(get_avg(score))
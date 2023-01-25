# test score, mid : 50, final : 75

from re import A
from tkinter import N


class Score():
    def __init__(self, mid, final):
        self.mid = mid;
        self.final = final;


    def score_int(self, mid, final):
        self.mid = mid
        self.final = final
        return (mid+final) / 2
         

# 출력함수
score = Score(50, 75)
print((score.mid + score.final) / 2)
from numpy import character
import pygame

pygame.init() # 초기화 (반드시 필요)

# 화면크기 설정
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))

# 화면 타이틀 설정
pygame.display.set_caption("MiniGame")

# 화면배경 설정
background = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/background.jpg")

# 캐릭터(주인공) 불러오기
character = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/character.jpg")
character_size = character.get_rect().size # 이미지 크기 설정
character_width = character_size[0] # 캐릭터 가로
character_height = character_size[1] # 캐릭터 세로
character_x_pos = (screen_width / 2) - (character_width / 2) # 캐릭터 위치(가로)
character_y_pos = screen_height - character_height #캐릭터 위치(세로)

# 이벤트 종료
running = True # 게임이 진행중인가?
while running:
    for event in pygame.event.get(): # 어떤 이벤트가 발생하였는지 체크
        if event.type == pygame.QUIT: # 창이 닫히는 이벤트가 발생하였는가?
            running = False # 게임 진행중이 아님

    # screen.fill((239, 224, 176))
    screen.blit(background, (0, 0)) # 배경그리기

    screen.blit(character, (character_x_pos, character_y_pos)) # 캐릭터 그리기

    pygame.display.update() # 게임화면 다시 그리기



# pygame 종료
pygame.quit()
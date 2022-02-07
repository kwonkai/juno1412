from tkinter import RIGHT
import pygame
import random
#######################################3
# 기본 초기화
pygame.init() 

# 화면크기 설정
screen_width = 400
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# 화면 타이틀 설정
pygame.display.set_caption("똥피하기")

# FPS 설정
clock = pygame.time.Clock()

#################################################################
# 1. 사용자 게임 초기화(배경화면, 게임 이미지, 좌표, 속도, 폰트 등 설정)
# 1-1. 배경만들기
background = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/미니게임/총알피하기/background2.jpg")

# 1-2. 캐릭터만들기(사람, 총알)
human = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/미니게임/총알피하기/human.jpg")
human_size = human.get_rect().size
human_width = human_size[0]
human_height = human_size[1]
human_x_pos = (screen_width / 2) - (human_width /2)
human_y_pos = screen_height - human_height
human_speed = 7
to_x = 0


bullet = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/juno1412-1/미니게임/총알피하기/bullet-gcf.jpg")
bullet_size = bullet.get_rect().size
bullet_width = bullet_size[0]
bullet_height = bullet_size[1]
bullet_x_pos = random.randint(0, screen_width - bullet_width)
bullet_y_pos = 0
bullet_speed = 7
to_x = 0

running = True 
while running:
    dt = clock.tick(60) 

    # 2. 이벤트 처리(키보드, 마우스 등)
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                to_x -= human_speed
            elif event.key == pygame.K_RIGHT:
                to_x += human_speed

        if event.type == pygame.KEYUP:
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                to_x = 0
    # 3. 게임 캐릭터 위치 정의
    
    human_x_pos += to_x
    
    if human_x_pos < 0:
        human_x_pos = 0
    elif human_x_pos > screen_width - human_width:
        human_x_pos = screen_width - human_width

    bullet_y_pos += bullet_speed

    if bullet_y_pos > screen_height:
        bullet_y_pos = 0
        bullet_x_pos = random.randint(0, screen_width - human_width)
    # 4. 충돌 처리

    human_rect = human.get_rect()
    human_rect.left = human_x_pos
    human_rect.top = human_y_pos

    bullet_rect = bullet.get_rect()
    bullet_rect.left = bullet_x_pos
    bullet_rect.top = bullet_y_pos

    if human_rect.colliderect(bullet_rect):
        print("충돌했어요")
        running = False

    # 5. 화면에 그리기
    screen.blit(background, (0,0))
    screen.blit(human, (human_x_pos, human_y_pos))
    screen.blit(bullet, (bullet_x_pos, bullet_y_pos))

    pygame.display.update() # 게임화면 다시 그리기
    
# 잠시 대기(2초)
pygame.time.delay(1000) 
# pygame 종료
pygame.quit()
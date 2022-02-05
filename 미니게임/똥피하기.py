import pygame
import random
import os
#######################################3
# 기본 초기화
pygame.init() 

# 화면크기 설정
WHITE = (255, 255, 255)
screen_size = [400, 600]
screen = pygame.display.set_mode(screen_size)

# 화면 타이틀 설정
pygame.display.set_caption("총알피하기")

# 실행여부
running = False

# FPS 설정
clock = pygame.time.Clock()

#################################################################
# 1. 사용자 게임 초기화(배경화면, 게임 이미지, 좌표, 속도, 폰트 등 설정)
# 1-1. 배경만들기
background = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/background2.jpg")

# 1-2. 캐릭터와 장애물 만들기
def runGame():
    bullet = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/bullet-gcf.png")
    bullet = pygame.transform.scale(bullet, (40, 60))
    bullets = []

    for i in range(5):
        bullet_rect = pygame.Rect(bullet.get_rect())
        bullet_rect.left = random.randint(0, screen[0])
        bullet_rect.top = -100
        dy = random.randint(3,9) # 총알 속도
        bullets.append({'rect' : bullet_rect, 'dy' : dy})

    character_image = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/human.png")
    character_image = pygame.transform.scale(character_image, (60, 60))
    character = pygame.Rect(character_image.get_rect())
    character.left = screen_size[0] // 2 - character.width // 2
    character.top = screen_size[1] - character.height
    character_x_pos = 0
    character_y_pos = 0

    global running
    while not running:
        clock.tick(60)
        screen.fill(WHITE)

    # 2. 이벤트 처리(키보드, 마우스 등)
        for event in pygame.event.get(): 
            if event.type == pygame.QUIT:
                    running = True
                    break
            elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        character_x_pos = -5
                    elif event.key == pygame.K_RIGHT:
                        character_x_pos = 5
                
            elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        character_x_pos = 0
                    elif event.key == pygame.K_RIGHT:
                        character_x_pos = 0

    for bullet in bullets:
        bullet['bullet_rect'].top += bullet['dy'] 
        if bullet['bullet_rect'].top > screen_size[1]:
            bullets.remove(bullet)
            rect = pygame.Rect(bullet.get_rect())
            rect.left = random.randint(0, screen_size[0])
            rect.top = -100
            bullet_unit = random.randint(3,9)
            bullets.append({'rect' : bullet_rect, 'unit' : bullet_unit})
    

    # 3. 게임 캐릭터 위치 정의

    if character_x_pos < 0:
        character_x_pos = 0
    elif character_x_pos > screen_size[0]- character.width:
        character_x_pos = screen_size[0] - character.width

    for bullet in bullets: 
        if bullet['rect'].colliderect(character):
            running = False

    # # 4. 충돌 처리
    # character_rect = character.get_rect()
    # character_rect.left = character_x_pos
    # character_rect.top = character_y_pos

    # dong_rect = dong.get_rect()
    # dong_rect.left = dong_x_pos
    # dong_rect.top = dong_y_pos

    # if character_rect.colliderect(dong_rect):
    #     print("gameover")
    #     running = False

    # 5. 화면에 그리기
    screen.blit(background, (0,0))
    screen.blit(character, (character_x_pos, character_y_pos))
    screen.blit(bullet, (character_image, character))


    pygame.display.update() # 게임화면 다시 그리기
    
# 잠시 대기(2초)
pygame.time.delay(1000) 
# pygame 종료
pygame.quit()
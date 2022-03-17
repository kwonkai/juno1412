import os
from numpy import char
import pygame
#######################################3
# 기본 초기화
pygame.init() 

# 화면크기 설정
screen_width = 640
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))

# 화면 타이틀 설정
pygame.display.set_caption("Ball Jump")

# FPS 설정
clock = pygame.time.Clock()

#################################################################
# 1. 사용자 게임 초기화(배경화면, 게임 이미지, 좌표, 속도, 폰트 등 설정)
current_path = os.path.dirname(__file__) # 현재파일위치 반환
image_path = os.path.join(current_path, "images") # images 폴더에 위치 반환

# 배경
background = pygame.image.load(os.path.join(image_path, "background.png"))

# 스테이지
stage = pygame.image.load(os.path.join(image_path, "bar.png"))
stage_size = stage.get_rect().size
stage_height = stage_size[1] # 스테이지 높이 위에 캐릭터 놔두기

# 캐릭터
character = pygame.image.load(os.path.join(image_path, "man.png"))
character_size = character.get_rect().size
character_width = character_size[0]
character_height = character_size[1]
character_x_pos = (screen_width / 2) - (character_width / 2 )
character_y_pos = screen_height - character_height - stage_height

# 캐릭터 이동방향
character_to_x = 0

# 캐릭터 이동 속도
character_speed = 3

# 무기만들기
weapon = pygame.image.load(os.path.join(image_path, "weapon.png"))
weapon_size = weapon.get_rect().size
weapon_width = weapon_size[0]

# 무기는 한 번에 여러발 발사 가능으로 설정
weapons = []

# 무기 이동속도
weapon_speed = 10


running = True 
while running:
    dt = clock.tick(30) 

    # 2. 이벤트 처리(키보드, 마우스 등)
    for event in pygame.event.get(): 
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                character_to_x -= character_speed
            elif event.key == pygame.K_RIGHT:
                character_to_x += character_speed
            elif event.key == pygame.K_SPACE: # 무기발사
                weapon_x_pos = character_x_pos + (character_width / 2) - (weapon_width / 2)
                weapon_y_pos = character_y_pos
                weapons.append([weapon_x_pos, weapon_y_pos])

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                    character_to_x = 0


    # 3. 위치조정
    # 3-1. 게임 캐릭터 위치 정의
    character_x_pos += character_to_x

    if character_x_pos < 0:
        character_x_pos = 0
    elif character_x_pos > screen_width - character_width:
        character_x_pos = screen_width - character_width

    # 3-2.무기 위치 정의
    weapons = [ [w[0], w[1] - weapon_speed] for w in weapons] # 무기위치 위로 올리기
    
    # 3-3. 천장 닿으면 제거
    weapons = [ [w[0], w[1]] for w in weapons if w[1] > 0]

    # 4. 충돌 처리

    # 5. 화면에 그리기
    # 배경 > 무기 > 스테이지
    screen.blit(background, (0,0))

    for weapon_x_pos, weapon_y_pos in weapons:
        screen.blit(weapon, (weapon_x_pos, weapon_y_pos))

    screen.blit(stage, (0,screen_height - stage_height))
    screen.blit(character, (character_x_pos, character_y_pos))
    
    

    pygame.display.update()

pygame.quit()

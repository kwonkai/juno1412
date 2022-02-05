from numpy import character
import pygame

pygame.init() # 초기화 (반드시 필요)

# 화면크기 설정
screen_width = 600
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))

# 화면 타이틀 설정
pygame.display.set_caption("MiniGame")

# FPS 설정
clock = pygame.time.Clock()

# 화면배경 설정
background = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/background.jpg")

# 캐릭터(주인공) 불러오기
character = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/character.jpg")
character_size = character.get_rect().size # 이미지 크기 설정
character_width = character_size[0] # 캐릭터 가로
character_height = character_size[1] # 캐릭터 세로
character_x_pos = (screen_width / 2) - (character_width / 2) # 캐릭터 위치(가로)
character_y_pos = screen_height - character_height #캐릭터 위치(세로)

# 이동할 좌표
to_x = 0
to_y = 0

# 이동속도
character_speed = 0.8

# 적 캐릭터
enemy = pygame.image.load("C:/Users/kwonk/Downloads/개인 프로젝트/미니게임/enemy.jpg")
enemy_size = enemy.get_rect().size # 이미지 크기 설정
enemy_width = enemy_size[0] # 캐릭터 가로
enemy_height = enemy_size[1] # 캐릭터 세로
enemy_x_pos = (screen_width / 2) - (enemy_width / 2) # 캐릭터 위치(가로)
enemy_y_pos = (screen_height / 2) - (enemy_height / 2) #캐릭터 위치(세로)


# 이벤트 종료
running = True # 게임이 진행중인가?
while running:
    dt = clock.tick(60) # 게임화면 초당 프레임 수

    for event in pygame.event.get(): # 어떤 이벤트가 발생하였는지 체크
        if event.type == pygame.QUIT: # 창이 닫히는 이벤트가 발생하였는가?
            running = False # 게임 진행중이 아님

        if event.type == pygame.KEYDOWN: # 키가눌러졌는지 확인
            if event.key == pygame.K_LEFT: # 캐릭터를 왼쪽으로
                to_x -= character_speed
            if event.key == pygame.K_RIGHT: # 캐릭터 오른쪽으로
                to_x += character_speed
            if event.key == pygame.K_UP: # 캐릭터 위로
                to_y -= character_speed
            if event.key == pygame.K_DOWN: # 캐릭터 아래쪽으로
                to_y += character_speed
        if event.type == pygame.KEYUP: # 방향키 떼면 멈춤
            if event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT:
                to_x = 0
            elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                to_y = 0

    character_x_pos += to_x * dt
    character_y_pos += to_y * dt

    # 게임 화면창 경계 설정
    if character_x_pos < 0:
        character_x_pos = 0
    elif character_x_pos > screen_width - character_width:
        character_x_pos = screen_width - character_width

    if character_y_pos < 0:
        character_y_pos = 0
    elif character_y_pos > screen_height - character_height:
        character_y_pos = screen_height - character_height

    # 충돌처리
    character_rect = character.get_rect()
    character_rect.left = character_x_pos
    character_rect.top = character_y_pos

    enemy_rect = enemy.get_rect()
    enemy_rect.left = enemy_x_pos
    enemy_rect.top = enemy_y_pos

    # 충돌 체크
    if character_rect.colliderect(enemy_rect):
        print("충돌")
        running = False


    # screen.fill((239, 224, 176))
    screen.blit(background, (0, 0)) # 배경그리기

    screen.blit(character, (character_x_pos, character_y_pos)) # 주인공 그리기

    screen.blit(enemy, (enemy_x_pos, enemy_y_pos)) # 적 그리기

    pygame.display.update() # 게임화면 다시 그리기



# pygame 종료
pygame.quit()
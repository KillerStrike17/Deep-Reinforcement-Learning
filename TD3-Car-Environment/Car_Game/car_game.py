import pygame
pygame.init()

win = pygame.display.set_mode((1430,660))
pygame.display.set_caption("Car Game ")
bg = pygame.image.load('citymap.png')
char = pygame.image.load('car.png')
# print(type(char))
# print(dir(char))
char = pygame.transform.scale(char, (20, 10))
clock = pygame.time.Clock()

class player(object):
    def __init__(self,x,y,width,height,vel):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.vel = vel
    
    def draw(self,win):
        win.blit(char,(self.x,self.y))

def redrawGameWindow():    
    win.blit(bg,(0,0))
    # pygame.draw.rect(win,(255,0,0),(x,y,width,height))
    car.draw(win)
    pygame.display.update()

car = player(50,425,40,60,5)
run = True

while run:
    clock.tick(27)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            run = False

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT] and car.x>car.vel:
        car.x -= car.vel 
    if keys[pygame.K_RIGHT] and car.x < 1430-car.width-car.vel:
        car.x += car.vel
    if keys[pygame.K_UP] and car.y>car.vel:
        car.y -= car.vel
    if keys[pygame.K_DOWN] and car.y<660-car.height-car.vel:
        car.y += car.vel

    redrawGameWindow()
pygame.quit()


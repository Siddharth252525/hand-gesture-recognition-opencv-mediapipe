import pygame

pygame.mixer.init()

sound = pygame.mixer.Sound("rasengan.wav")
sound.play()

input("Press Enter to exit")
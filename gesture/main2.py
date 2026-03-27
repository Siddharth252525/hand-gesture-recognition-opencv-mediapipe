import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import math
import random
import pygame

# -----------------------
# SOUND
# -----------------------

pygame.mixer.init()

try:
    rasengan_sound = pygame.mixer.Sound("rasengan.wav")
    chidori_sound = pygame.mixer.Sound("chidori.wav")
    kamehameha_sound = pygame.mixer.Sound("kamehameha.wav")
except:
    rasengan_sound = None
    chidori_sound = None
    kamehameha_sound = None

# -----------------------
# MEDIAPIPE MODEL
# -----------------------

model_path = "hand_landmarker.task"

base_options = BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------
# PARTICLE CLASS
# -----------------------

class Particle:

    def __init__(self,x,y):

        self.x = x
        self.y = y

        self.vx = random.uniform(-2,2)
        self.vy = random.uniform(-2,2)

        self.life = random.randint(10,25)

    def update(self):

        self.x += self.vx
        self.y += self.vy

        self.life -= 1

    def draw(self,frame):

        if self.life > 0:

            cv2.circle(frame,(int(self.x),int(self.y)),2,(255,200,0),-1)


particles = []

# -----------------------
# RASENGAN EFFECT
# -----------------------

def draw_rasengan(frame, x, y, angle):

    overlay = frame.copy()

    # 1️⃣ bright core
    cv2.circle(overlay,(x,y),40,(255,200,0),-1)

    # 2️⃣ spiral vortex
    for i in range(60):

        r = 15 + i*1.5
        a = angle + i*0.35

        px = int(x + math.cos(a)*r)
        py = int(y + math.sin(a)*r)

        cv2.circle(overlay,(px,py),3,(255,170,0),-1)

    # 3️⃣ outer energy particles
    for i in range(80):

        r = random.randint(40,80)
        a = random.random()*6.28

        px = int(x + math.cos(a)*r)
        py = int(y + math.sin(a)*r)

        cv2.circle(overlay,(px,py),2,(255,220,100),-1)

    # 4️⃣ strong glow
    blur = cv2.GaussianBlur(overlay,(55,55),0)

    frame = cv2.addWeighted(frame,1,blur,0.45,0)

    return frame


# -----------------------
# CHIDORI EFFECT
# -----------------------

def draw_chidori(frame,x,y):

    overlay = frame.copy()

    for _ in range(25):

        rx = x + random.randint(-120,120)
        ry = y + random.randint(-120,120)

        cv2.line(overlay,(x,y),(rx,ry),(255,255,255),2)

    blur = cv2.GaussianBlur(overlay,(25,25),0)

    frame = cv2.addWeighted(frame,1,blur,0.4,0)

    return frame


# -----------------------
# KAMEHAMEHA EFFECT
# -----------------------

def draw_kamehameha(frame,x,y,width):

    overlay = frame.copy()

    for i in range(0,width,25):

        cv2.circle(overlay,(x+i,y),25,(255,255,0),-1)

    cv2.line(overlay,(x,y),(x+width,y),(255,255,255),10)

    blur = cv2.GaussianBlur(overlay,(45,45),0)

    frame = cv2.addWeighted(frame,1,blur,0.4,0)

    return frame


# -----------------------
# CAMERA
# -----------------------

cap = cv2.VideoCapture(0)

angle = 0

# -----------------------
# MAIN LOOP
# -----------------------

while True:

    ret,frame = cap.read()

    if not ret:
        break

    h,w,_ = frame.shape

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        # TWO HANDS -> KAMEHAMEHA
        if len(result.hand_landmarks) == 2:

            frame = draw_kamehameha(frame,150,300,500)

            cv2.putText(frame,"KAMEHAMEHA",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,(255,255,0),3)

            if kamehameha_sound and not pygame.mixer.get_busy():
                kamehameha_sound.play()

        else:

            for i,landmarks in enumerate(result.hand_landmarks):

                handedness = result.handedness[i][0].category_name

                x = int(landmarks[0].x * w)
                y = int(landmarks[0].y * h)

                # RIGHT HAND -> RASENGAN
                if handedness == "Right":

                    frame = draw_rasengan(frame,x,y,angle)

                    cv2.putText(frame,"RASENGAN",
                                (40,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,(255,100,0),3)

                    if rasengan_sound and not pygame.mixer.get_busy():
                        rasengan_sound.play()

                    for _ in range(5):
                        particles.append(Particle(x,y))

                # LEFT HAND -> CHIDORI
                elif handedness == "Left":

                    frame = draw_chidori(frame,x,y)

                    cv2.putText(frame,"CHIDORI",
                                (40,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,(255,255,255),3)

                    if chidori_sound and not pygame.mixer.get_busy():
                        chidori_sound.play()

    # PARTICLE UPDATE
    for p in particles[:]:

        p.update()
        p.draw(frame)

        if p.life <= 0:
            particles.remove(p)

    angle += 0.2

    cv2.imshow("Anime Gesture AR System",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
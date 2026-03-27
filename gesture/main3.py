import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python import BaseOptions
import numpy as np
import math
import random

# -----------------------------
# Mediapipe model
# -----------------------------

model_path = "hand_landmarker.task"

base_options = BaseOptions(model_asset_path=model_path)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)

# -----------------------------
# Particle physics
# -----------------------------

class Particle:

    def __init__(self,x,y):

        self.x = x
        self.y = y

        self.vx = random.uniform(-3,3)
        self.vy = random.uniform(-3,3)

        self.life = random.randint(10,30)

    def update(self):

        self.x += self.vx
        self.y += self.vy
        self.life -= 1

    def draw(self,frame):

        if self.life > 0:

            cv2.circle(frame,(int(self.x),int(self.y)),3,(255,200,100),-1)


particles = []

# -----------------------------
# Rasengan vortex
# -----------------------------

def rasengan(frame,x,y,angle):

    overlay = frame.copy()

    # core sphere
    cv2.circle(overlay,(x,y),45,(255,200,0),-1)

    # vortex spiral
    for i in range(80):

        r = 10 + i*1.5
        a = angle + i*0.25

        px = int(x + math.cos(a)*r)
        py = int(y + math.sin(a)*r)

        cv2.circle(overlay,(px,py),3,(255,160,0),-1)

    # outer particles
    for i in range(80):

        r = random.randint(50,90)
        a = random.random()*6.28

        px = int(x + math.cos(a)*r)
        py = int(y + math.sin(a)*r)

        cv2.circle(overlay,(px,py),2,(255,230,120),-1)

    blur = cv2.GaussianBlur(overlay,(55,55),0)

    frame = cv2.addWeighted(frame,1,blur,0.45,0)

    return frame

# -----------------------------
# Branching lightning
# -----------------------------

def lightning(frame,x,y):

    for i in range(12):

        rx = x + random.randint(-200,200)
        ry = y + random.randint(-200,200)

        steps = 6

        px,py = x,y

        for j in range(steps):

            nx = px + (rx-px)//2 + random.randint(-20,20)
            ny = py + (ry-py)//2 + random.randint(-20,20)

            cv2.line(frame,(px,py),(nx,ny),(255,255,255),2)

            px,py = nx,ny

# -----------------------------
# Kamehameha beam
# -----------------------------

def kamehameha(frame,x,y,width):

    overlay = frame.copy()

    for i in range(0,width,30):

        cv2.circle(overlay,(x+i,y),35,(255,255,0),-1)

    cv2.line(overlay,(x,y),(x+width,y),(255,255,255),12)

    blur = cv2.GaussianBlur(overlay,(65,65),0)

    frame = cv2.addWeighted(frame,1,blur,0.45,0)

    return frame

# -----------------------------
# Camera
# -----------------------------

cap = cv2.VideoCapture(0)

angle = 0

# motion trail buffer
trail = None

while True:

    ret,frame = cap.read()

    if not ret:
        break

    h,w,_ = frame.shape

    # motion blur trail
    if trail is None:
        trail = frame.copy()

    frame = cv2.addWeighted(frame,0.7,trail,0.3,0)

    trail = frame.copy()

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:

        if len(result.hand_landmarks) == 2:

            frame = kamehameha(frame,150,300,500)

            cv2.putText(frame,"KAMEHAMEHA",
                        (40,60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,(255,255,0),3)

        else:

            for i,landmarks in enumerate(result.hand_landmarks):

                handedness = result.handedness[i][0].category_name

                x = int(landmarks[0].x * w)
                y = int(landmarks[0].y * h)

                if handedness == "Right":

                    frame = rasengan(frame,x,y,angle)

                    cv2.putText(frame,"RASENGAN",
                                (40,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,(255,150,0),3)

                    for _ in range(8):
                        particles.append(Particle(x,y))

                else:

                    lightning(frame,x,y)

                    cv2.putText(frame,"CHIDORI",
                                (40,60),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.5,(255,255,255),3)

    # particle updates

    for p in particles[:]:

        p.update()
        p.draw(frame)

        if p.life <= 0:
            particles.remove(p)

    angle += 0.35

    cv2.imshow("Anime Energy System",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
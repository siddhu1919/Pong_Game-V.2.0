import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from utils import *

# Initialize the webcam
cap = cv2.VideoCapture(1)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

# Importing all images and videos
imgBackground = cv2.imread('Resources/Background.png')
imgGameOver = cv2.imread('Resources/gameOver.png')
imgGameStart = cv2.imread('Resources/gameStart.png')
imgBall = cv2.imread('Resources/ball.png', cv2.IMREAD_UNCHANGED)
imgBat1 = cv2.imread('Resources/bat1.png', cv2.IMREAD_UNCHANGED)
imgBat2 = cv2.imread('Resources/bat2.png', cv2.IMREAD_UNCHANGED)
capSinglePlayer = cv2.VideoCapture('Resources/singlePlayer.gif')
capTwoPlayers = cv2.VideoCapture('Resources/TwoPlayers.gif')

# Hand Detector
detector = HandDetector(detectionCon=0.8, maxHands=2)

# Class for managing game logic
class HandPongGame:
    def __init__(self):
        # Initialize game variables
        self.ballPos = [200, 200]
        self.speedX = 40
        self.speedY = 40
        self.gameOver = False
        self.score = [0, 0]
        self.gameStart = True
        self.solo = True

    def checkBat(self, hand, img):
        # Check the position of the hand and update game state accordingly
        x, y, w, h = hand["bbox"]
        h1, w1, _ = imgBat1.shape
        y1 = np.clip(y - h1 / 2, 20, 415)

        if hand['type'] == "Left":
            img = cvzone.overlayPNG(img, imgBat1, (59, int(y1)))
            if 59 < self.ballPos[0] < 59 + w1 and y1 < self.ballPos[1] < y1 + h1:
                self.ballPos[0] += 30
                self.speedX = -self.speedX
                if self.solo:
                    self.score[0] += 1

        if hand['type'] == "Right":
            img = cvzone.overlayPNG(img, imgBat2, (1195, int(y1)))
            if 1195 - 50 < self.ballPos[0] < 1195 and y1 < self.ballPos[1] < y1 + h1:
                self.ballPos[0] -= 30
                self.speedX = -self.speedX
                if self.solo:
                    self.score[1] += 1

        return img

    def checkGameOver(self):
        # Check if the game is over and update game state accordingly
        if self.solo:
            if self.ballPos[0] < 40 or self.ballPos[0] > 1200:
                self.gameOver = True
        else:
            if self.ballPos[0] < 40:
                self.score[1] += 1
            elif self.ballPos[0] > 1200:
                self.score[0] += 1
            if self.ballPos[0] < 40 or self.ballPos[0] > 1200:
                self.ballPos = [100, 100]
                self.speedX = 30
                self.speedY = 30
            if max(self.score) == 5:
                self.gameOver = True

        return self.gameOver

    def updateScore(self, img):
        # Update score display on the screen
        cv2.putText(img, str(self.score[0]), (300, 650), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)
        cv2.putText(img, str(self.score[1]), (900, 650), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 5)

    def updateBall(self, img):
        # Update ball position and draw it on the screen
        if self.ballPos[1] >= 500 or self.ballPos[1] <= 10:
            self.speedY = -self.speedY

        self.ballPos[0] += self.speedX
        self.ballPos[1] += self.speedY

        img = cvzone.overlayPNG(img, imgBall, self.ballPos)
        return img

    def restart(self):
        # Reset all game variables for a new game
        self.ballPos = [200, 200]
        self.speedX = 30
        self.speedY = 30
        self.gameOver = False
        self.score = [0, 0]
        self.gameStart = True
        self.solo = True

    def start(self, solo):
        # Start a new game with specified mode (solo or two players)
        self.restart()
        self.solo = solo
        self.gameStart = False

    def showEndScore(self, img):
        # Show the final score on the game over screen
        if self.solo:
            cv2.putText(img, str(self.score[0] + self.score[1]).zfill(2), (585, 360), cv2.FONT_HERSHEY_DUPLEX,
                        2.5, (0, 0, 0), 5)
        else:
            cv2.putText(img, f"{str(self.score[0])}:{str(self.score[1])}", (585, 360), cv2.FONT_HERSHEY_DUPLEX,
                        2.5, (0, 0, 0), 5)


game = HandPongGame()


# Main loop for the game
while True:
    # Read frame from the webcam
    _, img = cap.read()

    # Flip the frame horizontally
    img = cv2.flip(img, 1)
    imgRaw = img.copy()

    # Find the hand and its landmarks
    hands, img = detector.findHands(img, flipType=False)  # with draw

    # Overlaying the background image
    img = cv2.addWeighted(img, 0.2, imgBackground, 0.8, 0)

    # Check for hands
    if hands:
        for hand in hands:
            img = game.checkBat(hand, img)

    # Check for game over or game start
    if game.checkGameOver():
        img = imgGameOver
        game.showEndScore(img)
    elif game.gameStart:
        img = imgGameStart
    # If game is not over, move the ball and update score
    else:
        # Move the Ball
        img = game.updateBall(img)
        game.updateScore(img)

    # Display the webcam feed and other game elements
    img[580:700, 20:233] = cv2.resize(imgRaw, (213, 120))
    cv2.imshow("Pong Game", img)

    # Wait for user input
    key = cv2.waitKey(1)
    if key == ord('r'):
        # Restart the game
        game.restart()
        imgGameOver = cv2.imread('Resources/gameOver.png')
    elif key == ord('1') and game.gameStart:
        # Start a solo game
        game.start(solo=True)
        displayGif(capSinglePlayer)
    elif key == ord('2') and game.gameStart:
        # Start a two-player game
        game.start(solo=False)
        displayGif(capTwoPlayers)
    elif key == ord('q'):
        # Quit the game
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

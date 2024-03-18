import cv2

def displayGif(gif, wind = "Pong Game"):
    i = 20
    while True:
        ret, img = gif.read()
        if ret:
            cv2.imshow(wind, img)
        else:
            break
        cv2.waitKey(i)
        i -= 1

import numpy as np
import cv2

def rescale_frame(frame, percent=76):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
def GrayTo3DGray(img_gray):
    h,w=img_gray.shape    
    gray3D=np.zeros((h,w,3),dtype=np.uint8)
    for i in range(0,3):
            gray3D[:,:,i]=img_gray
    return gray3D



file="videos/Dance_1.mp4"
#file="videos/APU_yosakoi.mp4"
cap = cv2.VideoCapture(file)

out=None
avg=None
i=0
w=None
h=None
BGRcol=[0,255,255]
img=cv2.imread("videos/Outpattern.jpg")
frameDelta=None
while(cap.isOpened()):
    i+=1
    
    ret,frame= cap.read()
    frame= rescale_frame(frame,percent=10)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if (avg is None) or (i % 50 ==1):
        avg = gray.copy().astype("float")
        
        continue

    cv2.accumulateWeighted(gray,avg,0.5)

    if i<10:
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
        newframeDelta=frameDelta
    else:
        newframeDelta=cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        frameDelta = cv2.addWeighted(
           newframeDelta, 1, 
           frameDelta, 0.9, 0)


    
    #領域を抽出
    threash=cv2.threshold(frameDelta,50,225,cv2.THRESH_BINARY)[1]
    newthreash=cv2.threshold(grey,100,255,cv2.THRESH_BINARY_INV)[1]

    contours, hierarchy = cv2.findContours(newthreash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = max(contours, key=lambda x: cv2.contourArea(x))


    
    #新しいのフラームを作成する
    if w==None:
        h,w,c=frame.shape
        background=np.zeros((h,w,3),dtype=np.uint8)
        mask=np.zeros((h,w,3),dtype=np.uint8)
     #   graymask=mask.copy()
        #h,w=gray.shape
        img=cv2.resize(img,(w,h))
        for c,col in enumerate(BGRcol):
            background[:,:,c]=img[:,:,c]
            mask[:,:,c]=frameDelta-newframeDelta
          #  graymask[:,:,c]

    mask=GrayTo3DGray(threash)
    newmask=GrayTo3DGray(newthreash)
    frame_masked = cv2.bitwise_and(background,mask)
    cv2.drawContours(mask, [contours], -1, (255,255,0), 2)
    gray=GrayTo3DGray(newframeDelta)
    #frameDelta3D=GrayTo3DGray(frameDelta)

    frameWeighted = cv2.addWeighted(frame, 1, frame_masked, 0.6, 0)
    displayframe=cv2.hconcat([mask,frameWeighted,newmask])
    cv2.imshow("Dance",displayframe)


    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()


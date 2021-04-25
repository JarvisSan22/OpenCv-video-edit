
import numpy as np
import cv2

def rescale_frame(frame, percent=75):
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
cap = cv2.VideoCapture(file)

fourcc =cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # ファイル形式(ここではmp4) #cv2.VideoWriter_fourcc(*'XVID')
outname="videos/Dance_layertest.mp4"

out=None
avg=None
i=0
w=None
h=None
BGRcol=[0,255,255]


while(cap.isOpened()):

    try:
        i+=1
        ret,frame= cap.read()
        frame= rescale_frame(frame,percent=8)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #accumulate Weighted
        # 前フレームを保存
        if avg is None:
            avg = gray.copy().astype("float")
        
            continue
        #if i % 50 ==1:
        #   avg = gray.copy().astype("float")
        #  continue

        cv2.accumulateWeighted(gray,avg,0.1)
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))

        #領域を抽出
        threash=cv2.threshold(frameDelta,50,225,cv2.THRESH_BINARY)[1]
        # 輪郭を見つける
        #_, contours, hierarchy = cv2.findContours(thresh.copy(), 
        #cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        #新しいのフラームを作成する
        if w==None:
            h,w,c=frame.shape
            background=np.zeros((h,w,3),dtype=np.uint8)
            mask=np.zeros((h,w,3),dtype=np.uint8)
            for c,col in enumerate(BGRcol):
                background[:,:,c]=col
                mask[:,:,c]=threash
        #print(background.shape)
        #print(mask.shape)
        mask=GrayTo3DGray(threash)
        frame_masked = cv2.bitwise_and(background,mask)

        gray=GrayTo3DGray(gray)
        frameDelta=GrayTo3DGray(frameDelta)

        frametop=cv2.hconcat([gray,frameDelta])
        framebot=cv2.hconcat([mask,frame_masked])
        outframe=cv2.vconcat([frametop,framebot])
        
        cv2.imshow("Dance",outframe)

        if out==None:
            h,w,c=gray.shape
            out = cv2.VideoWriter(outname,fourcc, 20.0, (2*h,2*w))

        out.write(outframe)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    except:
        break
cap.release()
out.release()
cv2.destroyAllWindows()
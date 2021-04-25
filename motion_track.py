
import numpy as np
import cv2
import matplotlib.pyplot as plt


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


def generate_frame_delta(img_gray,avg,frameDelta=None,newfadeW=1,oldfadeW=0.9):
    if frameDelta is None:
        frameDelta = cv2.absdiff(img_gray, cv2.convertScaleAbs(avg))
        newframeDelta=frameDelta
    else:
        newframeDelta=cv2.absdiff(img_gray, cv2.convertScaleAbs(avg))
        frameDelta = cv2.addWeighted(
           newframeDelta, newfadeW, 
           frameDelta, oldfadeW, 0)

    return frameDelta,newframeDelta

def create_background_mask(h,w,back_img):
        background=np.zeros((h,w,3),dtype=np.uint8)
        img=cv2.resize(back_img,(w,h))
        for c in range(0,3):
            background[:,:,c]=img[:,:,c]
        return background  


#画像の解析
def RBGhist(img,plot=False):
    h,w,c=img.shape
    #RBGのヒストグラム
    fig,axs=plt.subplots(2,1,figsize=(8,8))
    #monoのimgsを作成する
    RBG_imgs=[]
    colors=["blue","green","red"]
    for coli,col in zip(range(c),colors):
      #monoの画像
        img_c=np.zeros((h,w,c),dtype=np.uint8)
        img_c[:,:,coli]=img[:,:,coli]
        RBG_imgs.append(img_c)
    #色によりヒストグラム
    hist_c=cv2.calcHist([img],[coli],None,[256],[1,255])
    axs[1].plot(hist_c,color=col,label=col)

    RBG_monos=cv2.hconcat(RBG_imgs)
    axs[0].imshow(cv2.cvtColor(RBG_monos,cv2.COLOR_BGR2RGB))
    axs[0].set_axis_off()
    axs[1].legend()

    if plot:
        plt.show()


    return fig


def grayhist(img,plot=False):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray_hist = cv2.calcHist([img_gray],[0],None,[256],[0,256])
    fig,axs=plt.subplots(2,1,figsize=(8,8))
    axs[0].imshow(img_gray,cmap="gray")
    axs[0].set_axis_off()
    axs[1].plot(img_gray_hist,color="gray")
    if plot:
        plt.show()
    return fig,img_gray

def video_analysis(filevideo,percent=10):
    cap = cv2.VideoCapture(filevideo)
    i=0

    baseframe=None 
    while(cap.isOpened()):
        
        i+=1
        try:
            ret,frame= cap.read()
            frame= rescale_frame(frame,percent=percent)
            if baseframe is None:
                baseframe=frame
            else:
                baseframe+=frame
            cv2.imshow("Test",frame)
        
            if cv2.waitKey(1)  & 0xFF == ord('q'):
                break
        except:
            print(i)
           
       
    
            baseframe=baseframe //i
            fig_RBG=RBGhist(baseframe)
            fig_RBG.savefig(f"{filevideo[0:filevideo.find('.')]}_RBG_Analysis.jpg")
            fig_gray,img_gray=grayhist(baseframe)
            fig_gray.savefig(f"{filevideo[0:filevideo.find('.')]}_gray_Analysis.jpg")

            cap.release()
            cv2.destroyAllWindows()
          
            break

            


def video_background(filevideo,backgroundfile,
outtype="gray",percent=10,
accumulateWeightedval=0.5,frameweighting=0.6,threashval=[0,200]):
    #outtype: "gray", "newfade","pastfade" plus: "_inv"


    #file="videos/Dance_1.mp4"
    #file="videos/APU_yosakoi.mp4"
    cap = cv2.VideoCapture(filevideo)
    back_img=cv2.imread(backgroundfile)
    out=None
    avg=None
    i=0
    w=None
    h=None
    #BGRcol=[0,255,255]
    frameDelta=None
    background=None

    
    #Threadhold type
    ThreashType=cv2.THRESH_BINARY
    if "inv" in outtype:
        ThreashType=cv2.THRESH_BINARY_INV
    

    while(cap.isOpened()):
        i+=1
        ret,frame= cap.read()
        frame= rescale_frame(frame,percent=percent)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,0]

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        if (avg is None) or (i % 50 ==1):
            if avg is None:
                cv2.imwrite("frame_1.jpg",frame)
            if "hsv" in outtype:
                avg=hsv.copy().astype("float")
                
            else:
                avg = gray.copy().astype("float")
            h,w,c=frame.shape
            
            continue

        if i<10:
            if "hsv" in outtype:
                frameDelta,newframeDelta=generate_frame_delta(hsv,avg)
            else:
                frameDelta,newframeDelta=generate_frame_delta(gray,avg)
        else:
              if "hsv" in outtype:
                frameDelta,newframeDelta=generate_frame_delta(hsv,avg,frameDelta=frameDelta)
              else:
                frameDelta,newframeDelta=generate_frame_delta(gray,avg,frameDelta=frameDelta)

        #領域を抽出

        threash=cv2.threshold(frameDelta,threashval[0],threashval[1],cv2.THRESH_BINARY)[1]
        #cv2.THRESH_BINARY_INV
        newthreash=cv2.threshold(newframeDelta,threashval[0],threashval[1],ThreashType)[1]
        graythreash=cv2.threshold(gray,threashval[0],threashval[1],ThreashType)[1]
        hsvthreash=cv2.threshold(hsv,threashval[0],threashval[1],ThreashType)[1]

        if background is None:
            h,w,c=frame.shape
            background=create_background_mask(h,w,back_img)
        

        #processe imager type 

        #if "-" in outtype:

         #   outtype=outtype[]
        if "pastfade-gray" in outtype:
            combine_threash=cv2.bitwise_xor(graythreash,threash)

            mask=GrayTo3DGray(combine_threash)
            
        elif "gray" in outtype.lower():
            mask=GrayTo3DGray(graythreash)
        elif "newfade" in outtype.lower():
            mask=GrayTo3DGray(newthreash)
        elif "pastfade" in outtype.lower():
            mask=GrayTo3DGray(threash)
        elif "hsv" in outtype.lower():
            mask=GrayTo3DGray(hsvthreash)
    

        frame_masked = cv2.bitwise_and(background,mask)
        frameWeighted = cv2.addWeighted(frame, 1-frameweighting, frame_masked, frameweighting, 0)
        displayframe=cv2.hconcat([frame,mask,frameWeighted])
        cv2.imshow(outtype,displayframe)
        if cv2.waitKey(1)  & 0xFF == ord('q'):
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    filevideo="videos/pexels-anthony-shkraba-7569398.mp4"
    backgroundfile="videos/pexels-alex-andrews-816608.jpg"


   # video_analysis(filevideo,percent=20)
    #outtype= "gray", "newfade","pastfade"
    outtype= "hsv"


    video_background(filevideo,backgroundfile,
    outtype=outtype,percent=15,
    accumulateWeightedval=0.05,frameweighting=1,threashval=[25,255])
    


if __name__ == "__main__":
    main()
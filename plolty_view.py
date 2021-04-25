import plotly.express as px
import plotly
import plotly.graph_objects as go
import cv2
from ipywidgets import widgets
from plotly.subplots import make_subplots

def colorhist(img,imgtype,grayimg=None,save=False):
    #Plolty Hist 
    if imgtype=="BGR":
        colors=["blue","green","red"]
        labels=["blue","green","red"]
    elif imgtype=="HSV":
        colors=["blue","green","red"]
        labels=["H","S","V"]
    coldict={}
    ff=go.Figure()
    bgr_img=None
    for i,col in enumerate(colors):
        c_img=img[:,:,i]
        coldict[col]=c_img
        ff.add_trace(go.Histogram(x=c_img.flatten()[c_img.flatten()!=0],marker_color=col,name=labels[i]))
        if bgr_img is None:
            bgr_img=c_img
        else:
            bgr_img=cv2.hconcat([bgr_img,c_img])
    if grayimg:
        ff.add_trace(go.Histogram(x=grayimg.flatten()[grayimg.flatten()!=0],marker_color="gray",name="gray"))
    if save:           
        bgr=px.imshow(bgr_img)
        bgr.write_html(f"{imgtype}_colors.html")
        bgr.show()
        ff.write_html(f"{imgtype}_hist.html")
        ff.show()

frame= "frame_1.jpg"
img =  cv2.imread(frame)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

fig=px.imshow(img)
fig.write_html("pic.html")
fig.show()
gray_fig=px.imshow(gray)
gray_fig.write_html("pic_gray.html")
gray_fig.show()
#full_fig.add_trace(tx, row=1, col=1)


# Convert from BGR to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_fig=px.imshow(hsv)
hsv_fig.write_html("pic_hsv.html")
hsv_fig.show()

colorhist(hsv,"HSV",grayimg=None,save=True)
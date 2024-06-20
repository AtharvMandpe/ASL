from all_lib import *
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
    #print(voices[1].id)
engine.setProperty('voice', voices[0].id) # you can change to female voice just by replacing 0 by 1 in voices[0].
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
temp=np.zeros((512,512,3),np.uint8)
temp=cv.rectangle(temp,(110,150),(350,220),(255,255,255),-1)
cv.putText(temp,'ASL',(184,204),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
temp=cv.rectangle(temp,(110,350),(350,420),(255,255,255),-1)
cv.putText(temp,'EXIT',(164,404),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
cv.imshow('j9 project',temp)
button=10
k=0
def click(event,xdir,ydir,flags,param):
    if event==cv.EVENT_LBUTTONDOWN:
      #print("xcor = ",xdir," ycor = ",ydir)
      if(110<xdir<350 and 150<ydir<220):
          global button
          button=100
          cv.destroyAllWindows()
      if(110<xdir<350 and 250<ydir<320):
          button=20
          cv.destroyAllWindows()
      if(110<xdir<350 and 350<ydir<420):
          button = 40
          cv.destroyAllWindows()
cv.setMouseCallback('j9 project',click)
cv.waitKey(0)
if(button==100):
 cap=cv.VideoCapture(0)
 k=0
 mphands=mp.solutions.hands
 hands=mphands.Hands()
 dra=mp.solutions.drawing_utils
#h=cv.VideoWriter_fourcc(*'GDVC')
#c=cv.VideoWriter('yuyuyuyu.avi',h,30.00,(640,480))
 count=0
 word=[]
 while(True):
    ret,frame=cap.read()
    frame=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    #cv.imwrite('efrfe.jpg',frame)
    #print(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #c.write(frame)
    #print(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    lmlist=[]
    k=0
    results=hands.process(frame)
    if results.multi_hand_landmarks:#lmlist[4][1]
     for j in results.multi_hand_landmarks:
          
         for id,lm in enumerate(j.landmark):
             h,w,c=frame.shape
             cx,cy=int(lm.x*w),int(lm.y*h)
          
            # print("landmark ",id," xcor=",cx," ycor=",cy)
             
             lmlist.append([cx,cy])
             cv.circle(frame,(cx,cy),5,(255,255,0),cv.FILLED)
         dra.draw_landmarks(frame,j,mphands.HAND_CONNECTIONS)
         
    if(len(lmlist)!=0):
     #print(lmlist)
     #if(lmlist[4][1]<lmlist[6][1] and lmlist[20][1]>lmlist[16][1]>lmlist[12][1]):
     
       #cv.putText(frame,'like',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
     if(lmlist[4][1]<lmlist[6][1] and lmlist[4][1]<lmlist[10][1] and lmlist[4][1]<lmlist[14][1] and lmlist[4][1]<lmlist[18][1] and lmlist[12][1]>lmlist[16][1] and lmlist[12][1]>lmlist[20][1] and lmlist[12][1]>lmlist[8][1] and lmlist[20][0]<lmlist[8][0] and lmlist[4][0]>lmlist[6][0]):

         s=count
         string=str(s)
         cv.putText(frame,string+' A',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('a')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('A is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0

         
     elif(lmlist[4][0]<lmlist[5][0]  and lmlist[16][1]>lmlist[12][1] and lmlist[12][1]<lmlist[8][1] and lmlist[12][1]<lmlist[10][1] and lmlist[20][1]<lmlist[18][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' B',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('b')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('B is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(60<lmlist[4][1]-lmlist[12][1]<90 and lmlist[4][1]<lmlist[3][1] and -20<lmlist[16][1]-lmlist[12][1]<20 and lmlist[4][0]>lmlist[20][1] and lmlist[12][1]>lmlist[10][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' C',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('c')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('C is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(-20<lmlist[4][1]-lmlist[12][1]<20 and lmlist[1][1]-lmlist[8][1]>20 and lmlist[4][1]<lmlist[3][1] and -20<lmlist[16][1]-lmlist[12][1]<20 and -20<lmlist[8][1]-lmlist[12][1]<20 and -20<lmlist[20][1]-lmlist[12][1]<20 and lmlist[4][0]-lmlist[12][0]>0):
         s=count
         string=str(s)
         cv.putText(frame,string+' O',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('o')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('O is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(-10<lmlist[4][1]-lmlist[12][1]<20 and -10<lmlist[4][0]-lmlist[12][0]<20 and lmlist[8][1]<lmlist[12][1] and lmlist[8][1]<lmlist[16][1] and lmlist[8][1]<lmlist[20][1] and lmlist[8][1]<lmlist[4][1] and lmlist[4][1]-lmlist[8][1]>40 and lmlist[8][1]<lmlist[7][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' D',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('d')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('D is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(0<lmlist[4][1]-lmlist[20][1]<20 and 0<lmlist[3][1]-lmlist[12][1]>20 and lmlist[20][1]-lmlist[8][1]<15 and lmlist[4][0]-lmlist[12][0]<-25 and lmlist[4][1]>lmlist[7][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' E',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('e')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('E is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(-40<lmlist[4][1]-lmlist[8][1]<40 and lmlist[20][0]<lmlist[5][0] and lmlist[4][1]-lmlist[16][1]>50 and lmlist[10][1]>lmlist[12][1] and lmlist[11][1]>lmlist[12][1] and lmlist[20][1]<lmlist[18][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' F',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('f')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('F is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[8][0]>lmlist[4][0] and lmlist[8][0]>lmlist[12][0] and lmlist[8][0]>lmlist[16][0] and lmlist[8][0]>lmlist[20][0] and lmlist[8][1]<lmlist[12][1] and lmlist[4][1]<lmlist[6][1] and lmlist[12][1]<lmlist[20][1] and lmlist[4][0]-lmlist[12][0]>40):
         s=count
         string=str(s)
         cv.putText(frame,string+' G',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('g')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('G is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]<lmlist[4][1] and lmlist[20][1]<lmlist[16][1] and lmlist[20][1]<lmlist[12][1] and lmlist[20][1]<lmlist[8][1] and lmlist[4][0]<lmlist[6][0] and lmlist[18][1]>lmlist[20][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' I',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('i')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('I is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]<lmlist[4][1] and lmlist[20][1]<lmlist[16][1] and lmlist[20][1]<lmlist[12][1] and lmlist[20][1]<lmlist[8][1] and lmlist[12][0]-lmlist[4][0]<40 and lmlist[18][1]>lmlist[20][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' Y',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('y')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('Y is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][0]>lmlist[4][0] and lmlist[20][0]>lmlist[16][0] and lmlist[20][0]>lmlist[12][0] and lmlist[20][0]>lmlist[8][0]):
         s=count
         string=str(s)
         cv.putText(frame,string+' J',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('j')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('J is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[8][0]>lmlist[16][0] and lmlist[8][0]>lmlist[20][0] and lmlist[12][0]>lmlist[16][0] and lmlist[12][0]>lmlist[20][0] and lmlist[5][1]<lmlist[4][1] and -25<lmlist[8][0]-lmlist[12][0]<-7 and lmlist[17][1]>lmlist[0][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' H',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('h')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('H is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[8][0]>lmlist[4][0]>lmlist[12][0] and lmlist[16][1]-lmlist[12][1]>120 and lmlist[4][1]<lmlist[5][1] and lmlist[6][1]>lmlist[8][1] and lmlist[10][1]>lmlist[12][1] and lmlist[6][1]>lmlist[8][1] and lmlist[10][1]>lmlist[12][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' K',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('k')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('K is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[4][0]-lmlist[5][0]>20 and lmlist[4][0]-lmlist[12][0]>20 and lmlist[4][0]-lmlist[10][0]>20 and lmlist[4][1]-lmlist[8][1]>40 and  lmlist[12][1]-lmlist[8][1]>40 and lmlist[8][1]<lmlist[7][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' L',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('l')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('L is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]>lmlist[4][1] and lmlist[4][1]>lmlist[14][1] and lmlist[12][0]>lmlist[4][0] and lmlist[20][1]>lmlist[4][1] and lmlist[20][1]>lmlist[18][1] and lmlist[12][0]<lmlist[8][0] and lmlist[12][1]>lmlist[10][1] and lmlist[20][1]>lmlist[4][1] and lmlist[4][1]>lmlist[15][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' M',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('m')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('M is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[16][1]>lmlist[4][1] and lmlist[16][1]-lmlist[12][1]<150 and lmlist[12][0]>lmlist[4][0] and lmlist[12][0]<lmlist[8][0] and lmlist[4][1]>lmlist[11][1] and lmlist[8][1]>lmlist[6][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' N',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('n')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('N is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[5][0]>lmlist[4][0]>lmlist[9][0] and lmlist[16][1]<lmlist[12][1] and lmlist[12][1]>lmlist[4][1] and lmlist[8][1]>lmlist[4][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' P',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('p')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('P is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]>lmlist[4][1] and lmlist[20][1]>lmlist[8][1] and lmlist[20][1]>lmlist[12][1] and lmlist[20][1]>lmlist[16][1] and lmlist[10][1]>lmlist[0][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' Q',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('q')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('Q is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[12][0]>lmlist[8][0] and lmlist[4][1]<lmlist[16][1] and lmlist[16][1]>lmlist[20][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' R',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('r')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('R is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[4][1]<lmlist[11][1] and lmlist[4][1]<lmlist[7][1] and lmlist[4][0]<lmlist[6][0] and lmlist[0][1]>lmlist[4][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' S',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('s')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('S is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[4][1]<lmlist[11][1] and lmlist[7][1]<lmlist[4][1] and lmlist[14][1]>lmlist[18][1] and lmlist[18][1]<lmlist[20][1] and lmlist[0][1]>lmlist[4][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' T',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('t')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('T is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]>lmlist[4][1] and lmlist[4][1]>lmlist[14][1]  and lmlist[12][0]<lmlist[8][0] and lmlist[8][0]-lmlist[12][0]<50 and lmlist[10][1]>lmlist[12][1] and lmlist[20][1]>lmlist[16][1] and lmlist[16][1]>lmlist[14][1] and lmlist[11][1]>lmlist[12][1] and lmlist[6][1]>lmlist[8][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' U',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('u')
         if(count==51):
             engine=pyttsx3.init()
             engine.say('U is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]>lmlist[4][1] and lmlist[4][1]>lmlist[14][1] and lmlist[16][1]>lmlist[14][1] and lmlist[10][1]>lmlist[12][1] and lmlist[4][0]<lmlist[5][0] and lmlist[20][1]>lmlist[16][1] and lmlist[16][1]>lmlist[14][1] and lmlist[11][1]>lmlist[12][1] and lmlist[6][1]>lmlist[8][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' V',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('v')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('V is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[20][1]>lmlist[4][1] and lmlist[4][1]>lmlist[14][1] and lmlist[16][1]>lmlist[14][1] and lmlist[10][1]>lmlist[12][1] and lmlist[4][0]>lmlist[5][0]):
         s=count
         string=str(s)
         cv.putText(frame,string+' Z',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('z')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('Z is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[4][1]>lmlist[14][1] and lmlist[10][1]>lmlist[12][1] and lmlist[14][1]>lmlist[16][1] and lmlist[18][1]<lmlist[20][1] and lmlist[6][1]>lmlist[8][1] and lmlist[10][1]>lmlist[12][1] and lmlist[14][1]>lmlist[16][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' W',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('w')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('W is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[8][1]<lmlist[4][1] and lmlist[8][1]<lmlist[20][1] and lmlist[8][1]<lmlist[16][1] and lmlist[8][1]<lmlist[12][1] and lmlist[8][1]>lmlist[7][1] and lmlist[4][1]>lmlist[14][1] and lmlist[11][1]>lmlist[8][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' X',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'      is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append('x')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('X is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(lmlist[4][0]>lmlist[8][0] and lmlist[16][1]>lmlist[14][1] and lmlist[12][1]>lmlist[10][1] and lmlist[4][1]>lmlist[0][1] and lmlist[4][0]>lmlist[5][0]):
         s=count
         string=str(s)
         cv.putText(frame,string+' space',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'          is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.append(' ')

         if(count==51):
             engine=pyttsx3.init()
             engine.say('space is Collected')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     elif(-40<lmlist[4][1]-lmlist[8][1]<40 and lmlist[20][0]<lmlist[5][0] and lmlist[4][1]-lmlist[16][1]>50 and lmlist[10][1]>lmlist[12][1] and lmlist[11][1]>lmlist[12][1]):
         s=count
         string=str(s)
         cv.putText(frame,string+' delete',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
         count=count+1
         if(count==50):
             cv.putText(frame,'           is Collected',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             word.pop()

         if(count==51):
             engine=pyttsx3.init()
             engine.say('deleted previous letter')
             engine.runAndWait()
             cv.waitKey(2000)
             count=0
     #elif():
        # cv.putText(frame,'Z',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
     elif(lmlist[4][1]>lmlist[14][1] and lmlist[10][1]>lmlist[12][1] and lmlist[14][1]>lmlist[16][1] and lmlist[8][1]<lmlist[6][1]):
         count=count+1
         if(count==50):

             tttt=''.join(word)
             import pyttsx3
             engine=pyttsx3.init()
             cv.putText(frame,tttt,(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
             engine.say(tttt)
             engine.runAndWait()
             word=[]
         if(count==51):
             count=0
             cv.waitKey(5000)
        
     #elif(lmlist[12][1]<lmlist[4][1] and lmlist[12][1]<lmlist[16][1] and lmlist[12][1]<lmlist[12][1] and lmlist[12][1]<lmlist[8][1] and lmlist[8][0]>lmlist[12][0] and lmlist[8][0]>lmlist[20][0] and lmlist[8][0]>lmlist[12][0] and lmlist[4][0]>lmlist[16][0]):
      #   cv.putText(frame,'Z',(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)
    cv.imshow('j9 project',frame)
    f=cv.waitKey(1)
    if(f==ord('q')):
        break
 cv.destroyAllWindows()
 cap.release()
if(button==20):
    cv.destroyAllWindows()

if(k==40):
 cv.destroyAllWindows()
 cap.release()




























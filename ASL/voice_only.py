import pyttsx3#pip install pyttsx3
import speech_recognition as sr #pip install speechRecognition
import datetime  
import re 
import wikipedia #pip install wikipedia
import webbrowser
import os                            
import smtplib
import sys
import requests
from socket import timeout
from bs4 import BeautifulSoup 
from datetime import date
import cv2
import cv2 as cv
import mediapipe as mp
import pyttsx3
import numpy as np
from googleplaces import GooglePlaces,types,lang
engine = pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
    #print(voices[1].id)
engine.setProperty('voice', voices[0].id) # you can change to female voice just by replacing 0 by 1 in voices[0].
def speak(audio):
    engine.say(audio)
    engine.runAndWait()
speak("Which mode do u want?")
k=int(input())
 # if showing any error in importing the libraray do "pip install libname" in your terminal or cmd prompt
 # installed but again showing error close your app pycharm or vscode and again open 
 # also everytime upgrade your pip version by "pip install --upgrade pip" in terminal after installing new library
if(k==1):
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    #print(voices[1].id)
    engine.setProperty('voice', voices[0].id) # you can change to female voice just by replacing 0 by 1 in voices[0].


    def speak(audio):
        engine.say(audio)
        engine.runAndWait()


    def wishMe():
        hour = int(datetime.datetime.now().hour)
        if hour>=0 and hour<12:
            speak("Good Morning!")

        elif hour>=12 and hour<18:
            speak("Good Afternoon!")   

        else:
            speak("Good Evening!")  

        speak("Hey there . My name is friday . Please tell me how may I help you")
        # you can change any speak sentence according to your need.      

    def takeCommand():
        #It takes microphone input from the user and returns string output

        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            r.pause_threshold = 1
            audio = r.listen(source)

        try:
            print("Recognizing...")    
            query = r.recognize_google(audio, language='en-in')
            print(f"User said: {query}\n")

        except Exception as e:
            # print(e)    
            print("Say that again please...")  
            return "None"
        return query



    def sendEmail(to, content):
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.ehlo()
        server.starttls()
        server.login('devansh.lathiya21@vit.edu','12111023') # also enable less secure apps setting in security setting of your mail account 
        server.sendmail('devansh.lathiya21@vit.edu', to, content)
        server.close()

    if __name__ == "__main__":
        wishMe()
        while True:
        # if 1:
            query = takeCommand().lower()
            if 'wikipedia' in query:
                speak('Searching Wikipedia...')
                query = query.replace("wikipedia", "")
                results = wikipedia.summary(query, sentences=2)
                speak("According to Wikipedia")
                print(results)
                speak(results)

            elif 'open youtube' in query:
                webbrowser.open("youtube.com")

            elif 'open google' in query:
                webbrowser.open("google.com")

            elif 'open stackoverflow' in query:
                webbrowser.open("stackoverflow.com") 

            elif 'search' in query :
                speak("what you want to search")
                find=takeCommand()
                webbrowser.open("https://www.google.com/search?q="+find+"&ei=xlCeYsWZNPnWz7sP2sycmA0&oq=flipk&gs_lcp=Cgdnd3Mtd2l6EAEYADILCAAQsQMQgwEQkQIyCwgAELEDEIMBEJECMgsIABCABBCxAxCDATILCAAQgAQQsQMQgwEyCwgAEIAEELEDEMkDMgUIABCSAzIFCAAQkgMyCwgAEIAEELEDEIMBMgsIABCABBCxAxCDATIICAAQgAQQsQM6BQgAEJECOgsILhCABBCxAxDUAjoRCC4QgAQQsQMQgwEQxwEQ0QNKBAhBGABKBAhGGABQydvWAViS6dYBYO771gFoAXABeACAAaMCiAHwBpIBBTAuNC4xmAEAoAEBsAEAwAEB&sclient=gws-wiz")

            elif 'order' in query :

                speak("from where you want to order from")
                order=takeCommand()
                #speak("opening"+find+"for you")
                webbrowser.open("https://www."+order+".com")

            elif "write a note" in query:
                speak("What should i write, sir")
                note = takeCommand()
                file = open('jarvis.txt', 'w')
                speak("Sir, Should i include date and time")
                snfm = takeCommand()
                if 'yes' in snfm or 'sure' in snfm:
                    strTime = datetime.datetime.now().strftime("% H:% M:% S")
                    file.write(strTime)
                    file.write(" :- ")
                    file.write(note)
                else:
                    file.write(note)    
                
            elif 'play movie' in query:
                movie_dir = 'D:\\Movies'
                songs = os.listdir(movie_dir)
                print(songs) 
                speak("Which movie you will love to see")   
                os.startfile(os.path.join(movie_dir,takeCommand()+".mkv")) # you have to change the path according to your device where your movie folder is present
                                                                        # also change ".mkv" according to your movie properties to open it where ".mkv" is file extension 
            elif 'the time' in query:
                strTime = datetime.datetime.now().strftime("%H:%M:%S")    
                speak(f"the time is {strTime}")
            
            elif 'presentation' in query:
                speak("opening Power Point presentation")
                power = r"C:\\Program Files\\Microsoft Office\\root\\Office16\\POWERPNT.EXE"  #  change path where ppt is present in your device
                os.startfile(power)

            elif 'open code' in query:
                codePath = "C:\\Users\\Haris\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe" # change path according to your device
                os.startfile(codePath)

            elif 'email to' in query:
                try:
                    speak("Enter the email in the console, to whom you want to sent the email")
                    to = input()
                    
                    speak("What should I say?")
                    content = takeCommand()
                    
                    sendEmail(to, content)
                    speak("Email has been sent!")
                except Exception as e:
                    print(e)
                    speak("Sorry sir. I am not able to send this email")

            
            elif "weather" in query:
                speak("today's weather according to google baba")
                webbrowser.open("https://www.google.com/search?q=today+weather&oq=todays+whe&aqs=chrome.5.69i57j0i10j0i10i512l3j0i10i131i433i457j0i402l2j0i10i512l2.7759j0j15&sourceid=chrome&ie=UTF-8")
            

            elif "train" in query:
                speak("please tell me train number and name ")
                train_no=takeCommand()
                speak("here is running status of train")
                webbrowser.open("https://www.railyatri.in/live-train-status/"+train_no+"?utm_source=lts_dweb_Check_status")
            
            elif "pnr" in query:
                speak("please enter ten digit PNR number in console ")
                pnr_no=input("PNR NO.=")
                speak("here is PNR status of train")
                webbrowser.open("railyatri.in/pnr-status/"+pnr_no)

            elif "journey" in query:
                speak("how would you like to go by bus or tain ")
                choice=takeCommand()
                if choice == 'bus':
                    speak("your prefrence by MSRTC OR sleeper")
                    pref=takeCommand()
                    if pref == 'msrtc':
                        webbrowser.open('https://www.redbus.in/online-booking/msrtc')
                        
                    else:
                        webbrowser.open('https://www.redbus.in/')
                        
                else:
                    speak("here i have opened booking site for you now enter the details and check")
                    webbrowser.open("https://www.railyatri.in/trains-between-stations?utm_source=")
                    

            elif 'open learning' in query:
                webbrowser.open("https://learner.vierp.in/")

            elif 'how to' in query:
                speak("what you want to search")
                search=takeCommand()
                webbrowser.open("https://www.youtube.com/results?search_query="+search)
            
            elif 'direction' in query:
                speak("from where ")
                place1=takeCommand()
                speak("to where")
                place2=takeCommand()
                webbrowser.open("https://www.google.com/search?q=how+to+go "+place1+" to "+place2+" &ei=fuidYtbJF_zd4-EPj62E8Ag&oq=&gs_lcp=Cgdnd3Mtd2l6EBJKBAhBGABKBAhGGABQAFgAYJKNA2gBcAF4AIABAIgBAJIBAJgBAKABAaABBrABAMABAQ&gs_ivs=1&sclient=gws-wiz#tts=0")

            elif 'spotify' in query:
                speak("opening spotify")
                codePath = "C:\\Users\\HP\\AppData\\Roaming\\Spotify\\Spotify.exe"
                os.startfile(codePath)

            elif 'play music' in query:
                music_dir = 'C:\\Users\\HP\\Downloads\\music'
                songs = os.listdir(music_dir)
                print(songs)
                os.startfile(os.path.join(music_dir, songs[0]))

            elif 'thank you' in query:
                speak("your welcome, anything else....")

            elif 'open camera' in query:
                speak('opening camera ')
                cam=cv2.VideoCapture(0)
                
                while(True):
                    ret,frame=cam.read()
                    cv2.imshow('frame',frame)
            
                    c=cv2.waitKey(10)
                    
                    if c==27:
                        break

                cam.release()
                cv2.destroyAllWindows()


            elif 'cab' in query:
                speak("where you want to go")
                speak("please tell me your pickup place")
                sen=takeCommand()
                key= 'UpAP0LMQsAHWeCkTykdBdFt12Zhwr2i2'
                url= 'http://www.mapquestapi.com/geocoding/v1/address?key='
                loc = sen
                main_url= url + key + '&location=' + loc
                r = requests.get(main_url)
                data = r.json()['results'][0]
                location=  data['locations'][0]
                lat1= str(location['latLng']['lat'])
                lon1= str(location['latLng']['lng'])
                print(lat1,lon1)
                speak("please tell me your destination place")
                sen2=takeCommand()
                key= 'UpAP0LMQsAHWeCkTykdBdFt12Zhwr2i2'
                url= 'http://www.mapquestapi.com/geocoding/v1/address?key='
                loc = sen2
                main_url= url + key + '&location=' + loc
                r = requests.get(main_url)
                data = r.json()['results'][0]
                location=  data['locations'][0]
                lat2= str(location['latLng']['lat'])
                lon2= str(location['latLng']['lng'])  
                print(lat2,lon2)
                #webbrowser.open("https://m.uber.com/looking?drop%5B0%5D=%7B%22latitude%22%3A"+lat2+"%2C%22longitude%22%3A"+lon2+"%2C%22addressLine1%22%3A%22"+sen2+"%22%2C%22id%22%3A%22ChIJs5JWhYHAwjsRFLpMZjiERgs%22%2C%22provider%22%3A%22google_places%22%2C%22index%22%3A0%7D&pickup=%7B%22latitude%22%3A"+lat1+"%2C%22longitude%22%3A"+lon1+"%2C%22addressLine1%22%3A%22"+sen+"%22%2C%22id%22%3A%22El9BbWJhIE1hdGEgTWFuZGlyLSBSYWogQmFzZXJhIFNvY2lldHkgUmQsIFJhamFzIFNvY2lldHksIEthdHJhaiwgUHVuZSwgTWFoYXJhc2h0cmEgNDExMDQ2LCBJbmRpYSIuKiwKFAoSCQ9UL-Xo6sI7ERBfpLxYuGESEhQKEglDaEYl6OrCOxEIKSHslVEp-A%22%2C%22provider%22%3A%22google_places%22%2C%22index%22%3A0%7D&vehicle=2032")
                #webbrowser.open("https://m.uber.com/looking?drop=latitude%22%3A"+lat2+"%2C%22longitude%22%3A"+lon2+"%2C%22addressLine1%22%3A%22"+sen2+"2C%22provider%22%3A%22google_places%22%&pickup=latitude"+lat1+"%2C%22longitude"+lon1+"%2C%22addressLine1%22%3A%22"+sen+"%22%2C%22id%22%%22provider%22%3A%22google_places%22%2C%22index%22%3A0%7D&vehicle=2032")
                webbrowser.open("https://m.uber.com/looking?drop%5B0%5D=%7B%22latitude%22%3A"+lat2+"%2C%22longitude%22%3A"+lon2+"%2C%22addressLine1%22%3A%22"+sen2+"%22%2C%22id%22%3A%22ChIJs5JWhYHAwjsRFLpMZjiERgs%22%2C%22provider%22%3A%22google_places%22%2C%22index%22%3A0%7D&pickup=%7B%22latitude%22%3A"+lat1+"%2C%22longitude%22%3A"+lon1+"%2C%22addressLine1%22%3A%22"+sen+"%22%2C%22id%22%3A%22El9BbWJhIE1hdGEgTWFuZGlyLSBSYWogQmFzZXJhIFNvY2lldHkgUmQsIFJhamFzIFNvY2lldHksIEthdHJhaiwgUHVuZSwgTWFoYXJhc2h0cmEgNDExMDQ2LCBJbmRpYSIuKiwKFAoSCQ9UL-Xo6sI7ERBfpLxYuGESEhQKEglDaEYl6OrCOxEIKSHslVEp-A%22%2C%22provider%22%3A%22google_places%22%2C%22index%22%3A0%7D&vehicle=2032")

            elif 'you can rest now' in query:
                speak("Thank you, if you want any help please call me")
                sys.exit()
                break
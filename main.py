from skimage.metrics import structural_similarity
import cv2
import pyaudio
import wave
import audioop
import math
from collections import deque
from vidstream import StreamingServer
import threading
import speech_recognition as sr  
import wave
import os          
import speech_recognition as sr
import time


count = 0
terminate = False

class FaceComparision:

    #Works well with images of different dimensions
    def orb_sim(img1, img2):
        # SIFT is no longer available in cv2 so using ORB
        orb = cv2.ORB_create()

        # detect keypoints and descriptors
        _, desc_a = orb.detectAndCompute(img1, None)
        _, desc_b = orb.detectAndCompute(img2, None)

        # define the bruteforce matcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        #perform matches. 
        matches = bf.match(desc_a, desc_b)
        #Look for similar regions with distance < 50. Goes from 0 to 100 so pick a number between.
        similar_regions = [i for i in matches if i.distance < 50]  
        if len(matches) == 0:
            return 0
        return len(similar_regions) / len(matches)

    #Needs images to be same dimensions
    def structural_sim(img1, img2):

        sim, _ = structural_similarity(img1, img2, full=True)
        return sim

    def result(orb_similarity, ssim):
        if orb_similarity < 0.4 and ssim < 0.4:
            count += 1
        print("The orb_similarity is ", orb_similarity)
        print("The ssim is ", ssim)

    def execute():
        img00 = cv2.imread('profile.jpeg', 0)
        img01 = cv2.imread('duplicate.jpeg', 0)

        orb_similarity = FaceComparision.orb_sim(img00, img01)  #1.0 means identical. Lower = not similar
        #Resize for SSIM
        # from skimage.transform import resize
        # img5 = resize(img, (img1.shape[0], img1.shape[1]), anti_aliasing=True, preserve_range=True)

        ssim = FaceComparision.structural_sim(img00, img01) #1.0 means identical. Lower = not similar

        FaceComparision.result(orb_similarity, ssim)
    

class VoiceRecognization:
    def record_on_detect(file_name, silence_limit=4, silence_threshold=1000, chunk=1024, rate=44100, prev_audio=1):
        CHANNELS = 2
        FORMAT = pyaudio.paInt16

        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(2),
                        channels=CHANNELS,
                        rate=rate,
                        input=True,
                        output=False,
                        frames_per_buffer=chunk)

        listen = True
        started = False
        rel = rate/chunk
        frames = []

        prev_audio = deque(maxlen=int(prev_audio * rel))
        slid_window = deque(maxlen=int(silence_limit * rel))

        while listen:
            data = stream.read(chunk)
            slid_window.append(math.sqrt(abs(audioop.avg(data, 4))))

            if(sum([x > silence_threshold for x in slid_window]) > 0):
                if(not started):
                    print("Starting record of phrase")
                    started = True
            elif (started is True):
                started = False
                listen = False
                prev_audio = deque(maxlen=int(0.5 * rel))

            if (started is True):
                frames.append(data)
            else:
                prev_audio.append(data)

        stream.stop_stream()
        stream.close()

        p.terminate()


        wf = wave.open(f'{file_name}.wav', 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(rate)

        wf.writeframes(b''.join(list(prev_audio)))
        wf.writeframes(b''.join(frames))
        wf.close()
    
    def convert():
            
        sound = 'example.wav'
        r = sr.Recognizer()
            
        with sr.AudioFile(sound) as source:
            r.adjust_for_ambient_noise(source)
            print("Converting Audio To Text and saving to file.. . ") 
            audio = r.listen(source)
            
        try:
            value = r.recognize_google(audio)            # API call to google for speech recognition
            #os.remove(sound)      Deletes the audio file
            print(value)
            if str is bytes: 
                result = u"{}".format(value).encode("utf-8")
            else: 
                result = "{}".format(value)

                
            with open("test.txt","w") as f:
                f.write(result)
                f.write(" ")
                f.close()
                    
        except sr.UnknownValueError:
            print("")
        except sr.RequestError as e:
                print("{0}".format(e))
        except KeyboardInterrupt:
            pass                
        from nltk.corpus import stopwords 
        from nltk.tokenize import word_tokenize 

         
        file = open("test.txt") ## Student speech file
        data = file.read()
        file.close()
        stop_words = set(stopwords.words('english'))   
        word_tokens = word_tokenize(data) ######### tokenizing sentence
        filtered_sentence = [w for w in word_tokens if not w in stop_words]  
        filtered_sentence = [] 
        
        for w in word_tokens:   ####### Removing stop words
            print(w)
            if w not in stop_words: 
                filtered_sentence.append(w) 
                                
        ##### checking whether proctor needs to be alerted or not
        file = open("question_paper.txt") ## Question file
        data = file.read()
        file.close()
        stop_words = set(stopwords.words('english'))   
        word_tokens = word_tokenize(data) ######### tokenizing sentence
        filtered_questions = [w for w in word_tokens if not w in stop_words]  
        filtered_questions = [] 
        
        for w in word_tokens:   ####### Removing stop words
            if w not in stop_words: 
                filtered_questions.append(w) 
                
        def common_member(a, b):     
            a_set = set(a) 
            b_set = set(b) 
            
            # check length  
            if len(a_set.intersection(b_set)) > 0: 
                return(a_set.intersection(b_set))   
            else: 
                return([]) 

        comm = common_member(filtered_questions, filtered_sentence)
        print('Number of common words spoken by test taker:', len(comm))
        print(comm)
            
        print("Done")
        if(len(comm)):
            print("Stop cheating...")
            os.remove('test.txt')
        else:
            os.remove('example.wav')
            print("You are making noise..")
            os.remove("test.txt")  

    def execute():
        VoiceRecognization.record_on_detect('example')
        VoiceRecognization.convert()

class ObjectDetection:
    def img_cap():
        videoCaptureObject = cv2.VideoCapture(0)
        result = True
        while(result):
            ret,frame = videoCaptureObject.read()
            cv2.imwrite("NewPicture.png",frame)
            result = False
        videoCaptureObject.release()
        cv2.destroyAllWindows()
        return


    def detect_img():
        img = cv2.imread('NewPicture.png')
        className= ['person','laptop','cellphone']
        classFile = 'coco.names'
        items = []
        with open(classFile,'rt') as f:
            className = f.read().rstrip('\n').split('\n')
        configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightsPath = 'frozen_inference_graph.pb'
        
        net = cv2.dnn_DetectionModel(weightsPath,configPath)
        net.setInputSize(320,320)
        net.setInputScale(1.0/ 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        classIds, confs, bbox = net.detect(img,confThreshold=0.5)
        # print(classIds,bbox)
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            cv2.putText(img,className[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            items.append(className[classId-1].upper())
        # cv2.imshow("Output",img)
        print(items)
        cv2.waitKey(0)
        return
    
    def execute():
        ObjectDetection.img_cap()
        ObjectDetection.detect_img()

# class Error:
#     def __init__():
#         print("Due to too many suspecious acitivities your exam has been terminated.")

# class Success:
#     def __init__():
#         print("Your exam has been sucessfully submited.")


if __name__ == '__main__':
    fac = FaceComparision
    obj = ObjectDetection
    voic = VoiceRecognization
    obj.execute()
    fac.execute() 
    
    voic.execute()
    
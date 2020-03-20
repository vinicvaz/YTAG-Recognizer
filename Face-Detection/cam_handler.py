import face_recognition
import cv2
from datetime import datetime, timedelta
import numpy as np
import face_handler as face_handler
from face_handler import *




def main_loop():
    
    ## Start video capture, 0 is default for webcam.
    cap = cv2.VideoCapture(0)
    
    number_of_faces_since_save = 0
    
    while True:
        
        ret, frame = cap.read()
        
        ## Resize image to 1/4 for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        ## OpenCV uses BGR color and face_recognition uses RGB color, so we have to convert it
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        ## Can also just invert the array order with a slice --> [:, :, ::-1]
        
        # Find all the face locations and face encodings in the current frame of video
        ## Returns an array of bounding boxes of human faces in a image
        face_locations = face_recognition.face_locations(rgb_small_frame) 
        ## Given an image, return the 128-dimension face encoding for each face in the image.
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        # Loop through each detected face and see if it is one we have seen before
        # If so, we'll give it a label that we'll draw on top of the video.
        face_labels = []
        for location, encoding in zip(face_locations, face_encodings):
            metadata = lookup_known_faces(encoding)
        
            if metadata is not None:
                time_at_door = datetime.now() - metadata['first_seen_this_interaction']
                face_label = f"Time: {int(time_at_door.total_seconds())}s"
            else:
                face_label = "New visitor!"
            
                # Grab the image of the the face from the current frame of video
                top, right, bottom, left = location
                face_image = small_frame[top:bottom, left:right]
                face_image = cv2.resize(face_image, (150, 150))

                # Add the new face to our known face data
                register_new_face(encoding, face_image)
                
            face_labels.append(face_label)
        
        for (top,right,bottom,left), face_label in zip(face_locations, face_labels):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, face_label, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
        
        number_of_recent_visitors = 0
        for metadata in face_handler.known_face_metadata:

            # If we have seen this person in the last minute, draw their image
            if (datetime.now() - metadata['last_seen'] < timedelta(seconds=10)) and (metadata['seen_frames'] > 5):
                # Draw the known face image
                x_position = number_of_recent_visitors * 150
                frame[30:180, x_position:x_position + 150] = metadata["face_image"]
                number_of_recent_visitors += 1
                
                # Label the image with how many times they have visited
                visits = metadata['seen_count']
                visit_label = f"{visits} visits"
                if visits==1:
                    visit_label = 'First Visit'
                cv2.putText(frame, visit_label, (x_position + 10, 170), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
                
        if number_of_recent_visitors > 0:
            cv2.putText(frame, "Visitors:", (5, 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 0), 1)
            
        
        # Display the final frame of video with boxes drawn around each detected fames
        cv2.imshow('Video', frame)
        
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            save_known_faces()
            break
            
        # We need to save our known faces back to disk every so often in case something crashes.
        if len(face_locations) > 0 and number_of_faces_since_save > 100:
            print('saving')
            save_known_faces()
            number_of_faces_since_save = 0
        else:
            number_of_faces_since_save += 1
            
            
    # Release handle to the webcam
    cap.release()
    cv2.destroyAllWindows()  
        

        
if __name__ == "__main__":
    load_known_faces()
    main_loop()
import numpy as np 
import cv2 
import time 

cap = cv2.VideoCapture(0) 
prev_frame_time = 0
new_frame_time = 0

while(cap.isOpened()): 
	ret, frame = cap.read() 
	if not ret: 
		break
	frame = cv2.resize(frame, (200, 200)) 
	new_frame_time = time.time() 

	fps = 1/(new_frame_time-prev_frame_time) 
	prev_frame_time = new_frame_time 

	fps = int(fps) 	# chuyển đổi fps thành số nguyên
	fps = str(fps) 	# chuyển đổi fps thành chuỗi để chúng tôi có thể hiển thị nó trên khung	bằng hàm putText

	# puting the FPS count on the frame 
	cv2.putText(frame, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX , 3, (100, 255, 0), 3, cv2.LINE_AA) 
	cv2.imshow('frame', frame) 
	if cv2.waitKey(1) & 0xFF == ord('q'): 
		break

cap.release() 
cv2.destroyAllWindows() 


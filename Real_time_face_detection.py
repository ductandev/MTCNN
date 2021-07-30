import cv2
import time
import tensorflow as tf
import detect_face
import numpy as np

def video_init(is_2_write=False,save_path=None):
	writer = None
	cap = cv2.VideoCapture(0)
	height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)			#default 480
	width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)			#default 640

	# width = 480
	# height = 640
	# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
	# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

	'''
	ref:https://docs.opencv.org/master/dd/d43/tutorial_py_video_display.html
	FourCC is a 4-byte code used to specify the video codec. 
	In Fedora: DIVX, XVID, MJPG, X264, WMV1, WMV2. (XVID is more preferable.) MJPG results in high size video. X264 gives very small size video)
	'''

	if is_2_write is True:
		#fourcc = cv2.VideoWriter_fourcc('x', 'v', 'i', 'd')
		#fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
		fourcc = cv2.VideoWriter_fourcc(*'divx')
		if save_path is None:
			save_path = 'demo.avi'
		writer = cv2.VideoWriter(save_path, fourcc, 30, (int(width), int(height)))

	return cap,height,width,writer

def face_detection_MTCNN(detect_multiple_faces=False):
	#----var
	frame_count = 0
	FPS = "Initialing"
	no_face_str = "No faces detected"

	#----video streaming init
	cap, height, width, writer = video_init(is_2_write=False)

	#----MTCNN init
	color = (0,255,0)				# màu khung ( 0, xanh lá,0)
	minsize = 20  					# minimum size of face
	threshold = [0.6, 0.7, 0.7]  	# three steps's threshold
	factor = 0.709  				# scale factor : hệ số tỉ lệ
	with tf.Graph().as_default():
		config = tf.ConfigProto(log_device_placement=True,
								allow_soft_placement=True,
								)
		config.gpu_options.allow_growth = True
		# config.gpu_options.per_process_gpu_memory_fraction = 0.7
		sess = tf.Session(config=config)
		with sess.as_default():
			pnet, rnet, onet = detect_face.create_mtcnn(sess, None)


	while (cap.isOpened()):
		#----get image
		ret, img = cap.read()
		if ret is True:
			#----image processing: 							# XỬ LÝ HÌNH ẢNH
			img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)	# Chuyển ảnh từ BGR sang RGB
			#----face detection: 							# PHÁT HIỆN KHUÔN MẶT
			bounding_boxes, points = detect_face.detect_face(img_rgb, minsize, pnet, rnet, onet, threshold, factor) #bounding_boxes: khung hộp giới hạn
			#print(bounding_boxes)
			#----bounding boxes processing
			nrof_faces = bounding_boxes.shape[0]			# nrof_faces: thu hẹp khuôn mặt # shape: khuôn, mẫu, hình, làm khuôn, tạo hình
			#print(bounding_boxes.shape)					# trả về ma trận (1x5) về 1 nếu có khuôn mặt , ngược lại bằng 0
			if nrof_faces > 0:
				points = np.array(points)
				points = np.transpose(points, [1, 0])
				points = points.astype(np.int16)

				det = bounding_boxes[:, 0:4]
				det_arr = []
				img_size = np.asarray(img.shape)[0:2]
				if nrof_faces > 1:
					if detect_multiple_faces:
						for i in range(nrof_faces):
							det_arr.append(np.squeeze(det[i]))
					else:
						bounding_box_size = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
						img_center = img_size / 2
						offsets = np.vstack(
							[(det[:, 0] + det[:, 2]) / 2 - img_center[1], (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
						offset_dist_squared = np.sum(np.power(offsets, 2.0), 0)
						index = np.argmax(
							bounding_box_size - offset_dist_squared * 2.0)  # some extra weight on the centering
						det_arr.append(det[index, :])
				else:
					det_arr.append(np.squeeze(det))

				det_arr = np.array(det_arr)
				det_arr = det_arr.astype(np.int16)

				for i, det in enumerate(det_arr):
					#det = det.astype(np.int32)
					cv2.rectangle(img, (det[0],det[1]), (det[2],det[3]), color, 2)

					#----draw 5 point on the face	
					facial_points = points[i]
					#print (facial_points)	#x1, x2, x3, x4, x5, y1, y2, y3, y4, y5;
					print ( "\nMắt phải:   ",facial_points[0],facial_points[5],
							"\nMắt trái:   ",facial_points[1],facial_points[6],
							"\nMũi:        ",facial_points[2],facial_points[7],
							"\nMiệng phải: ",facial_points[3],facial_points[8],
							"\nMiệng trái: ",facial_points[4],facial_points[9])						# in ra 5 điểm khuôn mặt
					for j in range(0,5):															# Vòng lặp 5 lần ( j chạy từ 0 đến 4)
						#cv2.circle																	# Vẽ hình tròn có tâm
						#print("------",j)															# (j=0,1,2,3,4)
						cv2.circle(img, (facial_points[j], facial_points[j + 5]), 2, (0, 0, 255), -1, 1) #cv2.circle (hình ảnh, tọa độ trung tâm của vòng tròn x y, bán kính, màu sắc, độ dày)
						#print("A", j, facial_points[j])
						#print("B", j, facial_points[j + 5])
						
			# ----no faces detected
			else:
				cv2.putText(img, no_face_str, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

			#----FPS count
			if frame_count == 0:
				t_start = time.time()
			frame_count += 1
			print(frame_count)
			if frame_count >= 10:
				print("time.time() = ",time.time())
				print("t_start     = ",t_start)
				print("time.time() - t_start = ",time.time() - t_start)
				FPS = "FPS=%1f" % (10 / (time.time() - t_start))
				print(FPS)
				frame_count = 0

			# cv2.putText
			cv2.putText(img, FPS, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

			#----image display
			cv2.imshow("Result of MTCNN", img)

			#----image writing
			if writer is not None:
				writer.write(img)

			#----'q' key pressed
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			print("get image failed")
			break

	#----release
	cap.release()
	if writer is not None:
		writer.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
    face_detection_MTCNN(detect_multiple_faces=False) #detect multiple faces: phát hiện nhiều khuôn mặt


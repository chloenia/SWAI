from flask import Flask, render_template, request, jsonify
import cv2
from pathlib import Path
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Define body parts and pose pairs
BODY_PARTS = {"Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
              "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
              "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
              "Background": 15}

POSE_PAIRS = [["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
              ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
              ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
              ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"]]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        logger.debug("Received request to process video")

        # Open the camera
        capture = cv2.VideoCapture(0)

        BASE_DIR = Path(__file__).resolve().parent
        protoFile = str(BASE_DIR / "/Users/chloenia/Downloads/deploy-663af4ba917836abf09d00ce/templates/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
        weightsFile = str(BASE_DIR / "/Users/chloenia/Downloads/deploy-663af4ba917836abf09d00ce/templates/openpose/models/pose/mpi/pose_iter_160000.caffemodel")


        # Debugging: Try loading the network model
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


        inputWidth = 320
        inputHeight = 240
        inputScale = 1.0 / 255

        # Output video format
        output_file = "output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 20.0
        frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)


        while True:
            has_frame, frame = capture.read()

            if not has_frame:
                logger.debug("No frame captured, exiting loop")
                break

            frame_width = frame.shape[1]
            frame_height = frame.shape[0]

            inp_blob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
            net.setInput(inp_blob)
            output = net.forward()

            # Key point detection
            points = []
            for i in range(0, 15):
                prob_map = output[0, i, :, :]
                min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)
                x = (frame_width * point[0]) / output.shape[3]
                y = (frame_height * point[1]) / output.shape[2]
                if prob > 0.1:
                    points.append((int(x), int(y)))
                else:
                    points.append(None)

            for pair in POSE_PAIRS:
                part_a = pair[0]
                part_a = BODY_PARTS[part_a]
                part_b = pair[1]
                part_b = BODY_PARTS[part_b]
                if points[part_a] and points[part_b]:
                    cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 2)

            cv2.imshow("Output-Keypoints", frame)

            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.debug("Exit key pressed, exiting loop")
                break

        capture.release()
        out.release()
        cv2.destroyAllWindows()

        logger.debug("Video processing complete")
        return jsonify({"video_url": f"/{output_file}"})
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)



# from flask import Flask, render_template, request, jsonify
# import cv2
# from pathlib import Path

# app = Flask(__name__)

# # MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
# BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                 "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                 "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
#                 "Background": 15 }

# POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
#                 ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
#                 ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
#                 ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]


# #쿠다 사용 안하면 밑에 이미지 크기를 줄이는게 나을 것이다
# # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #벡엔드로 쿠다를 사용하여 속도향상을 꾀한다
# # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # 쿠다 디바이스에 계산 요청


# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_video', methods=['POST'])
# def process_video():
#     print("Received request to process video")

#     # 각 파일 path
#     BASE_DIR = Path(__file__).resolve().parent
#     protoFile = str(BASE_DIR / "/Users/chloenia/Downloads/deploy-663af4ba917836abf09d00ce/templates/openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
#     weightsFile = str(BASE_DIR / "/Users/chloenia/Downloads/deploy-663af4ba917836abf09d00ce/templates/openpose/models/pose/mpi/pose_iter_160000.caffemodel")

#     # 위의 path에 있는 network 모델 불러오기
#     net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#     # 카메라 캡처
#     capture = cv2.VideoCapture(0)

#     inputWidth = 320
#     inputHeight = 240
#     inputScale = 1.0 / 255

#     # 비디오 출력 설정
#     output_file = "static/output_video.mp4"
    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = 20.0
#     frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)


#     while True:
#         hasFrame, frame = capture.read()
#         if not hasFrame:
#             break

#         frameWidth = frame.shape[1]
#         frameHeight = frame.shape[0]

#         inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
#         net.setInput(inpBlob)
#         output = net.forward()

#         points = []
#         for i in range(0, 15):
#             probMap = output[0, i, :, :]
#             minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
#             x = (frameWidth * point[0]) / output.shape[3]
#             y = (frameHeight * point[1]) / output.shape[2]
#             if prob > 0.1:
#                 points.append((int(x), int(y)))
#             else:
#                 points.append(None)

#         for pair in POSE_PAIRS:
#             partA = pair[0]
#             partA = BODY_PARTS[partA]
#             partB = pair[1]
#             partB = BODY_PARTS[partB]
#             if points[partA] and points[partB]:
#                 cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

#         out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture.release()
#     out.release()
#     cv2.destroyAllWindows()

#     print(f"Video file saved to {output_file}")

#     return jsonify({"video_url": "/" + output_file})

# if __name__ == '__main__':
#     app.run(debug=True, port=5001)


# from flask import Flask, render_template, request, jsonify
# import cv2
# from pathlib import Path

# app = Flask(__name__)

# # MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
# BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                 "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                 "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
#                 "Background": 15 }

# POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
#                 ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
#                 ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
#                 ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_video', methods=['POST'])
# def process_video():
#     # 카메라 캡처
#     capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)
#     # 비디오 출력 설정
#     output_file = "output_video.mp4"
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     fps = 20.0
#     frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
#     out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

#     # 모델 경로
#     BASE_DIR = Path(__file__).resolve().parent
#     protoFile = str(BASE_DIR / "openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
#     weightsFile = str(BASE_DIR / "openpose/models/pose/mpi/pose_iter_160000.caffemodel")

#     # OpenPose 모델 불러오기
#     net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
#     inputWidth=320;
#     inputHeight=240;
#     inputScale=1.0/255;

#     while True:
#         hasFrame, frame = capture.read()

#         if not hasFrame:
#             break

#         frameWidth = frame.shape[1]
#         frameHeight = frame.shape[0]

#         inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
#         net.setInput(inpBlob)
#         output = net.forward()

#         points = []
#         for i in range(0, 15):
#             probMap = output[0, i, :, :]
#             minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
#             x = (frameWidth * point[0]) / output.shape[3]
#             y = (frameHeight * point[1]) / output.shape[2]
#             if prob > 0.1:
#                 points.append((int(x), int(y)))
#             else:
#                 points.append(None)

#         for pair in POSE_PAIRS:
#             partA = pair[0]
#             partA = BODY_PARTS[partA]
#             partB = pair[1]
#             partB = BODY_PARTS[partB]
#             if points[partA] and points[partB]:
#                 cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)

#         cv2.imshow("Output-Keypoints", frame)

#         out.write(frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     capture.release()
#     out.release()
#     cv2.destroyAllWindows()

#     return jsonify({"video_url": output_file})

# if __name__ == '__main__':
#     app.run(debug=True)



# import cv2
# from pathlib import Path

# # MPII에서 각 파트 번호, 선으로 연결될 POSE_PAIRS
# BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
#                 "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
#                 "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
#                 "Background": 15 }

# POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
#                 ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
#                 ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
#                 ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]

# # 각 파일 path
# BASE_DIR = Path(__file__).resolve().parent
# protoFile = str(BASE_DIR / "openpose/models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt")
# weightsFile = str(BASE_DIR / "openpose/models/pose/mpi/pose_iter_160000.caffemodel")


# # 위의 path에 있는 network 모델 불러오기
# net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# #쿠다 사용 안하면 밑에 이미지 크기를 줄이는게 나을 것이다
# # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) #벡엔드로 쿠다를 사용하여 속도향상을 꾀한다
# # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA) # 쿠다 디바이스에 계산 요청


# ###카메라랑 연결...?
# capture = cv2.VideoCapture(0) #카메라 정보 받아옴
# # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #카메라 속성 설정
# # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # width:너비, height: 높이

# inputWidth=320;
# inputHeight=240;
# inputScale=1.0/255;

 
#  #output 비디오 포멧
# output_file = "output_video.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# fps = 20.0
# frame_size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

 
#  #반복문을 통해 카메라에서 프레임을 지속적으로 받아옴
# while cv2.waitKey(1) <0:  #아무 키나 누르면 끝난다.
#     #웹캠으로부터 영상 가져옴
#     hasFrame, frame = capture.read()  
    
#     #영상이 커서 느리면 사이즈를 줄이자
#     #frame=cv2.resize(frame,dsize=(320,240),interpolation=cv2.INTER_AREA)
    
#     #웹캠으로부터 영상을 가져올 수 없으면 웹캠 중지
#     if not hasFrame:
#         cv2.waitKey()
#         break
    
#     # 
#     frameWidth = frame.shape[1]
#     frameHeight = frame.shape[0]
    
#     inpBlob = cv2.dnn.blobFromImage(frame, inputScale, (inputWidth, inputHeight), (0, 0, 0), swapRB=False, crop=False)
    
#     imgb=cv2.dnn.imagesFromBlob(inpBlob)
#     #cv2.imshow("motion",(imgb[0]*255.0).astype(np.uint8))
    
#     # network에 넣어주기
#     net.setInput(inpBlob)

#     # 결과 받아오기
#     output = net.forward()


#     # 키포인트 검출시 이미지에 그려줌
#     points = []
#     for i in range(0,15):
#         # 해당 신체부위 신뢰도 얻음.
#         probMap = output[0, i, :, :]
    
#         # global 최대값 찾기
#         minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

#         # 원래 이미지에 맞게 점 위치 변경
#         x = (frameWidth * point[0]) / output.shape[3]
#         y = (frameHeight * point[1]) / output.shape[2]

#         # 키포인트 검출한 결과가 0.1보다 크면(검출한곳이 위 BODY_PARTS랑 맞는 부위면) points에 추가, 검출했는데 부위가 없으면 None으로    
#         if prob > 0.1 :    
#             cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED) # circle(그릴곳, 원의 중심, 반지름, 색)
#             cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
#             points.append((int(x), int(y)))
#         else :
#             points.append(None)
    

#     # 각 POSE_PAIRS별로 선 그어줌 (머리 - 목, 목 - 왼쪽어깨, ...)
#     for pair in POSE_PAIRS:
#         partA = pair[0]             # Head
#         partA = BODY_PARTS[partA]   # 0
#         partB = pair[1]             # Neck
#         partB = BODY_PARTS[partB]   # 1
        
#         #partA와 partB 사이에 선을 그어줌 (cv2.line)
#         if points[partA] and points[partB]:
#             cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)
    
#     cv2.imshow("Output-Keypoints",frame)
    
#     out.write(frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
 
# capture.release()  #카메라 장치에서 받아온 메모리 해제
# cv2.destroyAllWindows() #모든 윈도우 창 닫음
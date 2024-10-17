import cv2
import onnxruntime
import numpy as np

INPUT_WIDTH = 512
INPUT_HEIGHT = 384
segmentation_colors = np.array([[0,    0,    0],
                                [255,  191,  0],
                                [192,  67,   251]], dtype=np.uint8)

def prepare_input(image):
    input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_img = cv2.resize(input_img, (INPUT_WIDTH, INPUT_HEIGHT))

    # Normalize in-place for efficiency
    input_img = input_img / 255.0
    input_img -= np.array([0.485, 0.456, 0.406])
    input_img /= np.array([0.229, 0.224, 0.225])

    input_img = input_img.transpose(2, 0, 1)
    input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

    return input_tensor

def inference(session, input_tensor):
    # Get model inputs/outputs
    model_inputs = session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    model_outputs = session.get_outputs()
    output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    # Run inference
    outputs = session.run(output_names, {input_names[0]: input_tensor})

    return outputs, output_names

def draw_seg(seg_map, image, alpha=0.5):
    color_segmap = cv2.resize(image, (seg_map.shape[1], seg_map.shape[0]))
    color_segmap[seg_map > 0] = segmentation_colors[seg_map[seg_map > 0]]

    color_segmap = cv2.resize(color_segmap, (image.shape[1], image.shape[0]))

    if alpha == 0:
        combined_img = np.hstack((image, color_segmap))
    else:
        combined_img = cv2.addWeighted(image, alpha, color_segmap, (1 - alpha), 0)

    return combined_img

# Initialize video and ONNX session
cap = cv2.VideoCapture("video7.mp4")
model_path = "models/hybridnets_384x512.onnx"
session = onnxruntime.InferenceSession(model_path)

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 20, (1280, 720))
cv2.namedWindow("Road Detections", cv2.WINDOW_NORMAL)

frame_count = 0
skip_frames = 2  # Process every 2nd frame

while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    ret, frame = cap.read()
    if not ret:
        break

    # Skip frames to speed up processing
    if frame_count % skip_frames == 0:
        new_frame = prepare_input(frame)
        outputs, output_names = inference(session, new_frame)
        out_seg = outputs[output_names.index("segmentation")]
        seg_map = np.squeeze(np.argmax(out_seg, axis=1))
        seg_img = draw_seg(seg_map, frame)

        cv2.imshow("Road Detections", seg_img)
        out.write(seg_img)

    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()

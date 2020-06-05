import argparse
parser = argparse.ArgumentParser(description='person detection v1.0')
parser.add_argument("input", help="Input image file name")
parser.add_argument("-s","--save", help="Save processed image",action="store_true")
parser.add_argument("-m","--mode", help="Select mode 1-image, 2-video",default="1")
args = parser.parse_args()

print("person detection v1.0, loading..")
import numpy as np
import tensorflow.compat.v1 as tf
import cv2
import time
import imutils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def processFrame(self, image):
        image_np_expanded = np.expand_dims(image, axis=0)
        start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        end_time = time.time()

        print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
			int(boxes[0,i,1]*im_width),
			int(boxes[0,i,2] * im_height),
			int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":
    model_path = './model.pb'
    odapi = DetectorAPI(path_to_ckpt=model_path)
    threshold = 0.7

    if int(args.mode) == 1:
        print('image mode')
        image = cv2.imread(args.input)
        image = imutils.resize(image,width=720)
        boxes, scores, classes, num = odapi.processFrame(image)

        for i in range(len(boxes)):
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(134,235,52),2)
                cv2.rectangle(image, (box[1],box[0]-30),(box[1]+125,box[0]),(134,235,52), thickness=cv2.FILLED)
                cv2.putText(image, '  Person '+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)

        cv2.imshow("preview", image)
        if args.save:
            print("saving...")
            cv2.imwrite("processed-"+args.input,image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    if int(args.mode) == 2:
        cap = cv2.VideoCapture(args.input)
        while True:
            r, image = cap.read()
            image = imutils.resize(image,width=720)
            boxes, scores, classes, num = odapi.processFrame(image)
            for i in range(len(boxes)):
                if classes[i] == 1 and scores[i] > threshold:
                    box = boxes[i]
                    cv2.rectangle(image,(box[1],box[0]),(box[3],box[2]),(134,235,52),2)
                    cv2.rectangle(image, (box[1],box[0]-30),(box[1]+125,box[0]),(134,235,52), thickness=cv2.FILLED)
                    cv2.putText(image, '  Person '+str(round(scores[i],2)), (box[1],box[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225,255,225), 1)
            cv2.imshow("preview", image)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

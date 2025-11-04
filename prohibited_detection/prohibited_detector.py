import onnxruntime
import numpy as np
import cv2

class ProhibitedDetector:
    def __init__(self, model_path = "best.onnx"):   
        self.class_names = [
            "baton",
            "pliers",
            "hammer",
            "powerbank",
            "scissors",
            "wrench",
            "gun",
            "bullet",
            "sprayer",
            "handcuffs",
            "knife",
            "lighter"
        ]

        self.init_model(model_path)

    def init_model(self, model_path):
        self.ort_session = onnxruntime.InferenceSession(model_path)
        self.ort_session.set_providers(["CUDAExecutionProvider", "CPUExecutionProvider"])

        self.get_input_details()

    def get_input_details(self):
        model_inputs = self.ort_session.get_inputs()

        self.input_width = model_inputs[0].shape[3]
        self.input_height = model_inputs[0].shape[2]
        self.input_name = model_inputs[0].name

    def letterbox(self, image):
        h, w = image.shape[:2]
        new_w, new_h = self.input_width, self.input_height

        scale = min(new_w / w, new_h / h)

        new_unpad_w = int(round(w * scale))
        new_unpad_h = int(round(h * scale))
        new_unpad = (new_unpad_w, new_unpad_h)

        padding_w = new_w - new_unpad_w
        padding_h = new_h - new_unpad_h

        padding_w /= 2
        padding_h /= 2

        if (w, h) != new_unpad:
            image = cv2.resize(image, new_unpad, interpolation = cv2.INTER_LINEAR)

        top, bottom = int(round(padding_h - 0.1)), int(round(padding_h + 0.1))
        left, right = int(round(padding_w - 0.1)), int(round(padding_w + 0.1))
        padding = (top, bottom, left, right)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value = (114, 114, 114))

        return image, padding
    
    def preprocess(self, image):
        image, padding = self.letterbox(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis = 0)
        image = image.astype(np.float32)
        input_tensor = image / 255.0

        return input_tensor, padding
    
    def inference(self, input_tensor):
        results = self.ort_session.run(None, {self.input_name: input_tensor})

        return results
    
    def postprocess(self, results, padding, image_size, threshold):
        results = np.squeeze(results)

        postprocessed = []
        top, bottom, left, right = padding
        image_height, image_width = image_size

        for result in results:
            x0, y0, x1, y1, conf, class_id = result

            if conf < threshold:
                continue

            x0, y0, x1, y1 = x0 - left, y0 - top, x1 - left, y1 - top

            x0 /= (self.input_width - left - right)
            y0 /= (self.input_height - top - bottom)
            x1 /= (self.input_width - left - right)
            y1 /= (self.input_height - top - bottom)

            x0 *= image_width
            y0 *= image_height
            x1 *= image_width
            y1 *= image_height

            postprocessed.append((x0, y0, x1, y1, conf, class_id))

        return postprocessed
    
    def predict(self, image, conf = 0.5):
        input_tensor, padding = self.preprocess(image)
        results = self.inference(input_tensor)
        results = self.postprocess(results, padding, image.shape[:2], conf)

        detections = []

        for result in results:
            x0, y0, x1, y1, confidence, class_id = result
            class_name = self.class_names[int(class_id)]

            detections.append({
                "label": class_name,
                "confidence": f"{confidence:.4f}",
                "box": {
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1
                },
            })

        return detections
    
    def __call__(self, image, *, conf = 0.5):
        return self.predict(image, conf)
    
import cv2
import os
import numpy as np
from segment_anything import sam_model_registry, SamPredictor


def apply_mask(image, mask, alpha_channel=True): 
    if alpha_channel:
        alpha = np.zeros_like(image[..., 0])  
        alpha[mask == 1] = 255 
        image = cv2.merge((image[..., 0], image[..., 1], image[..., 2], alpha)) 
    else:
        image = np.where(mask[..., None] == 1, image, 0)
    return image
 

def apply_color_mask(image, mask, color, color_dark=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - color_dark) + color_dark * color[c], image[:, :, c])
    return image
 

def save_mask(mask, save_dir):
    save_name = os.path.join(save_dir, 'mask.png')
    cv2.imwrite(save_name , (mask * 255).astype(np.uint8))

def save_masked_image(image, mask, output_dir, filename, crop_mode_):  
    if crop_mode_:
        y, x = np.where(mask)
        y_min, y_max, x_min, x_max = y.min(), y.max(), x.min(), x.max()
        cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]
        cropped_image = image[y_min:y_max + 1, x_min:x_max + 1]
        masked_image = apply_mask(cropped_image, cropped_mask)
    else:
        masked_image = apply_mask(image, mask)
    filename = filename[:filename.rfind('.')] + '.png'
    new_filename = get_next_filename(output_dir, filename)
 
    if new_filename:
        if masked_image.shape[-1] == 4:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        else:
            cv2.imwrite(os.path.join(output_dir, new_filename), masked_image)
        print(f"Saved as {new_filename}")
    else:
        print("Could not save the image. Too many variations exist.")
 

input_point = []
input_label = []

def mouse_click(event, x, y, flags, param):
    global input_point, input_label
    if event == cv2.EVENT_LBUTTONDOWN:  
        input_point.append([x, y])
        input_label.append(1) 
    elif event == cv2.EVENT_RBUTTONDOWN: 
        input_point.append([x, y])
        input_label.append(0)




class InteractiveSAM:

    def __init__(self, ckpt='checkpoints\sam_vit_h_4b8939.pth') -> None:
        self.sam = sam_model_registry["vit_h"](checkpoint="checkpoints\sam_vit_h_4b8939.pth")
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)  


    def predict(self, filename='output\out.png', save_dir='output/0'):

        cv2.namedWindow("image")
        cv2.setMouseCallback("image", mouse_click)

        if not isinstance(filename, np.ndarray):
            image_orin = cv2.imread(filename)
            image = cv2.cvtColor(image_orin, cv2.COLOR_BGR2RGB)  
        else:
            image_orin = filename.copy()
            image = image_orin.copy()
        selected_mask = None
        logit_input = None

        while True:

            image_display = image_orin.copy()
            for point, label in zip(input_point, input_label):
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(image_display, tuple(point), 5, color, -1)
            if selected_mask is not None:
                color = tuple(np.random.randint(0, 256, 3).tolist())
                selected_image = apply_color_mask(image_display, selected_mask, color)
    
            cv2.imshow("image", image_display)
            key = cv2.waitKey(1)

            if key == ord("s"):


                if len(input_point) > 0 and len(input_label) > 0:
                    self.predictor.set_image(image)  
                    input_point_np = np.array(input_point)  
                    input_label_np = np.array(input_label) 
                    masks, scores, logits = self.predictor.predict(
                        point_coords=input_point_np,
                        point_labels=input_label_np,
                        mask_input=logit_input[None, :, :] if logit_input is not None else None,
                        multimask_output=True,
                    )

                color = tuple(np.random.randint(0, 256, 3).tolist())  
                image_select = image_orin.copy()
                selected_mask = masks[0]  
                selected_image = apply_color_mask(image_select, selected_mask, color)
                cv2.imshow("image", selected_image)                
                save_mask(selected_mask, save_dir)

            elif key==ord('q'):

                cv2.destroyAllWindows()
                return selected_mask


    

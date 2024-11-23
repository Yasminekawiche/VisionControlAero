#Imports
import os
from flask import Flask, jsonify, render_template ,  send_from_directory , request,redirect, url_for
from PIL import Image, ImageDraw, ImageFont
import logging  
import torch
from collections import OrderedDict
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import os, json, cv2, random
from matplotlib import pyplot as plt
from PIL import Image
from roboflow import Roboflow
from collections import defaultdict
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.utils.events import EventStorage
from detectron2.modeling import build_model
import detectron2.utils.comm as comm
from detectron2.engine import default_argument_parser, default_setup, default_writers, launch
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from PIL import Image
import numpy as np 
import matplotlib.pyplot as plt 
setup_logger()
app = Flask(__name__)


register_coco_instances("test", {}, "ALL References.v13i.coco/test/_annotations.coco.json", "ALL References.v13i.coco/test")
register_coco_instances("train", {}, "ALL References.v13i.coco/train/_annotations.coco.json", "ALL References.v13i.coco/train")
register_coco_instances("valid", {}, "ALL References.v13i.coco/valid/_annotations.coco.json", "ALL References.v13i.coco/valid")


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "weightss.pth" 
output_dir = 'C:/Users/HP/Desktop/stage hutchinson/my project/static'
output_path = os.path.join(output_dir, 'predicted_image.jpg')
predictor = DefaultPredictor(cfg)
dataset_dicts = DatasetCatalog.get("valid")
metadata = MetadataCatalog.get("train")

dataset_train = DatasetCatalog.get("train")
class_counts = defaultdict(int)
image_path = 'image.jpg'

im = cv2.imread(image_path)
im = cv2.resize(im, (640,480))
outputs = predictor(im)  
instances = outputs["instances"]
scores = instances.scores.cpu().numpy()
filtered_instances_ = []
class_colors = {
    'etiquette': 'red',
    'boite': 'blue',
    'Piece': 'green',
    'CROCHETS': 'purple',
    'Ouverture': 'purple',
}
target_classes = ['etiquette', 'boite', 'Piece', 'CROCHETS', 'Ouverture']

image_width = 640
image_height = 480
fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)
boite_position_message =''

##########################################################################################
def get_polygon_centroid(polygon):
    x = np.array(polygon[0::2])
    y = np.array(polygon[1::2])
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    return centroid_x, centroid_y

def get_filtered_instances(image, instances, metadata):
    filtered_indices = []
    class_counts = defaultdict(int)
    
    scores = instances.scores.cpu().numpy() 

    for i in range(len(instances)):
        class_id = instances.pred_classes[i].item()
        class_name = metadata.thing_classes[class_id]
        if class_name == 'CROCHETS' and scores[i] < 0.93:
            continue
        if class_name == 'boite' and scores[i] < 0.95:
            continue
        if class_name == 'Piece' and scores[i] < 0.95:
            continue
        if class_name == 'etiquette' and scores[i] < 0.95:
            continue
        filtered_instances_.append(i)
        class_counts[class_name] += 1

    filtered_instances = instances[filtered_indices]
    return filtered_instances, class_counts

def calculate_centroids_from_filtered_instances(image, instances, metadata):
    centroids = defaultdict(list)
    pred_classes = instances.pred_classes.cpu().numpy()
    pred_masks = instances.pred_masks.cpu().numpy()
    scores = instances.scores.cpu().numpy()

    #fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100), dpi=100)

    for i in range(len(instances)):
        class_id = pred_classes[i]
        class_name = metadata.thing_classes[class_id]
        color = class_colors[class_name] 


        if (class_name == 'CROCHETS' and scores[i] < 0.93) or \
        (class_name == 'boite' and scores[i] < 0.95) or \
        (class_name == 'Piece' and scores[i] < 0.95) or \
        (class_name == 'etiquette' and scores[i] < 0.95):
            continue 

        
        mask = pred_masks[i]
        if np.any(mask):  #

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                for contour in contours:
                    
                    contour = contour.flatten()
                    centroid_x, centroid_y = get_polygon_centroid(contour)

                    
                    centroids[class_name].append((centroid_x, centroid_y))

                    x = contour[0::2]
                    y = contour[1::2]
                    ax.plot(x, y, linewidth=2, color=color)
                    ax.plot(centroid_x, centroid_y, 'o', color=color)
                    ax.text(centroid_x, centroid_y, class_name, color=color, fontsize=12, ha='right')

    plt.axis('off')  
    plt.show()  
    return centroids

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

############################################distance######################################
def convert_pixels_to_cm(distance_pixels, resolution_ppp=96):
    return (distance_pixels / resolution_ppp) * 1.158299716157924

def draw_point(ax, point, color, label=None):
    ax.plot(point[0], point[1], 'o', color=color)
    if label:
        ax.text(point[0] + 5, point[1] - 5, label, color=color, fontsize=12, ha='center')

def draw_line(ax, point1, point2, color, label=None):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  
    if color is None:
        color = random.choice(colors)
    ax.plot([point1[0], point2[0]], [point1[1], point2[1]], color=color, linestyle='-', linewidth=2)
    if label:
        mid_point = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)
        ax.text(mid_point[0], mid_point[1], label, color=color, fontsize=12, ha='center')

def save_image_with_distances(image, sorted_crochets, distances, output_path):
    if image is None:
        raise ValueError("The image provided is None. Check if the image is loaded correctly.")

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    
    if len(sorted_crochets['top']) > 0:
        top_crochets = list(sorted_crochets['top'].values())
        for i, crochet in enumerate(top_crochets):
            draw_point(ax, crochet, 'purple', str(i + 1)) 
        for i in range(len(top_crochets) - 1):
            draw_line(ax, top_crochets[i], top_crochets[i + 1], 'purple', f'{distances["top"][i]:.3f} cm ')

    if len(sorted_crochets['bottom']) > 0:
        bottom_crochets = list(sorted_crochets['bottom'].values())
        for i, crochet in enumerate(bottom_crochets):
            draw_point(ax, crochet, 'orange', str(i + 1)) 
        for i in range(len(bottom_crochets) - 1):
            draw_line(ax, bottom_crochets[i], bottom_crochets[i + 1], 'orange', f'{distances["bottom"][i]:.3f} cm ')
    
    plt.axis('off') 
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def label_and_calculate_crochet_distances(image, instances, metadata):
    
    if image is None:
        raise ValueError("The image provided is None. Check if the image is loaded correctly.")
    

    centroids = calculate_centroids_from_filtered_instances(image, instances, metadata)


    labeled_crochets = {'top': [], 'bottom': []}
    

    if 'etiquette' in centroids and centroids['etiquette']:
        ticket_centroid = centroids['etiquette'][0] 
    
        crochets_centroids = centroids['CROCHETS']

        for crochet in crochets_centroids:
            if crochet[1] < ticket_centroid[1]: 
                labeled_crochets['bottom'].append(crochet)
            else: 
                labeled_crochets['top'].append(crochet)

        sorted_crochets = {}
        for region in ['top', 'bottom']:
            sorted_crochets[region] = sorted(labeled_crochets[region], key=lambda x: (x[0], -x[1]))  # Sort by y, then by x
            sorted_crochets[region] = {f"{i+1} {region}": point for i, point in enumerate(sorted_crochets[region])}

        
        distances = {'top': [], 'bottom': []}
        if len(sorted_crochets['top']) > 1:
            top_crochets = list(sorted_crochets['top'].values())
            for i in range(len(top_crochets) - 1):
                distance_pixels = calculate_distance(top_crochets[i], top_crochets[i+1])
                distance_cm = convert_pixels_to_cm(distance_pixels)
                distances['top'].append(distance_cm)

        if len(sorted_crochets['bottom']) > 1:
            bottom_crochets = list(sorted_crochets['bottom'].values())
            for i in range(len(bottom_crochets) - 1):
                distance_pixels = calculate_distance(bottom_crochets[i], bottom_crochets[i+1])
                print (distance_pixels) 
                distance_cm = convert_pixels_to_cm(distance_pixels)
                distances['bottom'].append(distance_cm)


        distance_table = []
        for region in ['top', 'bottom']:
            for i, distance in enumerate(distances[region]):
                row = {
                    "Pair": f"Distance entre {i+1} et {i+2} {region} crochets",
                    "Distance (cm)": f"{distance:.3f} cm"
                }
                distance_table.append(row)
                print(row) 

        return sorted_crochets, distances, distance_table

    return None, None

image_path = "image.jpg"  
im = cv2.imread(image_path)
im = cv2.resize(im, (640, 480))
output_dir = "static"  
output_path = os.path.join(output_dir, 'distances_image.jpg')
sorted_crochets, distances, distance_table= label_and_calculate_crochet_distances(im, instances, metadata)
if sorted_crochets and distances:
    save_image_with_distances(im, sorted_crochets, distances, output_path) 
output_dir = "static"
output_path = os.path.join(output_dir, 'distances_image.jpg')
if os.path.exists(output_path):
    print(f"Image saved successfully at {output_path}")
else:
    print(f"Failed to save image at {output_path}")

font_size = 20
font = ImageFont.load_default()  
padding = 10
row_height = font_size + 2 * padding
col_width = [500, 100]  
image_width = sum(col_width) + 2 * padding
image_height = (len(distance_table) + 1) * row_height + 2 * padding

image = Image.new('RGB', (image_width, image_height), 'white')
draw = ImageDraw.Draw(image)
draw.text((padding, padding), "Pair", fill='black', font=font)
draw.text((col_width[0] + 2 * padding, padding), "Distance (cm)", fill='black', font=font)

for i, row in enumerate(distance_table):
    draw.text((padding, (i + 1) * row_height + padding), row["Pair"], fill='black', font=font)
    draw.text((col_width[0] + 2 * padding, (i + 1) * row_height + padding), row["Distance (cm)"], fill='black', font=font)

output_dir = "static"
table_image_path = os.path.join(output_dir, 'distance_table.png')
image.save(table_image_path)

print(f"Table image saved successfully at {table_image_path}")


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/add_reference', methods=['POST'])
def add_reference():
    new_reference = request.form.get('newReference')
    print(f"New reference added: {new_reference}")
    
    return redirect(url_for('index'))

@app.route('/class_counts')
def class_counts_route():
    global class_counts
    global boite_position_message
    
    reference = request.args.get('reference')
    

    class_counts = defaultdict(int)
    
    for i in range(len(instances)):
        class_id = instances.pred_classes[i].item()
        class_name = metadata.thing_classes[class_id]
        if class_name == 'CROCHETS' and scores[i] < 0.93:
            continue
        if class_name == 'boite' and scores[i] < 0.95:
            continue
        if class_name == 'Piece' and scores[i] < 0.95:
            continue
        if class_name == 'etiquette' and scores[i] < 0.95:
            continue
        filtered_instances_.append(i)
        class_counts[class_name] += 1

    outputs["instances"] = instances[filtered_instances_]
    v = Visualizer(im[:, :, ::-1], metadata=metadata, scale=0.5, instance_mode=ColorMode.IMAGE_BW)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    im_with_prediction = Image.fromarray(out.get_image()[:, :, ::-1])
    im_with_prediction.save('static/predicted_image.jpg')

    messages = {
        "CROCHETS": "not ok. Verify piece.",
        "etiquette": "verify etiquette",
        "boite": "Boite n'exist pas"
    }
    reference = request.args.get('reference')
    if reference:
        if reference == "ENM426203151A" or reference == "ENM426201007A":
            if class_counts['CROCHETS'] == 12:
                messages["CROCHETS"] = "ok"
            if class_counts['etiquette'] != 0:
                messages["etiquette"] = "ok"
            if class_counts['boite'] != 0:
                messages["boite"] = "ok"

        elif reference == "ENM426205007A" or reference == "ENM426203831A" or reference == "ENM426205009A":
            if class_counts['CROCHETS'] == 6:
                messages["CROCHETS"] = "ok"
            if class_counts['etiquette'] != 0:
                messages["etiquette"] = "ok"
            if class_counts['boite'] != 0:
                messages["boite"] = "ok"

        elif reference == "ENM426205002A" or reference == "ENM426201008A":
            if class_counts['CROCHETS'] == 8:
                messages["CROCHETS"] = "ok"
            if class_counts['etiquette'] != 0:
                messages["etiquette"] = "ok"
            if class_counts['boite'] != 0:
                messages["boite"] = "ok"

        elif reference == "ENM426201021A" or reference == "ENM426201003A":
            if class_counts['CROCHETS'] == 10:
                messages["CROCHETS"] = "ok"
            if class_counts['etiquette'] != 0:
                messages["etiquette"] = "ok"
            if class_counts['boite'] != 0:
                messages["boite"] = "ok"

    return jsonify({
        'class_counts': {
            'CROCHETS': f" {class_counts['CROCHETS']} -> {messages['CROCHETS']}",
            'etiquette': f" {class_counts['etiquette']} -> {messages['etiquette']}",
            'boite': f"{class_counts['boite']} -> {messages['boite']}"
        }
    })
    

if __name__ == '__main__':
    app.run(debug=True)
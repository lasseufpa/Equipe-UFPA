from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import (
    create_mobilenetv1_ssd,
    create_mobilenetv1_ssd_predictor,
)
from vision.ssd.mobilenetv1_ssd_lite import (
    create_mobilenetv1_ssd_lite,
    create_mobilenetv1_ssd_lite_predictor,
)
from vision.ssd.squeezenet_ssd_lite import (
    create_squeezenet_ssd_lite,
    create_squeezenet_ssd_lite_predictor,
)
from vision.ssd.mobilenet_v2_ssd_lite import (
    create_mobilenetv2_ssd_lite,
    create_mobilenetv2_ssd_lite_predictor,
)
from vision.utils.misc import Timer
import cv2
import sys
from pathlib import Path
import numpy as np


if len(sys.argv) < 5:
    print(
        "Usage: python run_ssd_example.py <net type>  <model path> <label path> <image path>"
    )
    sys.exit(0)
net_type = sys.argv[1]
model_path = sys.argv[2]
label_path = sys.argv[3]
images_path = Path(sys.argv[4])

class_names = [name.strip() for name in open(label_path).readlines()]

if net_type == "vgg16-ssd":
    net = create_vgg_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd":
    net = create_mobilenetv1_ssd(len(class_names), is_test=True)
elif net_type == "mb1-ssd-lite":
    net = create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
elif net_type == "mb2-ssd-lite":
    net = create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
elif net_type == "sq-ssd-lite":
    net = create_squeezenet_ssd_lite(len(class_names), is_test=True)
else:
    print(
        "The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite."
    )
    sys.exit(1)
net.load(model_path)

if net_type == "vgg16-ssd":
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd":
    predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
elif net_type == "mb1-ssd-lite":
    predictor = create_mobilenetv1_ssd_lite_predictor(net, candidate_size=200)
elif net_type == "mb2-ssd-lite":
    predictor = create_mobilenetv2_ssd_lite_predictor(net, candidate_size=200)
elif net_type == "sq-ssd-lite":
    predictor = create_squeezenet_ssd_lite_predictor(net, candidate_size=200)
else:
    predictor = create_vgg_ssd_predictor(net, candidate_size=200)

new = images_path / "Annotations"
new.mkdir(parents=True, exist_ok=True)


def convert_bbs(bbs, img_shp):
    bbs = np.array(bbs)
    output = np.zeros_like(bbs)

    r, c = bbs.shape
    for i in range(r):
        xmin, ymin, xmax, ymax = bbs[i, :]
        xc = ((xmin + xmax) / 2.0) / img_shp[1]
        yc = ((ymin + ymax) / 2.0) / img_shp[0]
        w = (xmax - xmin) / float(img_shp[1])
        h = (ymax - ymin) / float(img_shp[0])
        output[i, :] = [xc, yc, w, h]

    return output


for image_path in images_path.glob("*.jpg"):
    orig_image = cv2.imread(str(image_path))
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    boxes, labels, probs = predictor.predict(image, 30, 0.3)
    path = str(new / image_path.name).split(".")[0] + ".txt"
    annotation = open(path, "a")
    bbs = convert_bbs(boxes, image.shape)
    for i in range(boxes.size(0)):
        annotation.write(
            "{} {} {} {} {}\n".format(
                labels[i] - 1, bbs[i, 0], bbs[i, 1], bbs[i, 2], bbs[i, 3]
            )
        )
    annotation.close()

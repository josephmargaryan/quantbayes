import cv2, argparse
from vision import SegmentationPipeline

ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",required=True)
ap.add_argument("--img"); ap.add_argument("--out")
args=ap.parse_args()
pipe=SegmentationPipeline.load(args.ckpt)
img=cv2.imread(args.img)
mask=pipe.predict(img)
cv2.imwrite(args.out,mask)

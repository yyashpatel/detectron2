
import cv2
import sys
import glob
import time
import torch
from torch import nn
from tqdm import tqdm
from read_inputs import parseArgs, checkFolderPaths
from visualize import myVisualizer
import detectron2.data.transforms as T
from detectron2.data.catalog import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine.defaults import create_ddp_model
from detectron2.engine import default_setup, default_argument_parser, DefaultPredictor
from detectron2.evaluation import inference_on_dataset, print_csv_format


class Inference(nn.Module):
    def __init__(self,cfg, args, metadata, result_path):
        super().__init__()
        self.args = args
        self.meta_data = metadata
        self.result_path = result_path
        self.model = instantiate(cfg.model)
        self.model.to(cfg.train.device)
        self.model.eval()
        self.model = create_ddp_model(self.model)
        DetectionCheckpointer(self.model).load(cfg.train.init_checkpoint)
        self.aug = T.ResizeShortestEdge(short_edge_length=800, max_size=1333)
        # self.aug = T.ResizeShortestEdge(short_edge_length=2000, max_size=3333)
        self.input_format = cfg.model.input_format
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def forward(self, data_path):
        if self.args.input_data == "images":
            self.inferenceImage(data_path)

    def run_inference(self, image):        
        with torch.no_grad():
            if self.input_format == "RGB":
                image = image[:, :, ::-1]
            visualize_on_image = myVisualizer(image, self.meta_data, scale=1.2)
            height, width = image.shape[:2]
            image = self.aug.get_transform(image).apply_image(image)
            print('image after transform: ', image.shape)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            tic = time.time()
            predictions = self.model([inputs])[0]
            c = time.time() - tic
            print('cost: {}, fps: {}'.format(c, 1/c))
            detection_output = visualize_on_image(predictions["instances"].to("cpu"), self.args.score_threshold)
            return detection_output

    """
    Read Images
    """
    def inferenceImage(self, data_path):
        image_path = sorted((glob.glob(data_path + "/*.jpg")))
        assert len(image_path), "No images found at %s." % (data_path + "/*.jpg")
        for i, path in tqdm(enumerate(image_path), total=len(image_path), desc="Image"):
            image = read_image(path, format="BGR")
            predictions = self.run_inference(image)
            cv2.imwrite(self.result_path+"/pred_image_%06i.jpg"%i, predictions.get_image()[:, :, ::-1])


def main(cfg, args):
    
    folder_path = "/".join(sys.path[0].split("/")[:-1]) + "/"
    data_path = folder_path  + "data/" + args.input_data

    result_path = folder_path  + args.output_path
    checkFolderPaths([result_path])

    default_setup(cfg, args)
    metadata = MetadataCatalog.get(cfg.dataloader.test.dataset.names)
    
    run_inference = Inference(cfg, args, metadata, result_path)
        
    run_inference(data_path)


if __name__ == "__main__":
    args = parseArgs()
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    main(cfg, args)

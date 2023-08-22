import os
import logging
import pickle
import random
from typing import Any, List
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from datasets.base_dataset import BaseDataset, TestBaseTransform, TrainBaseTransform
from datasets.image_reader import build_image_reader
from datasets.transforms import RandomColorJitter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

random_seed = 666

print("Random Seed: ", random_seed)
logger = logging.getLogger("global_logger")
classes = [
    "abudefduf_vaigiensis", "acanthurus_coeruleus", "acarospora_socialis", "acris_gryllus", "adolphia_californica",
    "agaricus_augustus", "amanita_parcivolvata", "ameiurus_natalis", "anartia_jatrophae", "aplysia_californica",
    "arcyria_denudata", "arion_ater", "aulostomus_chinensis", "bjerkandera_adusta", "bombus_pensylvanicus",
    "callianax_biplicata", "calochortus_apiculatus", "campsis_radicans", "cepaea_nemoralis", "cladonia_chlorophaea",
    "cochlodina_laminata", "crepidotus_applanatus", "cypripedium_macranthos", "dendrobates_auratus",
    "diaulula_sandiegensis", "drosera_rotundifolia", "echinocereus_pectinatus_pectinatus", "eremnophila_aureonotata",
    "etheostoma_caeruleum", "fistularia_commersonii", "ganoderma_tsugae", "halysidota_tessellaris",
    "herichthys_cyanoguttatus", "hippocampus_whitei", "hygrocybe_miniata", "hypomyces_chrysospermus",
    "juniperus_turbinata", "kuehneromyces_mutabilis", "laetiporus_gilbertsonii", "lepisosteus_platyrhincus",
    "leucocoprinus_cepistipes", "lissachatina_fulica", "lycium_californicum", "megapallifera_mutabilis",
    "mononeuria_groenlandica", "neverita_lewisii", "octopus_tetricus", "orienthella_trilineata",
    "phyllotopsis_nidulans", "planorbarius_corneus", "protoparmeliopsis_muralis", "quercus_dumosa",
    "ruditapes_philippinarum", "salamandra_lanzai", "swietenia_macrophylla", "teloschistes_chrysophthalmus",
    "tramea_onusta", "umbra_limi", "vespula_squamosa", "zelus_renardii"
]
def build_species_dataloader(cfg, training, distributed=True):
    image_reader = build_image_reader(cfg.image_reader)
    normalize_fn = transforms.Normalize(mean=cfg["pixel_mean"], std=cfg["pixel_std"])
    if training:
        transform_fn = TrainBaseTransform(
            cfg["input_size"], cfg["hflip"], cfg["vflip"], cfg["rotate"]
        )
    else:
        transform_fn = TestBaseTransform(cfg["input_size"])
    colorjitter_fn = None
    if cfg.get("colorjitter", None) and training:
        colorjitter_fn = RandomColorJitter.from_params(cfg["colorjitter"])
    #logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))
    dataset = SpeciesDataset(
        image_reader,
        training,
        transform_fn=transform_fn,
        normalize_fn=normalize_fn,
        colorjitter_fn=colorjitter_fn,
        classes=classes,
        root_dir = cfg["image_reader"]['kwargs']["image_dir"],
    )
    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = RandomSampler(dataset)
    data_loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        sampler=sampler,
    )
    #logger.info("building CustomDataset from: {}".format(cfg["meta_file"]))


    return data_loader

class SpeciesDataset(BaseDataset):
    def __init__(
        self,
        image_reader,
        training,
        transform_fn,
        normalize_fn,
        colorjitter_fn=None,
        classes=None,
        root_dir = None
    ):
        self.image_reader = image_reader
        self.training = training
        self.transform_fn = transform_fn
        self.normalize_fn = normalize_fn
        self.colorjitter_fn = colorjitter_fn
        self.classes = classes
        self.num_classes = len(classes)
        self.root = root_dir

        ## class 명 기준 절반으로 나누기
        # num_normal = self.num_classes // 2
        # self.normals = self.classes[:num_normal]
        # self.abnormals = self.classes[num_normal:]

        ## random seed로 30개 class를 랜덤으로 지정하기
        if random_seed is not None:
            random.seed(random_seed)
            random.shuffle(self.classes)  
        
        num_normals = 30  
        self.normals = random.sample(self.classes, num_normals)
        self.abnormals = [cls for cls in self.classes if cls not in self.normals]
        print('Normal casses: ', self.normals)

        self.data = []
        self.targets = []
        data_dir = "one_class_train" if self.training else "one_class_test"
        print("Data directory:", data_dir)  # Print the data directory being use
        for class_name in os.listdir(os.path.join(self.root, data_dir)):
            class_dir = os.path.join(self.root, data_dir, class_name)
            if class_name in self.normals:
                label = 0  # Normal class
            else:
                label = 1  # Anomaly class
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                self.data.append(image_path)
                self.targets.append(label)
        self._load_meta()

    def _load_meta(self) -> None:
        self.classes = classes  # Use the globally defined classes
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index: int):
        image_path, target = self.data[index], self.targets[index]
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)  # Convert PIL Image to numpy array
        height, width = img.shape[0], img.shape[1]
        
        mask = torch.zeros((1, height, width)) if target == 0 else torch.ones((1, height, width))
        
        input_data = {
            "filename": os.path.basename(image_path),
            "image": img,
            "mask": mask,
            "height": height,
            "width": width,
            "label": target,
            "clsname": "species",
        }
        
        if self.transform_fn:
            img_pil = Image.fromarray(img)
            mask_pil = Image.fromarray(mask.squeeze().numpy(), mode='L')  # Convert mask to grayscale PIL Image
            img_pil, mask_pil = self.transform_fn(img_pil, mask_pil)
            img = np.array(img_pil)
            mask = np.array(mask_pil)
            
        if self.colorjitter_fn:
            img = self.colorjitter_fn(img)
            
        img = transforms.ToTensor()(img)
        mask = torch.tensor(mask, dtype=torch.uint8)
        mask = mask.unsqueeze(0)
        
        if self.normalize_fn:
            img = self.normalize_fn(img)
            
        input_data.update({"image": img, "mask": mask})
        return input_data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def extra_repr(self) -> str:
        split = "Train" if self.train else "Test"
        return f"Split: {split}"


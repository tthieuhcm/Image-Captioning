from PIL import Image
import json
from torch.utils.data import Dataset


class MSCOCO_Dataset(Dataset):
    """MSCOCO dataset"""

    def __init__(self, annotation_file, root_dir, transform=None):
        """
        Args:
            annotation_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        annotations = json.load(open(annotation_file, 'r'))
        self.transform = transform
        # storing the captions and the image name in vectors
        self.all_captions = []
        self.all_img_name_vector = []

        for annot in annotations['annotations']:
            caption = '<start> ' + annot['caption'] + ' <end>'
            image_id = annot['image_id']
            full_coco_image_path = root_dir + 'COCO_train2014_' + '%012d.jpg' % image_id

            self.all_img_name_vector.append(full_coco_image_path)
            self.all_captions.append(caption)

    def __len__(self):
        return len(self.all_img_name_vector)

    def __getitem__(self, idx):
        img_name = self.all_img_name_vector[idx]
        image = Image.open(img_name)
        caption = self.all_captions[idx]

        if self.transform:
            image = self.transform(image)

        return {'image': image, 'caption': caption, 'path': img_name}
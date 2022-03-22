import numpy as np
import cv2
from detectron2.data import transforms
from detectron2.data import DatasetMapper


class CustomMapper(DatasetMapper):
    def __call__(cls, dataset_dict):
        dataset_dict = super().__call__(dataset_dict)
        dataset_dict["height"] = dataset_dict["instances"]._image_size[0]
        dataset_dict["width"] = dataset_dict["instances"]._image_size[1]
        return dataset_dict


LR_NOISE_DIV = 100
LR_NOISE_SCALE = 1
MR_NOISE_DIV = 50
MR_NOISE_SCALE = 0.5
HR_NOISE_SCALE = 0.2
NOISE_MUL = 0.05
class RandomColourNoise(transforms.Augmentation):
    def get_transform(self, image):
        noise_img = np.zeros_like(image)
        if np.random.choice([True, False]):
            img_shape = image.shape
            lr_shape = (img_shape[0]//LR_NOISE_DIV, img_shape[1]//LR_NOISE_DIV, 3)
            mr_shape = (img_shape[0]//MR_NOISE_DIV, img_shape[1]//MR_NOISE_DIV, 3)
            lowres_noise = np.random.normal(scale=LR_NOISE_SCALE, size=lr_shape)
            medres_noise = np.random.normal(scale=MR_NOISE_SCALE, size=mr_shape)
            highres_noise = np.random.normal(scale=HR_NOISE_SCALE, size=img_shape)
            highres_noise += cv2.resize(lowres_noise, img_shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            highres_noise += cv2.resize(medres_noise, img_shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
            noise_img = highres_noise + np.min(highres_noise)
            noise_img = NOISE_MUL * noise_img
            noise_img = (255*noise_img).astype(np.uint8)
        return transforms.ColorTransform(lambda x: x + noise_img)


class RandomErasing(transforms.Augmentation):
    def get_transform(self, image):
        noise_img = np.ones_like(image)
        if np.random.choice([True, False]):
            img_shape = image.shape
            lr_shape = (img_shape[0]//50, img_shape[1]//50, 1)
            mr_shape = (img_shape[0]//20, img_shape[1]//20, 1)
            hr_shape = (img_shape[0]//1, img_shape[1]//1, 1)
            lowres_noise = np.random.normal(scale=LR_NOISE_SCALE, size=lr_shape)
            medres_noise = np.random.normal(scale=MR_NOISE_SCALE, size=mr_shape)
            highres_noise = np.random.normal(scale=HR_NOISE_SCALE, size=hr_shape)
            highres_noise += cv2.resize(lowres_noise, img_shape[:2][::-1], interpolation=cv2.INTER_CUBIC)[:, :, None]
            highres_noise += cv2.resize(medres_noise, img_shape[:2][::-1], interpolation=cv2.INTER_CUBIC)[:, :, None]
            noise_img = highres_noise < 1
            noise_img = noise_img.astype(np.uint8)
        return transforms.ColorTransform(lambda x: x * noise_img)


class GeometricTransform(transforms.Transform):
    # def get_transform(self, img):
    #     img_shape = img.shape
    #     img_points = np.vstack([np.indices(img_shape[:2])[::-1], np.zeros(img_shape[:2])[None], np.ones(img_shape[:2])[None]]).reshape((4, -1))
    #
    #     Q = np.eye(4)
    #     Q[:3, 3] = (np.random.random(size=(3,)) - 0.5) * 100
    #     Q[:3, :3] += (np.random.random(size=(3, 3)) - 0.5) / 2
    #     img_pnts_T = np.matmul(Q, img_points).astype(np.int32)
    #     pixel_coords = img_pnts_T[:2][::-1]
    #     rows = pixel_coords[0].clip(0, img_shape[0]-1)
    #     cols = pixel_coords[1].clip(0, img_shape[1]-1)
    #     img_T = img[(rows, cols)].reshape(img_shape)
    #
    #     return transforms.Transform()
    def __init__(self):
        super().__init__()
        self.Q = np.eye(4)

    def __call__(self):
        print("asdfg")

    def apply_image(self, img):
        print("img")
        # Randomise transform
        self.Q[:3, 3] = np.random.normal(scale=200, size=(3,))
        # self.Q[:3, :3] += (np.random.random(size=(3, 3)) - 0.5) / 2
        # self.Q[:3, 2] = np.cross(self.Q[:3, 0], self.Q[:3, 1])
        # self.Q[:3, 1] = np.cross(self.Q[:3, 2], self.Q[:3, 0])
        # self.Q[:3, :3] /= np.linalg.norm(self.Q[:3, :3], axis=0)

        img_shape = img.shape
        img_points = np.vstack([np.indices(img_shape[:2])[::-1], np.zeros(img_shape[:2])[None], np.ones(img_shape[:2])[None]]).reshape((4, -1))
        img_pnts_T = np.matmul(np.linalg.inv(self.Q), img_points).astype(np.int32)
        pixel_coords = img_pnts_T[:2][::-1]
        rows = pixel_coords[0].clip(0, img_shape[0]-1)
        cols = pixel_coords[1].clip(0, img_shape[1]-1)
        return img[(rows, cols)].reshape(img_shape)

    def apply_box(self, bbox):
        print('bbox')
        return bbox

    def apply_polygons(self, polygons):
        print('polys')
        return polygons

    def apply_segmentation(self, segmentation):
        print('seg')
        return segmentation

    def apply_coords(self, coords):
        print(coords.shape)
        print(np.mean(coords, axis=0))
        points_2N = coords.transpose()
        points_4N = np.concatenate([points_2N, np.zeros_like(points_2N)], axis=0)
        points_4N[3, :] = 1
        points_T = np.matmul(self.Q, points_4N).astype(np.int32)
        return points_T[:2, :].transpose()

    # def apply_box(self, bbox):
    #     print(bbox)
    #     print("hello")
    #     return bbox
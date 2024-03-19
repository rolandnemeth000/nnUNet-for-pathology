import numpy as np
import cv2
from shapely.geometry import Point
from wholeslidedata.samplers.callbacks import BatchCallback

class PointsToSegBatchCallback(BatchCallback):
    def __init__(self, point_sizes_dict, spacing):
        self.point_sizes_dict = {int(k): int(v) for k, v in point_sizes_dict.items()}
        self.spacing = spacing
    
    @staticmethod
    def dist_to_px(dist, spacing):
        """ distance in um (or rather same unit as the spacing) """
        dist_px = int(round(dist / spacing))
        return dist_px

    @staticmethod
    def find_coords_for_values(arr, values):
        coords_dict = {}
        for value in values:
            coords = np.argwhere(arr == value)
            coords_dict[value] = coords
        return coords_dict

    @staticmethod
    def array_to_points_with_buffer(array, point_sizes_dict, spacing, add_centroids=True, centroid_size=1.5, centroid_value=None):
        result_dict = {}
        coords_dict = PointsToSegBatchCallback.find_coords_for_values(array, list(point_sizes_dict.keys()))
        for value, coords in coords_dict.items():
            buffer_distance = point_sizes_dict[value]
            buffer_distance_px = PointsToSegBatchCallback.dist_to_px(buffer_distance, spacing)
            for coord in coords:
                point = Point(coord[1], coord[0])            
                buffered_point = point.buffer(buffer_distance_px).simplify(1)
                result_dict.setdefault(value, []).append(buffered_point)
        
        # Add centroids if required
        if add_centroids:
            centroid_value = centroid_value if centroid_value is not None else max([int(k) for k in point_sizes_dict.keys()]) + 1
            buffer_distance_px = PointsToSegBatchCallback.dist_to_px(centroid_size, spacing)
            all_points = [Point(coord[1], coord[0]) for coords in coords_dict.values() for coord in coords]  # Original points
            buffered_all_points = [point.buffer(buffer_distance_px).simplify(1) for point in all_points]
            result_dict[centroid_value] = buffered_all_points

        return result_dict

    @staticmethod
    def seg_mask_from_points_dict(points_dict, input_array):
        result_array = np.copy(input_array)
        for class_value in points_dict:
            for point in points_dict[class_value]:
                coords = np.array(point.exterior.coords)
                coords = np.round(coords).astype(np.int32)
                coords = coords.reshape((-1, 1, 2))
                cv2.fillPoly(result_array, [coords], color=int(class_value))
        return result_array

    @staticmethod
    def point_mask_to_seg_mask(mask_patch, point_sizes_dict):
        points_dict = PointsToSegBatchCallback.array_to_points_with_buffer(mask_patch, point_sizes_dict, 0.5)
        seg_mask = PointsToSegBatchCallback.seg_mask_from_points_dict(points_dict, mask_patch)
        return seg_mask

    def __call__(self, x_batch, y_batch):
        transformed_y_batch = []
        for mask_patch in y_batch:
            seg_mask = self.point_mask_to_seg_mask(mask_patch, self.point_sizes_dict)
            transformed_y_batch.append(seg_mask)
        return x_batch, np.array(transformed_y_batch)
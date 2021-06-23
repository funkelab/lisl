import numpy as np
import h5py
import napari
from lisl.pl.evaluation import AnchorSegmentationValidation
# from affogato.segmentation import compute_mws_segmentation
# from embeddingutils.affinities import embedding_to_affinities
from skimage.measure import label as label_cont
import zarr
import torch

def _print_help():
    print("Interactive Mutex Watershed Application")
    print("Keybindigns:")
    print("[h] show help")


class InteractiveSegmentation:

    def __init__(self,
                 zarray_paths):

        self.data = {}
        for name, (filename, key) in zarray_paths.items():
            print(name, filename, key)
            z_array = zarr.open(filename, mode="r")
            self.data[name] = np.array(z_array[key])

        self.run()

    def run(self):
        # get the initial mws segmentation
        _print_help()

        # add initial layers to the viewer
        with napari.gui_qt():
            viewer = napari.Viewer()

            # add image layers and point layer for seeds
            for name, img in self.data.items():
                viewer.add_image(img, name=name)

            points_layer = viewer.add_points(
                # properties={'label': labels},
                edge_color='#ff7f0e',
                # edge_color_cycle='#ff7f0e',
                symbol='o',
                face_color='transparent',
                edge_width=2,
                size=1,
                # ndim=2
            )

            # def next_on_click(layer, event):
            #     """Mouse click binding to advance the label when a point is added"""
            #     if layer.mode == 'add':
            #         next_label()

            #         # by default, napari selects the point that was just added
            #         # disable that behavior, as the highlight gets in the way
            #         layer.selected_data = {}

            # # save the current segmentation
            @viewer.bind_key('s')
            def segment(viewer):
                support_position = points_layer._data
                support_index = np.unique(points_layer.edge_color, axis=0, return_inverse=True)[1]
                # Todo: filter points outside of data

                # load embeddings
                test_spatial_embeddings = 
                test_semantic_embeddings = 



                spatial_matching_log_probas(train_spatial_embeddings,
                                            train_semantic_embeddings,
                                            test_spatial_embeddings,
                                            test_semantic_embeddings,
                                            train_targets,
                                            test_targets,
                                            args.num_ways,
                                            eps=1e-8)
                

            #     nonlocal seg_path
            #     seg_path = _read_file_path(seg_path)
            #     seg = viewer.layers['segmentation'].data
            #     _save(seg_path, seg)

            # @viewer.bind_key('v')
            # def save_seeds(viewer):
            #     nonlocal seed_path
            #     seed_path = _read_file_path(seed_path)
            #     seeds = viewer.layers['seeds'].data
            #     _save(seed_path, seeds)

            # display help
            @viewer.bind_key('h')
            def print_help(viewer):
                _print_help()




if __name__ == '__main__':

    base_path = "/home/swolf/mnt/janelia/nrs/funke/wolfs2/lisl/experiments/semantic/c32/prediction/"
    frame = 24
    zarray_paths = {
        "raw": ('/home/swolf/mnt/janelia/nrs/funke/wolfs2/lisl/datasets/dsb_indexed.zarr', f'train/raw/{frame}'),
        "spatial_absolute": (base_path + 'matching_network.zarr', f'train/{frame}/matching_network_prediciton_spatial_absolute'),
        "spatial_relative": (base_path + 'matching_network.zarr', f'train/{frame}/matching_network_prediciton_spatial_relative'),
        "semantic": (base_path + 'matching_network.zarr', f'train/{frame}/matching_network_prediciton_semantic')}
    imws = InteractiveSegmentation(zarray_paths)

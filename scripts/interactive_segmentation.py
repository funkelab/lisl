import numpy as np
import h5py
import napari
from lisl.pl.evaluation import AnchorSegmentationValidation
from affogato.segmentation import compute_mws_segmentation
from embeddingutils.affinities import embedding_to_affinities
from skimage.measure import label as label_cont
import zarr
import torch

def _print_help():
    print("Interactive Mutex Watershed Application")
    print("Keybindigns:")
    print("[h] show help")


class InteractiveSegmentation:

    def __init__(self,
                 raw,
                 embedding_relative,
                 embedding,
                 offsets,
                 strides):

        self.raw = raw
        self.embedding = embedding
        self.embedding_relative = embedding_relative

        self.offsets = offsets
        self.strides = strides
        self.att_c = 2

        self.anchor_segmentation = AnchorSegmentationValidation()

        def affinity_measure(x, y, dim=0):
            distance = (x - y).norm(2, dim=dim)
            return (-distance.pow(2) / temperature).exp()

        self.affinity_measure = affinity_measure

        self.run()

    def run(self):
        # get the initial mws segmentation
        _print_help()

        # add initial layers to the viewer
        with napari.gui_qt():
            viewer = napari.Viewer()

            # add image layers and point layer for seeds
            viewer.add_image(self.raw, name='raw')
            viewer.add_image(self.embedding, name='embedding')
            viewer.add_image(self.embedding_relative, name='embedding_relative')
            viewer.add_labels(np.zeros_like(seg), name='seeds')

            emb = torch.from_numpy(self.embedding)
            affinities = self.affinities_from_embedding(emb)

            self.imws = InteractiveMWS(affinities.cpu().numpy(), self.offsets, n_attractive_channels=self.att_c, strides=self.strides)
            segmentation = self.imws()
            # self.segment(affinities)
            viewer.add_labels(segmentation, name='segmentation')


            # add key-bindings
            # update segmentation by re-running mws
            @viewer.bind_key('u')
            def update_mws(viewer):
                self.update_mws_impl(viewer)

            # save the current segmentation
            @viewer.bind_key('s')
            def save_segmentation(viewer):
                nonlocal seg_path
                seg_path = _read_file_path(seg_path)
                seg = viewer.layers['segmentation'].data
                _save(seg_path, seg)

            @viewer.bind_key('v')
            def save_seeds(viewer):
                nonlocal seed_path
                seed_path = _read_file_path(seed_path)
                seeds = viewer.layers['seeds'].data
                _save(seed_path, seeds)

            @viewer.bind_key('t')
            def training_step(viewer):
                self.training_step_impl(viewer)

            # display help
            @viewer.bind_key('h')
            def print_help(viewer):
                _print_help()


    def affinities_from_embedding(self, emb):
        affinities = embedding_to_affinities(emb,
                                             offsets=self.offsets,
                                             affinity_measure=self.affinity_measure)

        # compute affinityies
        affinities[self.att_c:] *= -1
        affinities[self.att_c:] += 1

        return affinities


    def segment(self, affinities):
        self.att_c = 2
        seg = compute_mws_segmentation(affinities,
                                        self.offsets,
                                        att_c,
                                        strides=self.strides,
                                        mask=None).astype(np.int32)
        return label_cont(seg)


    def update_mws_impl(self, viewer): 
        print("Update mws triggered")
        layers = viewer.layers
        seeds = layers['seeds'].data

        seg_layer = layers['segmentation']
        print("Clearing seeds ...")
        self.imws.clear_seeds()
        # FIXME this takes much to long, something is wrong here
        print("Updating seeds ...")
        self.imws.update_seeds(seeds)
        print("Recomputing segmentation from seeds ...")
        seg = self.imws()
        print("... done")
        seg_layer.data = seg
        seg_layer.refresh()



def interactive_napari(zarray_path):

    z_array = zarr.open(zarray_path, mode="r")

    raw = np.array(z_array["0/raw"])
    embedding_relative = np.array(z_array["0/embedding"])
    embedding = np.array(z_array["0/embedding_abs"])
    gt_segmentation = np.array(z_array["0/gt_segmentation"])

    strides = [4, 4]
    offsets = np.array([[-1, 0], [0, -1],
                        [-9, 0], [0, -9],
                        [-9, -9], [9, -9],
                        [-9, -4], [-4, -9], [4, -9], [9, -4],
                        [-27, 0], [0, -27]], int)

    print(raw.shape, embedding.shape, gt_segmentation.shape)
    imws = InteractiveSegmentation(raw,
                                   embedding_relative,
                                   embedding,
                                   offsets,
                                   strides)


if __name__ == '__main__':

    zarray_path = '/home/swolf/mnt/janelia/nrs/funke/wolfs2/lisl/experiments/dense_run_01/01_train/setup_t0031/evaluation/rotflip/00080799/embedding_6_0.zarr'
    interactive_napari(zarray_path)

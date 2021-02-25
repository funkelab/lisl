from funlib.show.neuroglancer import add_layer
from sklearn.neighbors import KNeighborsClassifier
import argparse
import daisy
import neuroglancer
import numpy as np
import time
import webbrowser
import mlpack as mlp
import unionfind
from skimage.measure import label
import networkx as nx

parser = argparse.ArgumentParser()
parser.add_argument(
    '--raw-file',
    '-rf',
    type=str,
    help="The path to the raw container to show")
parser.add_argument(
    '--raw-dataset',
    '-rd',
    type=str,
    help="The name of the raw dataset")
#parser.add_argument(
#    '--emb-file',
#    '-ef',
#    type=str,
#    help="The path to the embedding container to use")
parser.add_argument(
    '--emb-dataset',
    '-ed',
    type=str,
    help="The name of the embedding dataset")


class Classifier:

    def __init__(self, embedding):

        self.num_features = embedding.shape[0]
        self.embedding_shape = embedding.shape[1:]

        print("Reading embedding into memory...")
        self.embedding = embedding.data[:].transpose((1, 2, 0))
        self.embedding = self.embedding.reshape(-1, self.num_features)
        print("...done")
        print(f"Flattened embedding shape: {self.embedding.shape}")

        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

        self.samples = None
        self.labels = None

    def pos_to_index(self, pos):

        z, y, x = pos
        return (
            int(z)*self.embedding_shape[1]*self.embedding_shape[0] +
            int(y)*self.embedding_shape[1] +
            int(x))

    def add_sample(self, pos, label):

        index = self.pos_to_index(pos)
        print(f"Reading embedding at {pos}, index {index}")
        sample = self.embedding[index]

        print(f"New sample {sample} with label {label}")

        if self.samples is None:
            self.samples = np.array([sample])
            self.labels = np.array([label])
        else:
            self.samples = np.vstack([self.samples, [sample]])
            self.labels = np.concatenate([self.labels, [label]])

        print(f"Current samples: {self.samples}")
        print(f"Current labels: {self.labels}")

    def fit(self):

        print("Updating classifier...")
        start = time.time()
        self.knn.fit(self.samples, self.labels)
        print("...%.3fs" % (time.time() - start))

    def predict(self):

        print("Predicting on whole embedding...")
        start = time.time()
        prediction = self.knn.predict(self.embedding)
        print("...%.3fs" % (time.time() - start))
        prediction = prediction.reshape(self.embedding_shape)
        print("...done")

        return prediction


class InteractiveSegmentation:

    def __init__(self, raw, embedding, mst, classifier):

        self.raw = raw
        self.embedding = embedding
        self.classifier = classifier
        self.mst = mst

        self.points = []

        self.mst_graph = nx.Graph()
        self.mst_graph.add_weighted_edges_from(mst)

        self.threshold = 0.5

        self.raw_dimensions = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units='nm',
            scales=raw.voxel_size)

        self.dimensions = neuroglancer.CoordinateSpace(
            names=['c^', 'z', 'y', 'x'],
            units=[''] + 3*['nm'],
            scales=raw.voxel_size)

        # if len(raw.shape) > 3:
        #     volume_shape = raw.shape[1:]
        # else:
        volume_shape = raw.shape

        print(f"Creating segmentation layer with shape {volume_shape}")
        self.segmentation = np.arange(np.product(volume_shape),dtype=np.uint32)
        self.segmentation = self.segmentation.reshape(volume_shape)
        
        self.segmentation_volume = neuroglancer.LocalVolume(
            data=self.segmentation,
            dimensions=self.raw_dimensions)

        self.viewer = neuroglancer.Viewer()
        self.viewer.actions.add('label_fg', self._label_fg)
        self.viewer.actions.add('label_bg', self._label_bg)
        self.viewer.actions.add('update_seg', self._update_segmentation)

        with self.viewer.config_state.txn() as s:
            s.input_event_bindings.data_view['shift+mousedown0'] = 'label_fg'
            s.input_event_bindings.data_view['shift+mousedown1'] = 'label_bg'
            s.input_event_bindings.data_view['keyu'] = 'update_seg'

        with self.viewer.txn() as s:
            
            add_layer(s, self.raw, 'raw')
            add_layer(s, self.embedding, 'embedding')
            s.layers['embedding'].visible = False
            s.layers['points'] = neuroglancer.LocalAnnotationLayer(
                self.dimensions)
            s.layers['segmentation'] = neuroglancer.SegmentationLayer(
                source=self.segmentation_volume)

    def label(self, pos, label):

        print(f"Labelling {pos} as {label}")
        with self.viewer.txn() as s:
            s.layers['points'].annotations.append(
                neuroglancer.PointAnnotation(id=repr(pos), point=pos))

        self.classifier.add_sample(pos, label)
        self.classifier.fit()

    def _update_segmentation(self, action_state):
        
        if len(self.points) < 2:
            return


        n_pixels = np.product(self.embedding.data.shape[1:])
        uf = unionfind.UnionFind(n_pixels)

        label_image = np.zeros((1, ) + self.embedding.data.shape[1:])

        # find minimal threshold
        threshold = None
        for p0 in self.points:
            for p1 in self.points:
                if p0 < p1:
                    min_max_edge = None
                    short_path = nx.shortest_path(self.mst_graph, p0, p1)
                    for u, v in zip(short_path, short_path[1:]):
                        weight = self.mst_graph.get_edge_data(u, v)['weight']
                        if min_max_edge is None or weight > min_max_edge:
                            min_max_edge = weight

                    if threshold is None or threshold > min_max_edge:
                        threshold = min_max_edge

        uf.union_array(mst[mst[:, 2] < threshold][:, :2].astype(np.uint32))
        label_image = uf.get_label_image(self.embedding.data.shape[1:])[None]
        label_image = label_image.astype(np.uint32)
        self.segmentation[:] = label_image
        self.update_view()

    def update_view(self):
        self.segmentation_volume.invalidate()

    def _label_fg(self, action_state):
        pos = action_state.mouse_voxel_coordinates
        if pos is None:
            return
        print(self.segmentation.shape)
        self.points.append(int((self.segmentation.shape[1] * int(pos[1])) + int(pos[2])))
        print("pos", pos)
        print(self.points)
        print("------------------------------------")
        self._update_segmentation(action_state)

    def _label_bg(self, action_state):
        pos = action_state.mouse_voxel_coordinates
        if pos is None:
            return
        self.label(pos, 0)


if __name__ == "__main__":

    neuroglancer.set_server_bind_address('0.0.0.0')

    args = parser.parse_args()

    raw = daisy.open_ds(args.raw_file, args.raw_dataset)
    print("raw is open")
    embedding = daisy.open_ds(args.raw_file, args.emb_dataset)
    # mst = mlp.emst(embedding)["output"]
    channels = embedding.shape[0]
    
    embedding_transp = np.array(embedding.data).transpose(1,2,0)

    cx = np.arange(embedding_transp.shape[1], dtype=embedding_transp.dtype)
    cy = np.arange(embedding_transp.shape[0], dtype=embedding_transp.dtype)
    coords = np.meshgrid(cx, cy, copy=True)
    coords = np.stack(coords, axis=-1)

    print(embedding_transp.shape)
    embedding_transp[..., :2] += coords
    embedding_transp = embedding_transp.reshape((-1, channels))

    mst = mlp.emst(embedding_transp)["output"]

    # label_image = label(label_image).astype(np.uint32)
    # for i in range(embedding_transp.shape[0]//10):
    #     u,v,s = mst[i]
    #     uf.union(int(u), int(v))

    # from tqdm import tqdm
    # for i in tqdm(range(embedding_transp.shape[0])):
    #     label_image.flatten()[i] = uf.find(i)

    print("embedding is open")
    
    print(raw.data.shape, embedding.data.shape)

    for a in [raw, embedding]:
        if a.roi.dims() == 2:
            print("ROI is 2D, recruiting next channel to z dimension")
            a.roi = daisy.Roi(
                (0,) + a.roi.get_begin(),
                (a.shape[-3],) + a.roi.get_shape())
            a.voxel_size = daisy.Coordinate((1,) + a.voxel_size)
            a.n_channel_dims -= 1
            print(a.roi)
            print(a.shape)
            print(a.data.shape)

    # raw.shape == (92, 700, 1100)
    # embedding.shape == (64, 92, 700, 1100)

    print(embedding.roi)

    print(raw.data.shape, embedding.data.shape)

    classifier = Classifier(embedding)
    interactive_segmentation = InteractiveSegmentation(
        raw,
        embedding,
        mst,
        classifier)

    url = str(interactive_segmentation.viewer)
    print(url)
    webbrowser.open_new(url)
    print("Press ENTER to quit")
    input()

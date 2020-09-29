from funlib.show.neuroglancer import add_layer
from sklearn.neighbors import KNeighborsClassifier
import argparse
import daisy
import neuroglancer
import numpy as np
import time
import webbrowser


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
parser.add_argument(
    '--emb-file',
    '-ef',
    type=str,
    help="The path to the embedding container to use")
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
        self.embedding = embedding.data[:].transpose((1, 2, 3, 0))
        self.embedding = self.embedding.reshape(-1, self.num_features)
        print("...done")
        print(f"Flattened embedding shape: {self.embedding.shape}")

        self.knn = KNeighborsClassifier(n_neighbors=3, weights='distance')

        self.samples = None
        self.labels = None

    def pos_to_index(self, pos):

        z, y, x = pos
        return (
            int(z)*self.embedding_shape[2]*self.embedding_shape[1] +
            int(y)*self.embedding_shape[2] +
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

    def __init__(self, raw, embedding, classifier):

        self.raw = raw
        self.embedding = embedding
        self.classifier = classifier

        self.dimensions = neuroglancer.CoordinateSpace(
            names=['z', 'y', 'x'],
            units='nm',
            scales=raw.voxel_size)

        print(f"Creating segmentation layer with shape {raw.shape}")
        self.segmentation = np.zeros(
            raw.shape,
            dtype=np.uint8)
        self.segmentation_volume = neuroglancer.LocalVolume(
            data=self.segmentation,
            dimensions=self.dimensions)

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

        self.segmentation[:] = self.classifier.predict()
        self.update_view()

    def update_view(self):

        self.segmentation_volume.invalidate()

    def _label_fg(self, action_state):
        pos = action_state.mouse_voxel_coordinates
        if pos is None:
            return
        self.label(pos, 1)

    def _label_bg(self, action_state):
        pos = action_state.mouse_voxel_coordinates
        if pos is None:
            return
        self.label(pos, 0)


if __name__ == "__main__":

    neuroglancer.set_server_bind_address('0.0.0.0')

    args = parser.parse_args()

    raw = daisy.open_ds(args.raw_file, args.raw_dataset)
    embedding = daisy.open_ds(args.emb_file, args.emb_dataset)

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

    classifier = Classifier(embedding)
    interactive_segmentation = InteractiveSegmentation(
        raw,
        embedding,
        classifier)

    url = str(interactive_segmentation.viewer)
    print(url)
    webbrowser.open_new(url)
    print("Press ENTER to quit")
    input()

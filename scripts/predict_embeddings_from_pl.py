from tqdm import tqdm
import argparse
import configparser
import gunpowder as gp
import lisl
import torch
import logging
import numpy as np
from skimage.io import imsave
import zarr
import h5py

from lisl.pl.trainer import CPCTrainer
import daisy

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--checkpoint_path',
    help="checkpoint path to file with model weights.")
parser.add_argument('--raw_file')
parser.add_argument('--raw_dataset')


if __name__ == "__main__":

    args = parser.parse_args()

    mi = CPCTrainer()
    mi = mi.load_from_checkpoint(args.checkpoint_path)

    model = mi.unet.cuda()
    model.eval()

    print(args.raw_file)
    raw = daisy.open_ds(args.raw_file, args.raw_dataset)

    input_shape = (256, 256)
    output_shape = (120, 120)

    offset = tuple((a-b)//2 for a,b in zip(input_shape, output_shape))

    d = raw.data.shape[-3]
    w = raw.data.shape[-2]
    h = raw.data.shape[-1]

    # output = np.zeros((64, d, w, h))
    # z1 = zarr.open('cpc_embedding.zarr', mode='w', shape=(64, d, w, h),
    #                chunks=(64, 1, 256, 256), dtype='i4')

    with h5py.File("/home/swolf/local/data/tmp/cpc_embedding_full.h5", "w") as outf:
        outf.create_dataset("embedding", shape=(64, d, w, h), dtype=np.float32, chunks=(64, 1, 256, 256))

        with torch.no_grad():

            for t in range(d):
                for x in range(0, w, output_shape[-2]):
                    for y in range(0, h, output_shape[-1]):
                        if x+input_shape[-2] > w:
                            x = w - input_shape[-2] - 1

                        if y+input_shape[-1] > h:
                            y = h - input_shape[-1] - 1

                        print(t,x,y)

                        try:
                            inputdata = raw.data[:1, t:t+1, x:x+input_shape[-2], y:y+input_shape[-1]].astype(np.float32)
                            inputdata = torch.from_numpy(inputdata).cuda()

                            outputdata = model(inputdata)

                            o1, o2 = offset
                            outf["embedding"][:, t, x+o1:x+o1+output_shape[-2],
                                         y+o2:y+o2+output_shape[-1]] = outputdata[0].cpu().numpy()
                            print(model(inputdata).shape)
                        except:
                            print("error")

    print("...")
    exit()

    # checkpoint = eval(config['model']['checkpoint'])
    # raw_normalize = eval(config['predict']['raw_normalize'])
    # out_filename = eval(config['predict']['out_filename'])
    # out_ds_names = eval(config['predict']['out_ds_names'])
    # out_dir = '.'.join(config_file.split('.')[:-1])
    # if 'model_output' in config['predict']:
    #     model_output = eval(config['predict']['model_output'])
    # else:
    #     model_output = 0

    # lisl.predict.predict_volume(
    #     model,
    #     dataset,
    #     out_dir,
    #     out_filename,
    #     out_ds_names,
    #     checkpoint,
    #     raw_normalize,
    #     model_output)

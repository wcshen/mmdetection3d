# Run this script in original environment rather than conda

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from pathlib import Path
import tensorrt as trt
import pycuda.autoinit  # noqa: F401
import pycuda.driver as cuda


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file, calibration_files, batch_size, tensor_channel, tensor_height, tensor_width):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.batch_size = batch_size
        self.tensor_channel = tensor_channel
        self.tensor_height = tensor_height
        self.tensor_width = tensor_width
        self.tensor_shape = [
            self.batch_size, self.tensor_channel, self.tensor_height, self.tensor_width]

        self.calib_file_path = calibration_files
        calib_dataset = os.listdir(str(self.calib_file_path))
        print("Load calibration dataset: {} files".format(
            len(calib_dataset)))
        self.calib_files = calib_dataset

        self.cache_file = cache_file
        self.device_input = cuda.mem_alloc(
            trt.volume(self.tensor_shape) * trt.float32.itemsize)
        self.indices = np.arange(len(self.calib_files))
        np.random.shuffle(self.indices)

        def load_batches():
            for i in range(0,
                           len(self.calib_files) - self.batch_size + 1,
                           self.batch_size):
                print("======== Calibrating on Batch %d ========" %
                      (i // self.batch_size))
                indexs = self.indices[i:i + self.batch_size]
                paths = [self.calib_file_path / self.calib_files[i]
                         for i in indexs]
                files = self.read_batch_file(paths)
                yield files

        self.batches = load_batches()

    def read_batch_file(self, filenames):
        tensors = []
        for filename in filenames:
            with open(str(filename), 'rb') as f:
                input_tensor = np.fromfile(f, dtype=np.float32)
            input_tensor = input_tensor.reshape(
                (self.batch_size, self.tensor_channel, -1, self.tensor_width))
            if input_tensor.shape[2] >= self.tensor_height:
                input_tensor = input_tensor[:, :, :self.tensor_height, :]
            input_tensor = np.ascontiguousarray(input_tensor, dtype=np.float32)
            tensors.append(input_tensor)
        return np.ascontiguousarray(tensors, dtype=np.float32)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, args):
        try:
            # Assume self.batches is a generator that provides batch data.
            data = next(self.batches)
            # Assume that self.device_input is a device buffer allocated by the constructor.
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]
        except StopIteration:
            # When we're out of batches, we return either [] or None.
            # This signals to TensorRT that there is no calibration data remaining.
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, 'wb') as f:
            f.write(cache)


def create_tensorrt(model_name):
    # Load calibration files
    # calib dataset need only about 500 to 1000 images
    calib_dataset_files = Path(
        "/home/yao.xu/python_projects/OpenPCDet/tools/export_onnx/INT8_calibration_dataset/%s" % model_name)
    int8_cache_file = "%s_int8.cache" % model_name.lower()
    model_onnx_file = "%s.onnx" % model_name.lower()
    trt_output_file = "%s.engine" % model_name.lower()

    if model_name == "PFE":
        calib = EntropyCalibrator(cache_file=int8_cache_file,
                                  calibration_files=calib_dataset_files,
                                  batch_size=1,
                                  tensor_channel=10,
                                  tensor_height=24000,
                                  tensor_width=32)
    elif model_name == "RPN":
        calib = EntropyCalibrator(cache_file=int8_cache_file,
                                  calibration_files=calib_dataset_files,
                                  batch_size=1,
                                  tensor_channel=64,
                                  tensor_height=160,
                                  tensor_width=800)

    # Load ONNX model
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network()
    onnxparser = trt.OnnxParser(network, TRT_LOGGER)
    with open(model_onnx_file, 'rb') as model:
        onnxparser.parse(model.read())
    print("{} model loaded.".format(model_onnx_file))

    # Define builder params
    builder.max_batch_size = 1
    builder.max_workspace_size = 1073741824
    builder.int8_mode = True
    builder.fp16_mode = True
    builder.int8_calibrator = calib

    # Build INT8 engine
    engine = builder.build_cuda_engine(network)
    # try:
    #     engine = builder.build_cuda_engine(network)
    # except Exception as e:
    #     print("Failed creating engine for TensorRT. Error: ", e)
    #     quit()

    print("Done generating tensorRT engine.")
    print("Serializing tensorRT engine for C++ interface")
    try:
        serialized_engine = engine.serialize()
        with open(trt_output_file, "wb") as f:
            f.write(serialized_engine)
    except Exception as e:
        print("Couldn't serialize engine. Error: ", e)


if __name__ == '__main__':
    model_name = "PFE"  # 'PFE' or 'RPN'
    create_tensorrt(model_name)

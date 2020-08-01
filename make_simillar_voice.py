import numpy as np
import os
from pathlib import Path
import sox

import utils.path_utils as pu
import utils.SignalProcessingTools as spt

def noise_cut(data, noise):
    assert data.shape[0] < noise.shape[0], "noise size is short."
    if data.shape[0] == noise.shape[0]:
        return
    else:
        cut_noise = np.zeros_like(data)
        cut_noise = noise[:data.shape[0]]
    return cut_noise

def add_noise(data, noise, amplitude=1.0):
    cut_noise = noise_cut(data, noise)
    return data + cut_noise * amplitude

def sox_resampling(input_path, SAMPLINGRATE=48000):
    name = os.path.basename(input_path)
    output_path = os.path.join("./output", name)
    if not (os.path.isfile(output_path)):
        transformer = sox.Transformer()
        transformer.rate(samplerate=SAMPLINGRATE)
        transformer.build(input_path, output_path)
    data, fs = spt.read_data(output_path)
    assert fs == SAMPLINGRATE
    return data

def sox_timestretch(input_path, name, FACTOR=1.0):
    assert FACTOR != 1.0, "please set FACTOR"
    name = name + "_stretch" + str(FACTOR) + ".wav"
    output_path = os.path.join("./output", name)
    if not (os.path.isfile(output_path)):
        transformer = sox.Transformer()
        transformer.tempo(factor=FACTOR)
        transformer.build(input_path, output_path)
    data, fs = spt.read_data(output_path)
    return data

def sox_pitchshift(input_path, name, PITCHSHIFT=0.0):
    assert PITCHSHIFT != 0.0, "please set PITCHSHIFT"
    name = name + "_pitch" + str(PITCHSHIFT) + ".wav"
    output_path = os.path.join("./output", name)
    if not (os.path.isfile(output_path)):
        transformer = sox.Transformer()
        transformer.pitch(n_semitones=PITCHSHIFT)
        transformer.build(input_path, output_path)
    data, fs = spt.read_data(output_path)
    return data

def sox_lpf(input_path, name, frequency = None):
    assert frequency != None, "please chande frequency"
    name = name + "_lpf" + str(frequency) + ".wav"
    output_path = os.path.join("./output", name)
    if not (os.path.isfile(output_path)):
        transformer = sox.Transformer()
        transformer.lowpass(frequency=frequency)
        transformer.build(input_path, output_path)
    data, fs = spt.read_data(output_path)
    return data

def lpf():
    pass


def make_similer_voice(signal_path, noise_path):
    assert os.path.isfile(signal_path)
    assert os.path.isfile(noise_path)

    signal_name = pu.get_filename(signal_path)
    # data = sox_timestretch(signal_path, signal_name, FACTOR=0.75)
    # data = sox_pitchshift(signal_path, signal_name, PITCHSHIFT=-3.0)
    # data = sox_lpf(signal_path, signal_name, frequency=1000)

    data, fs = spt.read_data(signal_path)
    noise_data, fs_n = spt.read_data(noise_path)
    if fs != fs_n:
        noise_data = sox_resampling(noise_path, fs)

    add_signal = add_noise(data, noise_data, 0.3)
    # add_signal = lpf(add_signal, )

    pass

def main():
    dataset_path = Path(__file__).parent
    dataset_path /= "../../Datasets"
    el_path = os.path.join(dataset_path, "el543mic/ne100.014.wav")
    ns_path = os.path.join(dataset_path, "sp543mic/ns100.014.wav")
    noise_path = os.path.join(dataset_path, "LPC_noise/ongen_14.wav")


    make_similer_voice(ns_path, noise_path)

if __name__ == '__main__':
    main()
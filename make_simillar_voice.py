import os
from pathlib import Path

import utils.SignalProcessingTools as spt

def add_noise(data, noise, para):
    return data + noise * para

def make_similer_voice(signal_path, noise_path):
    assert os.path.isfile(signal_path)
    assert os.path.isfile(noise_path)

    data, fs = spt.read_data(signal_path)
    noise_data, fs_n = spt.read_data(noise_path)

    # Todo resampling
    assert fs == fs_n

    # Todo data augment(stretch, pitch change)

    add_noise(data, noise_data, 0.3)

    # Todo LPF

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
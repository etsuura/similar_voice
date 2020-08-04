import librosa as librosa
import librosa.display
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pathlib
from pathlib import Path
import scipy
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


def sox_lpf(input_path, name, frequency=None):
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


def _spectrogram(data, n_fft=2048, hop_length=512):
    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram_librosa = np.abs(
        librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    return spectrogram_librosa_db


def plot_spectrogram(data, title=None, output_path=None, fs=48000, n_fft=2048, hop_length=512, save=True):
    spectrogram_librosa_db = _spectrogram(data, n_fft, hop_length)
    librosa.display.specshow(spectrogram_librosa_db, sr=fs, y_axis='log', x_axis='time', hop_length=hop_length)
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save == True:
        assert title != None, "please type title."
        if output_path == None:
            cd = pathlib.Path.cwd()
            path = cd + "/output/" + title + ".png"
        else:
            title += ".png"
            path = output_path / title
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()  # グラフの初期化、初期化しないとカラーバーが複数並ぶなどある


def plot_ELVoice(el_path):
    assert os.path.isfile(el_path)

    signal_name = pu.get_filename(el_path)
    data, fs = spt.read_data(el_path)
    if fs != 48000:
        noise_data = sox_resampling(el_path, fs)
    plot_spectrogram(data, "EL voice_" + signal_name)

    el_synthesised = spt.path2synth_voice(el_path)
    plot_spectrogram(el_synthesised, "EL voice_synthesised_" + signal_name)

    pass


def remove_middleman_interval(intervals):
    if intervals.shape[0] == 1:
        return intervals
    else:
        interval = np.zeros((1, 2))
        interval[0, 0] = intervals[0, 0]
        interval[0, 1] = intervals[-1, 1]
        return interval


def make_similer_voice():
    pass


def preliminary_experiment(signal_path, noise_path):
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

    # 前後の無音の削除　要db閾値検討
    intervals = librosa.effects.split(data, top_db=40)
    interval = remove_middleman_interval(intervals)
    data = librosa.effects.remix(data, interval)
    # plot_spectrogram(data, save=False)

    plot_spectrogram(data, "natural voice_" + signal_name)
    plot_spectrogram(noise_data, "noise")

    add_signal = add_noise(data, noise_data, 0.3)
    plot_spectrogram(add_signal, "NV add noise" + signal_name)
    save_path = "./output/" + signal_name + "_add_noise"
    spt.save_wav(add_signal, fs, save_path)

    syth_nv = spt.path2synth_voice(signal_path)
    plot_spectrogram(syth_nv, "synth_nv")

    syth_nv_add_noise = spt.path2synth_voice(save_path + ".wav")
    plot_spectrogram(syth_nv_add_noise, "synth_nv_add_noise")

    # 後処理
    # spt.save_wav(add_signal, fs, "./output/ns100.014_add_noise")
    # add_signal = lpf(add_signal, )

    pass

def save_plot(input_path, output_path):
    # Todo 並列処理にしたい
    for file in list(input_path.glob('**/*.wav')):
        data, fs = spt.read_data(file)
        plot_spectrogram(data, file.stem, output_path)

def save_plot_synth(input_path, output_path):
    # Todo 並列処理にしたい
    for file in list(input_path.glob('**/*.wav')):
        # data, fs = spt.read_data(file)
        synthesized = spt.path2synth_voice(file)
        title = file.stem + "_synthesized"
        plot_spectrogram(synthesized, title, output_path)

def create_plot_all(nv_dir, el_dir):
    output_path = pathlib.Path.cwd()
    output_nv_path = output_path / "output" / "nv_plot"
    output_el_path = output_path / "output" / "el_plot"
    output_el_synth_path = output_path / "output" / "el_synth_plot"

    output_nv_path.mkdir(exist_ok=True)
    output_el_path.mkdir(exist_ok=True)
    output_el_synth_path.mkdir(exist_ok=True)

    # save_plot(nv_dir, output_nv_path)
    # save_plot(el_dir, output_el_path)
    # save_plot_synth(el_dir, output_el_synth_path)



    pass


def main():
    # dataset_path = Path(__file__).parent.parent.parent
    dataset_path = pathlib.Path.cwd().parent.parent
    nv_dir = dataset_path / "Datasets" / "sp543mic"
    el_dir = dataset_path / "Datasets" / "el543mic"

    create_plot_all(nv_dir, el_dir)

    pass


    # el_path = os.path.join(dataset_path, "el543mic/ne100.014.wav")
    # ns_path = os.path.join(dataset_path, "sp543mic/ns100.014.wav")
    # noise_path = os.path.join(dataset_path, "LPC_noise/ongen_14.wav")

    # plot_ELVoice(el_path)
    # preliminary_experiment(ns_path, noise_path)


if __name__ == '__main__':
    main()

import librosa as librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
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

def _spectrogram(data, n_fft=2048, hop_length=512):
    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram_librosa = np.abs(librosa.stft(data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hann')) ** 2
    spectrogram_librosa_db = librosa.power_to_db(spectrogram_librosa, ref=np.max)
    return spectrogram_librosa_db

def plot_spectrogram(data, title=None, fs=48000, n_fft=2048, hop_length=512, save=True):
    spectrogram_librosa_db = _spectrogram(data, n_fft, hop_length)
    librosa.display.specshow(spectrogram_librosa_db, sr=fs, y_axis='log', x_axis='time', hop_length=hop_length)
    plt.title(title)
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    if save == True:
        assert title == None, "please type title."
        cd = os.path.abspath(".")
        path = cd + "/output/" + title + ".png"
        plt.savefig(path)
    else:
        plt.show()
    plt.clf()       #グラフの初期化、初期化しないとカラーバーが複数並ぶなどある

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

    # test
    # Todo 無音区間の削除
    # 切りたいのは音声の前後のみ（≠音声中）

    # intervals = librosa.effects.split(data, top_db=43)
    # data = librosa.effects.remix(data, intervals)
    plot_spectrogram(data, save=False)


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


def main():
    dataset_path = Path(__file__).parent
    dataset_path /= "../../Datasets"
    el_path = os.path.join(dataset_path, "el543mic/ne100.014.wav")
    ns_path = os.path.join(dataset_path, "sp543mic/ns100.014.wav")
    noise_path = os.path.join(dataset_path, "LPC_noise/ongen_14.wav")

    # plot_ELVoice(el_path)
    preliminary_experiment(ns_path, noise_path)

if __name__ == '__main__':
    main()
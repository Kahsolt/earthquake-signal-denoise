#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/01

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from argparse import ArgumentParser
from traceback import print_exc, format_exc

import librosa as L
import librosa.display as LD
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns

from utils import *

SPLITS = ['train', 'test']
N_FFT_LIST = [2**i for i in range(3, 11)]   # 8~1024
HOP_LEN_LIST = [e//2 for e in N_FFT_LIST]   # 4~512
WIN_LEN_LIST = [e//2 for e in N_FFT_LIST]   # 4~512

# defaults
N_FFT   = 128
HOP_LEN = 32
WIN_LEN = 128


class App:

  def __init__(self, args):
    self.args = args
    self.X, self.Y = None, None
    self.cur_idx = None

    self.setup_gui()
    self.setup_workspace()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_workspace(self):
    self.change_split()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Signal Visualizer')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # vars
    self.var_split   = tk.StringVar(wnd, value=self.args.split)
    self.var_idx     = tk.IntVar(wnd, value=0)
    self.var_n_fft   = tk.IntVar(wnd, value=N_FFT)
    self.var_hop_len = tk.IntVar(wnd, value=HOP_LEN)
    self.var_win_len = tk.IntVar(wnd, value=WIN_LEN)

    # top: query
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:

      frm11 = ttk.Label(frm1)
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm11, text='Dataset').pack(side=tk.LEFT, expand=tk.NO)
        cb = ttk.Combobox(frm11, state='readonly', values=SPLITS, textvariable=self.var_split)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.change_split())
        cb.pack(side=tk.LEFT)

        tk.Label(frm11, text='FFT size').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm11, state='readonly', values=N_FFT_LIST, textvariable=self.var_n_fft)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

        tk.Label(frm11, text='Hop length').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm11, state='readonly', values=HOP_LEN_LIST, textvariable=self.var_hop_len)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

        tk.Label(frm11, text='Window length').pack(side=tk.LEFT)
        cb = ttk.Combobox(frm11, state='readonly', values=WIN_LEN_LIST, textvariable=self.var_win_len)
        cb.bind('<<ComboboxSelected>>', lambda evt: self.redraw())
        cb.pack(side=tk.LEFT)

      frm12 = ttk.Label(frm1)
      frm12.pack(expand=tk.YES, fill=tk.X)
      if True:
        sc = tk.Scale(frm12, command=lambda _: self.redraw(), variable=self.var_idx, orient=tk.HORIZONTAL, from_=0, to=1000, tickinterval=500, resolution=1)
        sc.pack(expand=tk.YES, fill=tk.X)
        self.sc = sc

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      fig, axs = plt.subplots(4, 2, figsize=(16, 9))
      axs: List[List[Axes]]
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      toolbar = NavigationToolbar2Tk(cvs, frm2, pack_toolbar=False)
      toolbar.update()
      toolbar.pack(side=tk.BOTTOM, fill=tk.X)
      self.fig, self.axs, self.cvs = fig, axs, cvs

  def change_split(self):
    split = self.var_split.get()
    if split == 'train':
      self.X, self.Y = get_data_train()
    else:
      self.X = get_data_test()
      self.Y = get_submit_pred_maybe(len(self.X), self.args.fp)
    nlen = len(self.X)
    self.sc.config(to=nlen - 1)

    self.var_idx.set(min(self.var_idx.get(), nlen - 1))
    self.cur_idx = -1
    self.redraw()

  def redraw(self):
    idx     = self.var_idx    .get()
    n_fft   = self.var_n_fft  .get()
    hop_len = self.var_hop_len.get()
    win_len = self.var_win_len.get()

    idx_changed = self.cur_idx != idx

    if win_len >= n_fft:
      self.var_win_len.set(n_fft)
      win_len = n_fft
    if hop_len >= n_fft:
      self.var_hop_len.set(n_fft)
      hop_len = n_fft

    try:
      for i, x in enumerate([self.X[idx], self.Y[idx]]):
        if x is None: continue
        if self.args.T == 'norm':
          x = wav_norm(x)
        elif self.args.T == 'log1p':
          x = wav_log1p(x)

        D = L.stft(x, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
        M = np.clip(np.log10(np.abs(D) + 1e-15), a_min=EPS, a_max=None)
        c0 = L.feature.rms(y=x, frame_length=n_fft, hop_length=hop_len, pad_mode='reflect')[0]
        zcr = L.feature.zero_crossing_rate(x, frame_length=n_fft, hop_length=hop_len)[0]
        fft_data = np.abs(fft(np.expand_dims(x, axis=0), axis=1).squeeze(0))
        fft_data = fft_data[:len(fft_data)//128]

        ax0, ax1, ax2, ax3 = self.axs[:, i]
        if idx_changed:
          ax0.cla() ; ax0.plot(x, c='r' if i else 'b')
        ax1.cla() ; ax1.plot(c0, label='rms') ; ax1.plot(zcr, label='zcr') ; ax1.legend(loc='upper right')
        ax2.cla() ; sns.heatmap(M, ax=ax2, cbar=False) ; ax2.invert_yaxis()
        ax3.cla() ; ax3.plot(fft_data)
        self.cvs.draw()

        self.cur_idx = idx
    except:
      info = format_exc()
      print(info)
      tkmsg.showerror('Error', info)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('--fp', type=Path, help='submit file')
  parser.add_argument('--split', choices=['train', 'test'], default='train', help='init dataset split')
  parser.add_argument('-T', choices=['', 'norm', 'log1p'], help='raw signal transform')
  args = parser.parse_args()

  App(args)

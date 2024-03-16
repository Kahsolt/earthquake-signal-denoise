#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/15

import seaborn as sns

from models import Audio2Spec
from train_melgan import phase_loss
from utils import *


def vis_cmp(X:ndarray, Y:ndarray):
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR, device='cpu')
  for x, y in tqdm(zip(X, Y)):
    x_phase = fft(torch.from_numpy(x).unsqueeze_(0).unsqueeze_(0), ret_phase=True).squeeze()
    y_phase = fft(torch.from_numpy(y).unsqueeze_(0).unsqueeze_(0), ret_phase=True).squeeze()
    loss_p = phase_loss(x_phase, y_phase).item()
    loss_l1 = F.l1_loss(x_phase, y_phase).item()

    plt.clf()
    plt.subplot(131) ; sns.heatmap(x_phase.T, vmin=-np.pi*2, vmax=np.pi*2, cbar=False, cmap='coolwarm') ; plt.gca().invert_yaxis() ; plt.title('QZ')
    plt.subplot(132) ; sns.heatmap(y_phase.T, vmin=-np.pi*2, vmax=np.pi*2, cbar=False, cmap='coolwarm') ; plt.gca().invert_yaxis() ; plt.title('CZ')
    plt.subplot(133) ; sns.heatmap((y_phase - x_phase).abs().T, vmin=-np.pi*4, vmax=np.pi*4, cbar=False, cmap='coolwarm') ; plt.gca().invert_yaxis() ; plt.title('DZ')
    plt.suptitle(f'phase loss: {loss_p:.5f} / {loss_l1:.5f}')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  X, Y = get_data_train()
  X = np.stack([wav_norm(x) for x in X], axis=0)
  Y = np.stack([wav_norm(y) for y in Y], axis=0)

  vis_cmp(X, Y)

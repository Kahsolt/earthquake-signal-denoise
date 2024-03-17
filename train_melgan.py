#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/03/13

import sys
from argparse import ArgumentParser

import yaml
import torch.nn.functional as F
from torch.optim import Adam
from lightning import seed_everything
from tensorboardX import SummaryWriter

from utils import *
from data import SignalDataset, DataLoader, make_split
from models import Audio2Spec, Generator, Discriminator, GeneratorTE

torch.backends.cudnn.benchmark = True


def phase_loss(y_hat:Tensor, y:Tensor) -> Tensor:
  diff = F.l1_loss(y_hat, y, reduction='none')
  diff_conj = 2 * np.pi - diff
  return torch.where(diff < diff_conj, diff, diff_conj).mean()


def train(args):
  seed_everything(args.seed)
  print('>> cmd:', ' '.join(sys.argv))
  print('>> args:', vars(args))

  ''' Log '''
  root = Path(args.save_path)
  root.mkdir(parents=True, exist_ok=True)
  with open(root / "args.yml", "w") as f:
    yaml.dump(args, f)
  writer = SummaryWriter(str(root))

  ''' Model '''
  has_te = args.M == 'melgan-te'
  fft = Audio2Spec(N_FFT, HOP_LEN, WIN_LEN, SR).to(device)
  if has_te:
    netG = GeneratorTE(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
  else:
    netG = Generator(args.n_mel_channels, args.ngf, args.n_residual_layers).to(device)
  netD = Discriminator(args.num_D, args.ndf, args.n_layers_D, args.downsamp_factor).to(device)
  #print(netG)
  #print(netD)

  ''' Optim '''
  optG = Adam(netG.parameters(), lr=args.lr, betas=(0.5, 0.9))
  optD = Adam(netD.parameters(), lr=args.lr, betas=(0.5, 0.9))
  load_root = Path(args.load_path) if args.load_path else None
  if load_root and load_root.exists():
    print('>> loading ckpt')
    netG.load_state_dict(torch.load(load_root / "netG.pt"))
    optG.load_state_dict(torch.load(load_root / "optG.pt"))
    netD.load_state_dict(torch.load(load_root / "netD.pt"))
    optD.load_state_dict(torch.load(load_root / "optD.pt"))

  ''' Data '''
  X, Y = get_data_train()
  traindata, validdata = make_split(X, Y, ratio=0.02)
  dataloader_kwargs = {
    'num_workers': 0,
    'persistent_workers': False,
    'pin_memory': True,
  }
  trainset = SignalDataset(traindata, transform=wav_norm, n_seg=N_SEG, aug=True)
  validset = SignalDataset(validdata, transform=wav_norm,              aug=False)
  trainloader = DataLoader(trainset, args.batch_size, shuffle=True,  drop_last=True,  **dataloader_kwargs)
  validloader = DataLoader(validset, 1, shuffle=False, drop_last=False, **dataloader_kwargs)
  del X, Y, traindata, validdata, trainset, validset

  ''' Data (test) '''
  mel_test_list: List[Tuple[Tensor, Tensor]] = []
  with torch.inference_mode():
    for i, (x_n_t, x_t, ids) in enumerate(validloader):
      if i == args.n_test_samples: break
      s_t = fft(x_t.to(device)).detach()
      s_n_t = fft(x_n_t.to(device)).detach()
      mel_test_list.append([s_n_t, ids.to(device)])
      fig = plt.figure()
      fig.gca().plot(x_t.squeeze().cpu().numpy())
      writer.add_figure("raw/wav_%d" % i, fig, 0)
      #fig = plt.figure()
      #fig.gca().imshow(s_t.squeeze().cpu().numpy(), interpolation='none')
      #writer.add_figure("raw/spec_%d" % i, fig, 0)

  ''' Train '''
  costs: List[List[float]] = []
  start = time()
  best_mel_reconst = 1000000
  steps = 0
  for epoch in range(1, args.epochs + 1):
    for batch_idx, (x_n_t, x_t, ids) in enumerate(trainloader):
      x_t = x_t.to(device)
      ids = ids.to(device)
      s_n_t = fft(x_n_t.to(device)).detach()
      x_pred_t = netG(s_n_t, ids).to(device)   # noised spec => denoised wav

      ''' Discriminator '''
      with torch.no_grad():
        s_t = fft(x_t.detach())
        s_pred_t = fft(x_pred_t.detach())
        s_error = F.l1_loss(s_pred_t, s_t)    # denoised spec <=> target clean spec

      D_fake_det = netD(x_pred_t.detach())
      D_real = netD(x_t)

      loss_D = 0
      for scale in D_fake_det:
        loss_D += F.relu(1 + scale[-1]).mean()
      for scale in D_real:
        loss_D += F.relu(1 - scale[-1]).mean()

      netD.zero_grad()
      loss_D.backward()
      optD.step()

      ''' Generator '''
      D_fake = netD(x_pred_t)
      loss_G = 0
      for scale in D_fake:
        loss_G += -scale[-1].mean()

      loss_feat = 0
      feat_weights = 4.0 / (args.n_layers_D + 1)
      D_weights = 1.0 / args.num_D
      wt = D_weights * feat_weights
      for i in range(args.num_D):
        for j in range(len(D_fake[i]) - 1):
          loss_feat += wt * F.l1_loss(D_fake[i][j], D_real[i][j].detach())

      netG.zero_grad()
      loss_G_all = loss_G + args.lambda_feat * loss_feat
      loss_G_all.backward()
      optG.step()

      # bookkeep
      costs.append([loss_D.item(), loss_G.item(), loss_feat.item(), s_error.item()])
      writer.add_scalar("loss/disc", costs[-1][0], steps)
      writer.add_scalar("loss/gen",  costs[-1][1], steps)
      writer.add_scalar("loss/fmap", costs[-1][2], steps)
      writer.add_scalar("loss/mel",  costs[-1][3], steps)
      steps += 1

      if steps % args.save_interval == 0:
        st = time()
        with torch.inference_mode():
          for i, (s_n_t, ids) in enumerate(mel_test_list):
            x_pred_t = netG(s_n_t, ids)
            s_pred_t = fft(x_pred_t)
            fig = plt.figure()
            fig.gca().plot(x_pred_t.squeeze().cpu().numpy())
            writer.add_figure("gen/wav_%d" % i, fig, epoch)
            #fig = plt.figure()
            #fig.gca().imshow(s_pred_t.squeeze().cpu().numpy(), interpolation='none')
            #writer.add_figure("gen/spec_%d" % i, fig, epoch)

        torch.save(netG.state_dict(), root / "netG.pt")
        torch.save(optG.state_dict(), root / "optG.pt")
        torch.save(netD.state_dict(), root / "netD.pt")
        torch.save(optD.state_dict(), root / "optD.pt")

        if np.asarray(costs).mean(0)[-1] < best_mel_reconst:
          best_mel_reconst = np.asarray(costs).mean(0)[-1]
          torch.save(netD.state_dict(), root / "best_netD.pt")
          torch.save(netG.state_dict(), root / "best_netG.pt")

        print("Took %5.4fs to generate samples" % (time() - st))
        print("-" * 100)

      if steps % args.log_interval == 0:
        print(
          "[Epoch {}] Iters {} / {}: loss {} ({:5.2f} ms/b)".format(
            epoch,
            batch_idx,
            len(trainloader),
            np.asarray(costs).mean(0),
            1000 * (time() - start) / args.log_interval,
          )
        )
        costs = []
        start = time()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', default='melgan', choices=['melgan', 'melgan-te'])
  parser.add_argument("--save_path", required=True)
  parser.add_argument("--load_path", default=None)
  parser.add_argument("--n_mel_channels", type=int, default=N_SPEC)
  parser.add_argument("--ngf", type=int, default=32)
  parser.add_argument("--n_residual_layers", type=int, default=3)
  parser.add_argument("--ndf", type=int, default=16)
  parser.add_argument("--num_D", type=int, default=3)
  parser.add_argument("--n_layers_D", type=int, default=4)
  parser.add_argument("--downsamp_factor", type=int, default=4)
  parser.add_argument("--lambda_feat", type=float, default=10)
  parser.add_argument("--cond_disc", action="store_true")
  parser.add_argument("--data_path", default=None, type=Path)
  parser.add_argument("--batch_size", type=int, default=16)
  parser.add_argument("--seq_len", type=int, default=N_SEG)
  parser.add_argument("--epochs", type=int, default=3000)
  parser.add_argument("--log_interval", type=int, default=100)
  parser.add_argument("--save_interval", type=int, default=1000)
  parser.add_argument("--n_test_samples", type=int, default=8)
  parser.add_argument('-lr', '--lr', type=eval, default=1e-4)
  parser.add_argument('--seed', type=int, default=114514)
  args = parser.parse_args()

  train(args)

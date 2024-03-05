#!/usr/bin/env python3
# Author: Armit
# Create Time: 2024/02/16

import json
from zipfile import ZipFile

from utils import *


def process_zipfile(fp:Path) -> Tuple[ndarray, List[str], Dict[str, int]]:
  get_name = lambda zinfo: Path(zinfo.filename).stem[:-2]   # ignore '_Q' / '_C'
  zf = ZipFile(fp)
  zinfos = zf.infolist()
  zinfos.sort(key=get_name)
  X, fns, lens = [], [], {}
  for zinfo in tqdm(zinfos):
    if zinfo.is_dir(): continue
    fn = get_name(zinfo)
    fns.append(fn)
    with zf.open(zinfo) as fh:
      x = np.loadtxt(fh, dtype=np.float32)
    xlen = len(x)
    if xlen != NLEN:
      lens[fn] = xlen
      print(f'>> {fn} wrong len: {xlen}')
      x = np.pad(x, (0, NLEN - xlen))
    X.append(x)
  X = np.stack(X, axis=0).astype(np.float32)
  return X, fns, lens


def process_test():
  fp_out = DATA_PATH / 'test.npz'
  if fp_out.exists():
    print(f'>> ignore due to file exists: {fp_out.name}')
    return

  fp_in = DATA_PATH / 'test_img.zip'
  X, fns, rec = process_zipfile(fp_in)
  data = {'X': X}

  np.savez_compressed(fp_out, **data)
  with open(fp_out.with_suffix('.txt'), 'w', encoding='utf-8') as fh:
    for fn in fns:
      fh.write(fn)
      fh.write('\n')
  with open(fp_out.with_suffix('.json'), 'w', encoding='utf-8') as fh:
    json.dump(rec, fh, indent=2, ensure_ascii=False)


def process_train():
  fp_out = DATA_PATH / 'train.npz'
  if fp_out.exists():
    print(f'>> ignore due to file exists: {fp_out.name}')
    return

  data_dict = {}   # 由 强震信号QZ 推 测震信号CZ
  lens_dict = {}
  fns_list = []
  for fn, what in zip(['train_img_QZ.zip', 'train_img_CZ.zip'], ['X', 'Y']):
    fp_in = DATA_PATH / fn
    X, fns, lens = process_zipfile(fp_in)
    data_dict[what] = X
    lens_dict[what] = lens
    fns_list.append(fns)

  # sanity check
  fns_QZ, fns_CZ = fns_list
  assert all([x == y for x, y in zip(fns_QZ, fns_CZ)])

  np.savez_compressed(fp_out, **data_dict)
  with open(fp_out.with_suffix('.json'), 'w', encoding='utf-8') as fh:
    json.dump(lens_dict, fh, indent=2, ensure_ascii=False)


if __name__ == '__main__':
  process_test()
  process_train()

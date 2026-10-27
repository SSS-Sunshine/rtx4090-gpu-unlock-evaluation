# Third-party test sources

- `peer_matrix.cu` and `owner_verify.cu` are copied from
  [`xiaoyanzi191/rtx4090-small-bar-p2p`](https://github.com/xiaoyanzi191/rtx4090-small-bar-p2p)
  at commit `dbd0cdf`; see `LICENSE.xiaoyanzi191`.
- `p2pcheck.cu` and `cumem_resident_p2p.cu` are copied from the
  `595.84-p2p-48g` branch of
  [`mochgolf/open-gpu-kernel-modules`](https://github.com/mochgolf/open-gpu-kernel-modules)
  at commit `68c31e9`; see `COPYING.open-gpu-kernel-modules`.

These sources are included to make the published correctness and bandwidth
evidence reproducible. The repository's other files do not change the licenses
of these third-party files.

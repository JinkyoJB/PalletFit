# External baseline repositories

The Experiment 2 baselines are **not vendored** in this repository. To reproduce
them, obtain each upstream repo and make it available in this directory, either
by setting the corresponding variable in your `.env` (see the project-root
`.env.example`) or by creating a symlink here.

| Baseline          | Upstream                                              | `.env` var  | Symlink name |
|-------------------|-------------------------------------------------------|-------------|--------------|
| Online-3D-BPP-PCT | https://github.com/alexfrom0815/Online-3D-BPP-PCT     | `PCT_REPO`  | `PCT_Repo`   |
| Online-3D-BPP-DRL | https://github.com/alexfrom0815/Online-3D-BPP-DRL     | `DRL_REPO`  | `DRL_Repo`   |
| GOPT              | https://github.com/Xiong5Heng/GOPT                    | `GOPT_REPO` | `GOPT_Repo`  |

Example (symlink approach):

```bash
ln -s /path/to/Online-3D-BPP-DRL  DRL_Repo
ln -s /path/to/Online-3D-BPP-PCT  PCT_Repo
ln -s /path/to/GOPT               GOPT_Repo
```

These symlinks are git-ignored and are never committed (they point to
machine-specific paths). Only this README is tracked.

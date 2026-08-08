# tm

按 task list 顺序执行命令，可以先等卡再上机。

一条命令跑完且退出码为 0，才跑下一条；哪一步失败就立即终止，
并把那一步的命令、退出码、输出末尾原样报出来。

单文件，只依赖 `pyyaml`。挂在 `screen` / `tmux` 里，ssh 断了不影响。

```bash
cd ~/task_manager
./tm.py                                # 跑 ./task_list.yaml
./tm.py --config lists/ccfm_c.yaml     # 跑指定的清单
./tm.py --config lists/ccfm_c.yaml --dry-run
```

## task list 长什么样

```yaml
wait:                    # 可选：等条件满足再开跑
  after_pid: 243146      # 等这个已经在跑的进程结束
  gpu_free_gb: 50        # 等到有卡空闲显存 >= 50 GiB
  stable_for: 120        # 条件要连续满足 120 秒才算数

cwd: ~/my_project  # 所有命令的工作目录

vars:                    # {KEY} 占位符，几步共用一个 config 就提到这里
  CFG: config/ccfm_2p5d_fused_gauss_tsample.yaml

tasks:
  - name: train
    cmd: python train.py --config {CFG}
  - name: test
    cmd: python test.py --config {CFG} --run multiout_fused --cuda 0
```

顶层的键：

| 键 | 说明 |
|---|---|
| `tasks` | **必填**，命令列表，按顺序执行 |
| `cwd` | 所有命令的工作目录，默认是 yaml 文件自己所在的目录 |
| `vars` | `{KEY}` 占位符的值，可被 `-v KEY=VALUE` 覆盖 |
| `name` | 本次运行的名字，进日志文件名，默认取文件名 |
| `wait` | 开跑前等什么，见下 |

每个 task 是 `{name, cmd}`；只写一个字符串也认，名字自动叫 `step1`、`step2`。
命令交给 bash 执行，`&&`、管道、`$(...)`、环境变量前缀都能用：

```yaml
tasks:
  - name: test
    cmd: python test.py --config {CFG} --ckpt "$(ls -t {CKPT_DIR}/*.ckpt | head -1)"
```

`$(...)` 是在**这一步真正启动时**才求值的，所以能拿到上一步刚写出来的 ckpt。

命令很长用 YAML 折叠语法 `>`。**`{KEY}` 没定义会直接报错退出**，不静默放过打错的路径。
命令里要用字面花括号写 `{{` / `}}`（python f-string、`awk '{{print $1}}'`）。

## 等卡（抢卡脚本）

`wait:` 块下面的键：

| 键 | 默认 | 说明 |
|---|---|---|
| `after_pid` | 无 | 先等这个已经在跑的进程结束 |
| `gpu_free_gb` | 无 | 再等到有卡空闲显存 ≥ 这个数（**GiB**，`nvidia-smi` 的 MiB ÷ 1024） |
| `gpus` | `1` | 需要几张卡 |
| `gpu_index` | `any` | 限定候选卡：`any` / `0` / `[0, 1]` |
| `stable_for` | `120` | 条件要连续满足多少秒才算数 |
| `poll` | `30` | 轮询间隔秒 |
| `timeout` | 无 | 等这么久还没等到就放弃 |
| `on_oom` | `requeue` | 抢输了怎么办：`requeue` 退回去重抢 / `stop` 直接终止 |
| `max_retries` | `5` | 最多重抢几次，`0` = 不限 |

两个条件可以单独用也可以一起用。**`after_pid` 不是 `gpu_free_gb` 的冗余**：
显存空出来不等于你那个 train 跑完了，test 必须排在 train 后面。

命令行可以临时覆盖：`--wait-pid PID`、`--gpu-free GiB`、`--no-wait`。

### `stable_for` 不是可有可无的

裸的「free ≥ N 就上」会踩这个坑：别人的任务刚启动、正在读数据还没建显存池，
`nvidia-smi` 看着卡是空的；这时候冲进去，30 秒后两边一起 OOM。
要求条件**连续满足**两分钟能挡掉绝大部分这种窗口，代价只是多等两分钟。

### tm 接管显卡时，config 的 `devices` 必须写 `[0]`

tm 选好卡后会 `export CUDA_VISIBLE_DEVICES=<物理卡号>`，子进程眼里就只剩一张卡、
编号 0。config 再写 `trainer.devices: [1]` 会报错找不到卡，`test.py --cuda 1` 同理。

所以 **tm 开跑前就把这件事查掉**：它会从每条命令里抠出 `--config <path>`，
读那份 yaml 检查 `trainer.devices`，不对就直接退出——而不是让你等了六小时才发现。

```
error: config device check failed
  - task 'train': /home/me/my_project/config/ddpm_fdg.yaml
      trainer.devices is [1], must be [0] (or 1) — tm assigns the physical card via CUDA_VISIBLE_DEVICES
  - task 'test': --cuda 1 — tm remaps the card, so it must be 0
```

想拿物理卡号（写日志、命名输出目录之类），用 `{GPU}` 占位符。
不需要这套检查就加 `--no-device-check`。

### 抢输了怎么判定

只有**三条同时成立**才算「卡被抢走了」，才退回去重抢：

1. 挂的是**第一个**任务
2. 启动后 **10 分钟内**就挂了
3. 输出里有明确的显存不足（`CUDA out of memory` / `OutOfMemoryError` …）

跑到第 3 个 epoch 才 OOM 那是真问题，重抢没有意义，会当普通失败处理。
`max_retries` 默认 5：`gpu_free_gb` 填小了的话，每次抢到都 OOM，不设上限会无限循环。

### 仍然需要 guard 的场景

`after_pid` 等的进程不是 tm 的子进程，Linux 上拿不到非子进程的退出码——
tm 只知道它**结束了**，不知道它是跑完了还是崩了。所以：

- 链条是**全新的 train** → 不用管，跟前一个任务无关
- 链条是「接着我那个 train 跑 test」→ **加一步守卫任务**，确认产物真的完整

```yaml
tasks:
  - name: guard
    cmd: >
      python -c "import torch, sys;
      e = torch.load('{CKPT_DIR}/last.ckpt', map_location='meta', weights_only=False)['epoch'];
      print(f'last.ckpt epoch = {{e}} (need >= {LAST_EPOCH})');
      sys.exit(0 if e >= {LAST_EPOCH} else 1)"
```

`map_location='meta'` 只读元信息不加载权重，1 秒出结果。训练没跑满就停在这步。

## 残留进程清理

OOM 之后主进程死了、dataloader worker 和 DDP 的其他 rank 还活着继续占显存——
这种得手动 `kill` 的情况，tm 每个任务跑完都会自动处理一次。

做法是每个任务用 `setsid` 起一个独立的进程组（`pgid` == 那条命令的 pid），
任务结束后整组 `SIGTERM` → 5 秒 → `SIGKILL`：

```
<== [1/2] train  python train.py --config config/xxx.yaml
    3 leftover process(es) still alive, killing:
      256107  python train.py --config config/xxx.yaml
      256108  python train.py --config config/xxx.yaml
```

两个关键点：

- **按进程组捞，不按父子关系。** 主进程一死，worker 就被 init 收养、`ppid` 变成 1，
  顺着父子关系找不到它们；但 `setsid` 建的 `pgid` 不会变。
- **不用问 `nvidia-smi` 谁占了显存。** 那个组里只可能有 tm 自己起的东西，
  所以无条件杀是安全的，不存在误杀别人任务的可能。

成功的任务也照样清一遍。重抢之前也会先清干净，否则是在跟自己上一轮的残留抢卡。

代价是子进程 `setsid` 出去了，终端的 Ctrl-C 不再自动广播给它——改由 tm 转发给整个组，
`SIGINT` → `SIGTERM` → `SIGKILL` 逐级升级。

主进程**根本不退出**的情况（OOM 之后卡在 NCCL barrier 上等一个已经挂了的 rank）
不在这套机制里：tm 会一直阻塞在等它结束上。症状是 tm 不动了，不是显存没释放。

## 出错的时候

立即终止，后面的步骤标 `skip`，退出码 1。终端上打印：

```
======================================================================
FAILED at step 2/3: test
======================================================================
  exit code : 1
  duration  : 12m03s
  cwd       : /home/me/my_project
  gpu       : 1 (CUDA_VISIBLE_DEVICES)
  command   : python test.py --config config/xxx.yaml --run multiout_fused --cuda 0

  --- last 8 lines of output --------------------------------
  | Traceback (most recent call last):
  | FileNotFoundError: no such checkpoint: ...
  ------------------------------------------------------------

  full log  : /home/me/.tm/logs/20260808_1432_task_list.log
```

`command` 是真正交给 shell 的那一行，可以直接复制重跑。完整输出全在日志里，一个字不丢。
`grep -n '^==>' <log>` 列出每步的起点行号。

## 日志

输出同时进终端和日志文件，默认 `~/.tm/logs/<时间戳>_<名字>.log`，`--log` 可改。

终端里保留颜色和进度条（走 pty，Lightning / tqdm 的进度条正常显示）；
写进日志时剥掉 ANSI，并把 `\r` 原地刷新的进度条折叠成每行最终状态——
三万次 tqdm 更新在日志里只占一行。

## 命令行选项

| 选项 | 作用 |
|---|---|
| `-c, --config YAML` | 要跑的 task list，默认 `./task_list.yaml` |
| `-v KEY=VALUE` | 覆盖 `vars` 里的值，可重复 |
| `-C, --cwd DIR` | 覆盖 `cwd` |
| `-w, --wait-pid PID` | 覆盖 `wait.after_pid` |
| `-g, --gpu-free GiB` | 覆盖 `wait.gpu_free_gb` |
| `--no-wait` | 忽略整个 `wait` 块，立刻开跑 |
| `-n, --name NAME` | 本次运行的名字 |
| `--log PATH` | 日志路径 |
| `--dry-run` | 打印展开后的计划和当前显存，不执行 |
| `--no-device-check` | 跳过 `trainer.devices` 检查 |
| `--no-pty` | 不分配伪终端，输出显示不正常时用 |

## 退出码

`0` 全部成功 · `1` 有一步失败 / 被 Ctrl-C 中断 / 重抢用尽 · `2` 参数或 yaml 写错

Ctrl-C 会把整个进程组杀掉（SIGINT → SIGTERM → SIGKILL）并停止整个清单，
不留孤儿进程占显存，见「残留进程清理」。等卡期间 Ctrl-C 则一步都不执行。

## 不做什么

不排队、不做后台常驻（用 `screen`）、不重试失败的任务、**不做多实例互斥**。

最后一条要注意：同时挂两个 tm 抢卡，它们会**同时**看到卡空出来、一起冲上去。
新任务从启动到显存出现在 `nvidia-smi` 里有几十秒空窗，`stable_for` 挡不住这种情况。
同时只挂一个 tm。

## YAML 的坑

别用 `on:` 当键名——YAML 1.1 会把裸的 `on` / `off` / `yes` / `no` 解析成布尔值
（GitHub Actions 那个著名的坑）。所以「限定哪张卡」这个键叫 `gpu_index` 而不是 `on`。
真写了 `on:` 的话 tm 会认出来并提示你。

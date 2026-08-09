# tm

把 task list 排进队列，等到卡就自动上机。每个 task 跑在自己的 tmux 会话里。

一条命令跑完且退出码为 0 才跑下一条；哪一步失败就停在那里，pane 保留现场等你 attach。

只依赖 `pyyaml` 和 `tmux`（用到 `wait.gpu_free_gb` 时还要 `nvidia-smi`）。

```bash
ln -s ~/task_manager/tm.py ~/.local/bin/tm    # 装一下，之后在哪都能用

tm add lists/ccfm_c.yaml     # 加进队列
tm                           # 起调度器（建议挂在 tmux 里）
tm ls                        # 看进度。tm 在不在跑都能用
```

## 它是怎么组织的

tm 自己**不持有任何权威状态**。队列是一个目录，进度是任务自己写下的退出码文件，
全都在 `~/.tm/` 底下。所以：

- tm 随时可以被杀掉重启，进度不丢
- `tm ls` 是另一个进程，tm 没运行时照样能看
- 你可以在 tm 没跑的时候手改队列

```
~/.tm/
  lock                        单实例互斥（flock，进程一死内核自动释放）
  queue/010_ccfm_c.yaml       在这儿 = 还没开始。手动 cp 进来等价于 tm add
  runs/20260808_143301_ccfm_c/
    list.yaml                 从队列移过来的原件
    run.yaml                  tm 写的「计划」：状态、抢到的卡、每步的会话名和最终命令
    01.sh                     真正交给 tmux 执行的包装脚本
    01.rc                     *task 自己写的「结果」* —— 唯一的完成凭据
    events.log
```

核心划分是 **tm 写计划，task 写结果**。tm 死了，rc 文件照样在长；rc 文件是什么，tm 说了不算。

## 命令

```
tm [-v KEY=VALUE] [--root DIR] <子命令>
```

| 子命令 | 作用 |
|---|---|
| `tm` / `tm run` | 消费队列。默认常驻，队列空了就待命 |
| `tm ls` | 看进度：在跑的、排队的、最近跑完的、每张卡的显存 |
| `tm add F...` | 把 yaml 加进队列（先验一遍语法，坏的不放进去） |
| `tm check F` | 只解析不跑，看看变量展开成什么样、会等什么卡 |
| `tm attach [名字]` | attach 到正在跑的 task；有多个就列出来让你选 |
| `tm clean [-y]` | 清掉失败留下的 tmux 会话。默认只列出来，`-y` 才真杀 |

| 选项 | 属于 | 作用 |
|---|---|---|
| `-v KEY=VALUE` | 全局 | 覆盖 yaml 里 `vars:` 的值，可重复。**必须写在子命令前面** |
| `--root DIR` | 全局 | 覆盖 `~/.tm`，也可以用环境变量 `TM_ROOT`。测试时用 |
| `--poll SEC` | `run` | 轮询间隔，默认 10 秒 |
| `--once` | `run` | 队列跑空就退出，不待命 |
| `--no-device-check` | `run` | 跳过 `trainer.devices` 静态检查 |
| `-n N` | `ls` | 显示最近几个跑完的，默认 5 |
| `--seq N` | `add` | 指定排序号，默认排到最后 |

`tm run` 的退出码说的是**调度器**的事，不是任务的事：`0` 正常结束 ·
`2` 已经有一个 tm 在跑 / `~/.tm` 写不了 · `130` Ctrl-C。
任务成败去 `tm ls` 看——`tm run --once` 即使有 list 失败了也返回 0。

Ctrl-C 只停调度，**已经起来的 task 不受影响**，还在各自的 tmux 里跑着。重新 `tm` 会接着推进。

## task list 长什么样

```yaml
wait:                    # 可选：等卡够用了再上机。不写这块就是排到就立刻开跑
  gpu_free_gb: 50        # 等到有卡的空闲显存 >= 50 GiB
  gpus: 1                # 要几张
  gpu_index: any         # any | 0 | [0, 1]
  stable_for: 120        # 条件要连续满足 120 秒才算数

cwd: ~/my_project  # 所有命令的工作目录

vars:                    # {KEY} 占位符
  CFG: config/ccfm_2p5d_fused_gauss_tsample.yaml

tasks:
  - name: train
    cmd: python train.py --config {CFG}
  - name: test
    cmd: python test.py --config {CFG} --run multiout_fused --cuda 0
```

| 顶层键 | 说明 |
|---|---|
| `tasks` | **必填**，按顺序执行 |
| `cwd` | 工作目录，默认是 **yaml 文件自己所在的目录**（不是你 cd 到哪） |
| `vars` | `{KEY}` 的值，`-v KEY=VALUE` 可覆盖 |
| `name` | 这个 list 的名字，进 run 目录名和 tmux 会话名。默认取文件名（剥掉 `010_` 这种排序前缀） |
| `wait` | 上机条件，见下 |

每个 task 是 `{name, cmd}`；只写一个字符串也认，名字自动叫 `step1`、`step2`。
**名字只能用字母、数字、`_`、`-`** —— 它会变成 tmux 会话名的一部分，
而 `.` 和 `:` 是 tmux target 语法的分隔符。

`{KEY}` 没定义会直接报错退出，不静默放过打错的路径。要写字面花括号就用 `{{` / `}}`
（python 的 f-string、awk 的 `{{print $1}}`）。`{GPU}` 是特殊的：它等抢到卡之后才展开，
拿到的是**物理卡号**，适合用来命名输出目录。

命令交给 bash 执行，`&&`、管道、`$(...)`、环境变量前缀都能用。
`$(...)` 是在**这一步真正启动时**才求值的，所以能拿到上一步刚写出来的 ckpt：

```yaml
- name: test
  cmd: python test.py --config {CFG} --ckpt "$(ls -t {CKPT_DIR}/*.ckpt | head -1)"
```

### 等卡

| 键 | 默认 | 说明 |
|---|---|---|
| `gpu_free_gb` | 无 | 要求空闲显存 ≥ 这个数（**GiB**，`nvidia-smi` 的 MiB ÷ 1024）。不写 = 不管显卡 |
| `gpus` | `1` | 要几张卡 |
| `gpu_index` | `any` | 限定候选卡：`any` / `0` / `[0, 1]`。列的张数不能少于 `gpus` |
| `stable_for` | `120` | 条件要连续满足多少秒才算数 |
| `timeout` | 无 | 等这么久还没等到就放弃这个 list（标成 `timeout`，不挡住后面的） |

`stable_for` 不是可有可无的：别人的任务刚启动、正在读数据还没建显存池，
`nvidia-smi` 看着卡是空的；这时候冲进去，30 秒后两边一起 OOM。
要求条件**连续满足**两分钟能挡掉绝大部分这种窗口，代价只是多等两分钟。

别用 `on:` 当键名——YAML 1.1 会把裸的 `on`/`off`/`yes`/`no` 解析成布尔值
（GitHub Actions 那个著名的坑）。所以「限定哪张卡」这个键叫 `gpu_index`。

## 调度

**队列顺序就是优先级**，也就是文件名顺序。`tm add` 会自动编号（`010_`、`020_`…），
改优先级就是改文件名，取消就是 `rm`。tm 每个 tick 重扫目录，
所以**在线离线走的是同一条路径**，没有特权通道。

每个 tick 从队列顶往下扫一遍：

```
reserved = set()
for 每个待跑的 list（文件名顺序）:
    能用的 = 稳定空闲的卡 - 已分配 - reserved
    够  -> 一次拿满，起飞
    不够 -> 把它够得着的空卡塞进 reserved，后面的 list 这个 tick 别想碰
```

- **原子获取消灭死锁**：一个 list 要么全拿到（在跑，不等任何东西），要么一张不拿（在等，
  不占任何东西）。不存在「持有并等待」，构不成环。
- **reserved 消灭饥饿**：要 2 张卡的 A 不会被排在后面、只要 1 张的 B 一张张叼光。
- 代价是卡会空转着等 A 凑齐。有意换来的可预测性。

「车道」（卡 0 跑一串、卡 1 跑另一串）不是一个独立概念，它从 `gpu_index` 自己长出来：
两个 list 都写 `gpu_index: 0` 就自动串行，写不同的卡就自动并行。

一台机器上**同时只能有一个 tm 在调度**（flock）。第二个会直接报错退出，
不会跟第一个抢卡。

## 怎么判断一步跑完了

tm 不是 task 的父进程——真正 `wait()` 到退出码的是 tmux pane 里那个 bash。
它把退出码写进 `NN.rc`，这是唯一的跨进程通道。由此得到三态：

| 会话 | rc 文件 | 判定 |
|---|---|---|
| 在 | 无 | 还在跑 |
| — | 有 | 结束了，退出码就是文件内容 |
| **没** | **无** | **LOST —— 死了没来得及报告，当失败** |

第四行是承重墙。少了它，一个被 `kill -9` / OOM killer / 机器重启带走的 train
会被读成成功，然后 test 抱着半个 checkpoint 跑下去。

整个设计只保证一个方向：**可能把成功误报成失败，绝不会把失败误报成成功。**
（写不进盘 → 无 rc → 当失败；被硬杀 → 无 rc → 当失败。
所以 tm 开跑前会先验一遍 `~/.tm` 可写，不然跑到一半才发现太亏。）

### 包装脚本

每一步真正被执行的东西会落成 run 目录里的 `NN.sh`：

```bash
#!/usr/bin/bash
# tm: tm-ccfm_c-01-train
set -o pipefail
cd /home/me/my_project || exit 1

( python train.py --config config/xxx.yaml )
rc=$?

echo $rc > .../01.rc.tmp && mv .../01.rc.tmp .../01.rc

[ $rc -eq 0 ] && exit 0
exec bash -i
```

几个点值得知道：

- **`set -o pipefail`**：POSIX 里管道的退出码只看最后一个命令，
  `python train.py | tee log` 里 train 崩了也会得到 0。开了 pipefail 才拿得到真值。
  如果你**故意**想忽略某一段的失败，自己写 `|| true`。
- **`( )` 子 shell**：命令自己写了 `exit`（或者以 `exec` 收尾）的话，
  不套子 shell 会把包装一起带走，rc 文件永远写不出来。
- **先写 `.tmp` 再 `mv`**：rename 是原子的，rc 文件要么不存在要么内容完整。
- **成功就自己消失，失败就钉在原地**：成功的会话自动退出不留垃圾；
  失败的 pane 保留完整 scrollback，attach 进去是个站在 job 的 cwd 和环境里的
  交互 shell（conda 是活的），可以就地查。
- 想复现某一步，直接 `bash NN.sh`。

tm 注入两个环境变量：`PYTHONUNBUFFERED=1`，以及抢到卡时的 `CUDA_VISIBLE_DEVICES`。

## 出错的时候

失败那步的 pane 保留着，同时 tm 会把最后 12 行直接打出来，不用 attach 就能看到报错：

```
<== ccfm_c FAILED at step 2/3 (test) rc=1
    --- tm-ccfm_c-02-test 最后 12 行 ------------------------------
    | Traceback (most recent call last):
    | FileNotFoundError: no such checkpoint: ...
    现场还在：tmux attach -t tm-ccfm_c-02-test
    /home/me/.tm/runs/20260808_143301_ccfm_c
```

后面的步骤不会跑。`tm ls` 里这个 list 显示成 `FAILED  ccfm_c  1/3`。
查完了用 `tm clean -y` 把留下的会话清掉（在跑的一律不动）。

`tm ls` 大概长这样：

```
tm: pid 31337 since 2026-08-08 14:20:11

RUNNING
  ccfm_c         gpu1     [2/3] test          1h04m   -> tmux attach -t tm-ccfm_c-02-test

QUEUED   (顺序 = 优先级，改文件名即可调整)
  020_ddpm_fdg.yaml        2 tasks   1x50GiB on any gpu

RECENT
  ok      tok            3/3       3s  08-08 23:49
  FAILED  tpipe          0/2       1s  08-08 23:49  step1 piped rc=3

GPUS
  gpu0:   52.8/95.6 GiB free   util 100%
  gpu1:   94.9/95.6 GiB free   util   0%
```

### 卡住只报警，不自动 kill

pane 超过 30 分钟没有输出，`tm ls` 里标黄 `⚠ silent 42m10s`。但 tm 不会动它——
存 checkpoint 的几十秒、epoch 之间、CPU-bound 的 eval 都会长时间安静，
自动 kill 迟早有一天会掐死一个正在存 checkpoint 的健康任务，
而那恰好是最不能被打断的时刻。要不要杀你自己判断。

## config 的 `devices` 必须写相对编号

tm 选好卡后会设 `CUDA_VISIBLE_DEVICES=<物理卡号>`，子进程眼里就只剩 `gpus` 张卡、
编号从 0 开始。config 再写 `trainer.devices: [1]` 会报错找不到卡，`test.py --cuda 1` 同理。

所以 **tm 开跑前就把这件事查掉**：从每条命令里抠出 `--config <path>`，
读那份 yaml 检查 `trainer.devices`，不对就不让它上机——而不是让你等了六小时才发现。

```
tm: ddpm_fdg config device check 不通过，跳过：
  - task 'train': /home/me/my_project/config/ddpm_fdg.yaml
      trainer.devices is [1], must be [0] (or 1) — tm assigns the physical card via CUDA_VISIBLE_DEVICES
```

要物理卡号（写日志、命名输出目录）就用 `{GPU}`。不需要这套检查加 `--no-device-check`。

## 日志

**tm 不管日志。** 每个 task 在自己的 tmux 会话里跑，输出就在那个 pane 的 scrollback 里，
`tmux attach` 或 `tm attach` 直接看，进度条颜色一切正常。要留存档就在命令里自己
`| tee train.log`（pipefail 已经开了，train 崩了不会被 tee 吃掉）。

这是有意的：训练脚本本来就自己在写 log，tm 再抄一份是重复的。

## 不做什么

- **不自动重试。** 抢卡冲突已经用 `stable_for` 在事前挡了，事前避免比事后重试好——
  重试的前提是你已经浪费了一次启动。而真正的 bug、config 写错、NaN，重跑一遍还是会炸。
  supervisor 的价值不是「爬起来」，是**「不做错事」**：train 炸了别拿半个 checkpoint 去跑 test。
- **不自动 kill 卡住的任务**（见上）。
- **不保证 task 之间的语义正确性。** tm 只看退出码，它不知道 train 有没有写出 checkpoint、
  test 读的是不是那个 checkpoint。这是你写 task list 时自己要保证的。
- **不做资源公平调度。** 调度权在你手里，队列顺序就是优先级。
- **一个 list 内部的 task 是串行的。** 并行（用于可分片的数据预处理）计划中，还没实现。

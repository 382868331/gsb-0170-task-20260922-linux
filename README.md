# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

隔离 venv 准备（可选但推荐；不安装任何第三方包）：

```sh
python3 -m venv .venv
. .venv/bin/activate   # 之后直接用 python 即可
```

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 一次正确复制 + 一次可见性拒绝（约 0.1 秒）
python demo_iteration.py                   # 周期采集演示：慢消费者丢段/整圈/停止重启/计数回绕（< 90 秒）
```

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果
- `Owner.{CPU, DEVICE}`

### `DmaRing(capacity, pool_size)`

- `capacity`：2 的幂，∈ [1, 64]；`pool_size` ≥ 0。`ring.pool` 为共享字节池。
- CPU 侧：`cpu_write(index, *, offset, length, next=None)`（暂存，不落可见状态）、
  `cpu_release(index) -> int`（flush 并交给 device，返回新 generation）、
  `cpu_acquire(index) -> CompletionSnapshot`、`cpu_read_completion(index)`
- device 侧：`device_acquire(index, *, expected_generation=None) -> DescriptorSnapshot`、
  `device_read(index)`、`device_release(index, *, status, bytes_done=0, error=None)`
- 链与自省：`walk_chain(head) -> list[int]`（整链预验证）、
  `completion_count(index)`（每项完成次数）、`inspect(index)`（带外调试视图，不属于协议）

### `CopyEngine(ring)`（合成设备，同步复制参考）

- `submit_chain(head)`、`run()`、`reset_halt()`
- 输出：`engine.output`（复制字节结果）、`engine.results`（`ChainResult` 列表）、
  `engine.queued`、`engine.halted`

## 所有权与可见性协议

1. CPU `cpu_write` 只写暂存视图；`cpu_release` 一次性提交（flush）并把所有权交给
   device，generation +1。此前 device 侧任何 acquire/read 都被拒绝。
2. device 必须 `device_acquire` 后才能 `device_read`；可选 `expected_generation`
   用于检测环回复用造成的 ABA（不匹配抛 `GenerationError`）。
3. device 完成后 `device_release` 发布完成记录并归还 CPU。
4. CPU 必须 `cpu_acquire` 后才能 `cpu_read_completion`；device 未 release 时
   `cpu_acquire` 抛 `OwnershipError`。

## 错误语义（均可定位，不静默纠正）

| 异常 | 触发 |
|---|---|
| `TypeError` | 整数参数为 bool、非整浮点、非数值类型 |
| `ValueError` | NaN/Infinity、负数 offset/length、非法 capacity/status/bytes_done |
| `IndexError` | 环索引越界 |
| `OwnershipError` | 非持有方写/读/acquire/release（含缺 release） |
| `VisibilityError` | 未 acquire 就读描述符或完成字段 |
| `GenerationError` | expected_generation 与已提交 generation 不符（ABA） |
| `ChainError` | 整链验证失败；携带 `.index` 与 `.reason`（`cycle` / `buffer-out-of-range` / `next-out-of-range`） |

`cpu_write`/`device_release` 先完成全部校验再改状态，失败调用不留部分变更。

## 链处理语义（设计取舍）

- 链通过 `next` 索引链接，可跨环回绕（scatter/gather）。`walk_chain` 提供整链
  预验证；`CopyEngine` 则逐描述符 acquire→校验→复制→release。
- 环上 buffer 范围（offset+length 是否越出池）**不在** `cpu_write` 时检查，而在
  device 处理/整链验证时检查——模拟真实设备校验描述符的行为；类型、符号、
  有限性等输入非法仍在写入时立即拒绝。
- 坏描述符：该描述符以 `error` 完成（bytes_done=0），链错误完成，引擎 halt；
  链内其余描述符与后续已排队链**不被消费**，直到 `reset_halt()`。坏描述符之前
  已完成的描述符保留 `ok` 与已复制字节（半链部分完成，属明确说明的语义）。
- 环检测是精确的 visited-set 判环（证明有环），不是搜索预算耗尽；链长超过
  capacity 必然重复槽位，因此不存在"预算耗尽"与"证明无解"的混淆。

## 输入上限

- `capacity` ≤ 64 且为 2 的幂；链长 ≤ capacity；`pool_size` ≥ 0（演示用 96 字节，
  未穷举上限规模）；描述符 `next` 为 `None` 或合法环索引。

## 周期采集模式（`periodic_dma.py`）

在基线非周期 DMA（`DmaRing`/`CopyEngine`，接口与语义不变）之外，新增周期
采集模型：硬件在固定缓冲段上循环写入，软件用独立读取序号领取已完成段。

### 接口

- `PeriodicAcquisition(segment_count, segment_size, *, counter_modulus=None)`
  - `segment_count` ≥ 1 个固定段，每段 `segment_size` ≥ 1 字节。
  - `counter_modulus` ≥ 2 时模拟硬件回绕计数器（`hw_counter = hw_seq %
    counter_modulus`）；内部全部按无限序号记账。
- 硬件侧：`hw_write(nbytes)`、`hw_step()`（写满一段）。
- 软件侧：`claim(max_segments=None) -> ClaimResult`（一次可领取多个连续
  完成段）、`step(max_segments=None)`（同一步先硬件完成一段、后软件领取——
  同一步完成的段同一步即可领取）。
- 运行控制：`stop() -> StopReport`、`start()`；自省：`status()`、
  `partial_state()`、`read_seq`、`hw_seq`、`hw_counter`、`lost_total`。
- 数据：`SegmentView(seq, generation, slot, data)`（`generation` 为完成世代
  = 圈数）、`LostRange(first_seq, last_seq, count)`（闭区间）、
  `PartialState(seq, slot, bytes_written, segment_size)`。

### 语义

- 段序号 `s` 为无限序号：写入槽位 `s % segment_count`，完成世代
  `s // segment_count`。
- 采集内容为确定性递增字节：流位置 `p` 的字节为 `p % 256`（字节计数器
  在 256 处自然回绕），段 `s` 的期望内容可仅凭序号独立复现。
- 软件落后恰好一圈仍无损；超过一圈时，被覆盖段以 `LostRange` 明确报告
  并计入 `lost_total`，绝不返回已被新一轮覆盖的旧数据。
- 尚在写的半段不可交付，经 `PartialState` 单独报告。
- `stop()` 保留已完成未领取段（停止后仍可 `claim`）；半段在停止时放弃，
  `start()` 后同一序号从字节 0 重写，序号不回退。

### 错误语义

整数参数拒绝 bool（`TypeError`）、NaN/Infinity（`ValueError`）、非整浮点
（`TypeError`）；非正 `segment_count`/`segment_size`/`max_segments` 与
`counter_modulus < 2` 为 `ValueError`；停止后写硬件、重复 stop/start 抛
`StateError`。

## 浮点误差口径

本模块不产生浮点结果：输出为字节串与整数计数，无误差可言。数值**输入**拒绝
NaN/Infinity（`ValueError`），整数参数拒绝 bool（`TypeError`），非整浮点拒绝
（`TypeError`），整数值浮点（如 `4.0`）接受。

## 已知限制

- 单 CPU 单 device、同步顺序模型；不模拟并发、中断、缓存一致性协议或真实总线。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()` 绕过所有权规则，仅供测试/调试，不属于协议接口。
- 引擎 halt 后已排队链保留在队列中，需显式 `reset_halt()` 才会继续。
- 周期模式为单硬件写者单软件读者的同步步进模型；`status()`/`partial_state()`
  为带外自省，不属于领取协议。

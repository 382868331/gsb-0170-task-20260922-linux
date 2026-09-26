# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

### 隔离 venv 准备（可选）

    python3.14 -m venv .venv   # 或 python -m venv .venv
    . .venv/bin/activate       # 之后直接用 python 即可

venv 只隔离解释器环境；本模块无第三方依赖，是否激活均可运行。

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 一次正确复制 + 一次可见性拒绝（约 0.1 秒）
python demo_iteration.py                   # 周期采集演示：无损/丢失/一圈/回绕/停止重启（< 90 秒，实测 < 0.1 秒）
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

## 浮点误差口径

本模块不产生浮点结果：输出为字节串与整数计数，无误差可言。数值**输入**拒绝
NaN/Infinity（`ValueError`），整数参数拒绝 bool（`TypeError`），非整浮点拒绝
（`TypeError`），整数值浮点（如 `4.0`）接受。

## 已知限制

- 单 CPU 单 device、同步顺序模型；不模拟并发、中断、缓存一致性协议或真实总线。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()` 绕过所有权规则，仅供测试/调试，不属于协议接口。
- 引擎 halt 后已排队链保留在队列中，需显式 `reset_halt()` 才会继续。

## 周期采集模式（PeriodicRing）

在固定缓冲段上循环写入的周期采集模型，与上面的非周期链式 DMA 完全独立；
原有 DmaRing 与 CopyEngine 接口及语义不变。

### 模型

- 硬件把无限序号 sequence 的段写入固定槽位 sequence % capacity，
  每槽一段 segment_size 字节的固定缓冲（ring.buffers）。
- 段内容确定性生成：段 s 的第 j 字节为 (s * segment_size + j) & 0xFF，
  与 periodic_segment_bytes(s, segment_size) 一致；测试用独立重述的同一公式
  作期望值，不调用被测实现。
- 每段完成时记录 1 起始的圈数 lap 与该槽的完成世代 generation（每槽完成
  计数）。软件用独立读取序号（claim 游标）按序领取已完成段。

### 接口

- 生命周期：start() / stop()。stop() 保留全部已完成段可继续领取；
  若停止时硬件正在写某段，返回 PartialSegment(sequence, slot, bytes_written,
  segment_size) 单独报告半段状态（边界上停止返回 None）。start() 续传：
  序号与游标不动，半段作废并从第 0 字节重写。重复 start/stop 抛 OwnershipError。
- 硬件侧：hw_write_bytes(data)（任意字节数，可跨段）、hw_advance(n)、
  hw_step()（整段对齐快进；有半段时拒绝）。未运行时调用抛 OwnershipError。
- 软件侧：claim(max_segments=None) -> ClaimResult(segments, lost)，
  一次可领取多段连续完成段；hw_step_and_claim() 定义同一步的先后顺序：
  硬件先完成，领取随后观察到新段。
- 自省：next_sequence、claim_cursor、pending_completed()、hw_counter()、
  total_completed、total_lost。

### 覆盖与丢失语义

- 软件落后超过一圈时，硬件开始写 sequence 会覆盖一圈前同槽的旧段
  sequence - capacity；若旧段未被领取，记为丢失并合并成连续区间。
- claim() 在 lost 中返回 OverrunReport(lost_start, lost_end)（闭区间，
  精确指出丢了哪几段），被覆盖的旧数据绝不返回；每条丢失只报告一次。
- 正在写的段（含半段）绝不交付；交付的是领取时刻缓冲的不可变副本。

### 计数回绕模拟

- counter_bits=None（默认）：无限计数器，所有丢失精确报告。
- counter_bits=b：可见计数器 hw_counter() 按 2**b 回绕，但段序号仍是
  无限序号。若软件滞后达到整个模数（2**b 段），丢失区间无法从回绕后的
  计数器唯一确定，此时抛 OverrunError 而不是猜测。

### 适用范围与限制

- 同步单线程模型：领取与硬件完成不会真正同时发生，同一步的先后由
  hw_step_and_claim() 显式规定（硬件先、软件后）。
- 不模拟并发、中断、缓存一致性；段大小固定，容量不超过 64 且为 2 的幂。

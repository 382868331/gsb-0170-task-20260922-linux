# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。另提供可选的完成通知
合并器（`CompletionCoalescer`）：按计数阈值或逻辑超时把一段时间内的完成
记录合并为一次通知，**只合并通知，不延迟完成发布与所有权归还**。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

## 运行

固定环境为仓库内虚拟环境（Linux 原生 Python 3.14.7，仅标准库）：

```sh
./.venv/bin/python -m unittest discover -s tests -v   # 必要测试
./.venv/bin/python demo.py                            # 正确复制 + 可见性拒绝 + 通知合并（约 0.1 秒）
```

也可在装有 Python 3.14.7 的环境中用 `python -m unittest discover -s tests -v`
与 `python demo.py` 运行；除标准库外无需安装任何依赖。

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果
- `CompletionEvent(index, generation)` — 合并器观察到的一次完成，身份为 `(index, generation)`
- `NotificationBatch(completions, reason, time)` — 一次通知：一个合并窗口内全部
  `CompletionEvent`（按实际完成顺序，含错误）；`reason ∈ {"count", "timeout", "flush"}`
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

### `CompletionCoalescer(ring, *, count_threshold, timeout)`（通知合并器，可选）

挂在 ring 上的通知合并器。**合并只影响通知，不影响数据可见性**：完成记录仍在
`device_release` 内发布、所有权仍当场归还 CPU，`cpu_acquire` /
`cpu_read_completion` 从不等待通知。不挂合并器时，ring 行为与原来完全一致。

- 构造参数：`count_threshold`（≥1，达到这么多条完成即发一次）、`timeout`（≥1，
  逻辑时间单位；自最早未通知完成项起算，到点即发）。一个 ring 只能挂一个合并器。
- 逻辑时钟：`advance(delta=1) -> int`（`delta`≥1，**时间只由它驱动**，不读墙钟）、
  `now` 属性；完成发生在当前时刻。
- 触发：等待中的完成数达到阈值（`reason="count"`），或 `now == 最早项完成时刻
  + timeout`（边界**相等即触发**，`reason="timeout"`）。同一时刻两者同时满足只发
  **一份**（count 优先标注原因）；`advance` 跨越多拍时，通知时间戳记在真实边界
  （最早项时刻+timeout）而非跳跃终点。
- 显式发：`flush() -> NotificationBatch | None`，立即把当前等待项打包直付调用方；
  **空 flush 返回 `None` 且不入队**；已发出的通知不受影响。
- 读通知（消费一次）：`notifications() -> list[NotificationBatch]`（一次取走全部，
  **重复读返回空列表、不会二次消费**）、`pop_notification() -> NotificationBatch | None`、
  `pending`（等待中的条数）、`unread`（已生成未读批次数）。
- 完成项按**实际完成顺序**（即 `device_release` 提交顺序）排列，**设备错误同样计入**；
  槽位环绕复用后以 `generation` 区分，`(index, generation)` 是完成的唯一身份。

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

## 通知合并示例

```python
from dma_ring import CompletionCoalescer, DmaRing

ring = DmaRing(capacity=2, pool_size=16)
coal = CompletionCoalescer(ring, count_threshold=3, timeout=4)

def complete(index):
    ring.cpu_write(index, offset=0, length=2)
    ring.cpu_release(index)
    ring.device_acquire(index)
    ring.device_release(index, status="ok", bytes_done=2)

complete(0)                 # t=0，最早项，超时边界 t=4
coal.advance(2)            # t=2
complete(1)                # t=2
coal.advance(1)            # t=3：差一拍，阈值(3)/超时(t=4)都未到
coal.notifications()       # []
ring.cpu_acquire(0)
complete(0)                # t=3，槽位 0 环绕复用 -> generation=2，满 3 条
b = coal.pop_notification()
# b.reason == "count", b.time == 3
# b.completions == (CompletionEvent(0,1), CompletionEvent(1,1),
#                   CompletionEvent(0,2))   # 实际完成顺序，gen 区分复用
coal.notifications()       # [] —— 重复读不二次消费
coal.advance(4)            # t=7：已发出的窗口不会再补发超时
coal.flush()               # None —— 空 flush 无事件
```

要点：完成记录在 `device_release` 时已发布、所有权已归还，CPU 可以在通知被读取
之前就 `cpu_acquire` / `cpu_read_completion`；合并器只决定"何时把这段时间的完成
身份打包通知一次"。`demo.py` 第三段演示了阈值、超时边界、环绕 generation、错误链
与正常链相邻、空 flush、读一次即消费。

## 输入上限

- `capacity` ≤ 64 且为 2 的幂；链长 ≤ capacity；`pool_size` ≥ 0（演示用 96 字节，
  未穷举上限规模）；描述符 `next` 为 `None` 或合法环索引。

## 浮点误差口径

本模块不产生浮点结果：输出为字节串与整数计数，无误差可言。数值**输入**拒绝
NaN/Infinity（`ValueError`），整数参数拒绝 bool（`TypeError`），非整浮点拒绝
（`TypeError`），整数值浮点（如 `4.0`）接受。

## 已知限制

- 单 CPU 单 device、同步顺序模型；不模拟并发、中断、取消请求、多线程中断或真实
  总线。通知合并器的时间是 `advance` 驱动的纯逻辑时钟，不对应真实计时。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()` 绕过所有权规则，仅供测试/调试，不属于协议接口。
- 引擎 halt 后已排队链保留在队列中，需显式 `reset_halt()` 才会继续。
- 合并器只合并通知，绝不延迟完成记录发布或所有权归还；不挂合并器时 ring 行为
  与原先逐完成通知的用法完全一致。

# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试（旧引擎 + 新步进入口）
python demo.py                            # 正确复制 + 可见性拒绝 + 步进取消（约 0.1 秒）
```

（本仓库开发环境中 `python` 即 `./.venv/bin/python`，Python 3.14.7。）

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`（`skipped` 由新入口在取消/出错时归还槽位产生）
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果（旧引擎）
- `CopyRequest(request_id, head, slots, generations)` — 新入口的请求句柄，提交时冻结槽位与 generation
- `CopyRequestResult(request_id, head, status, error, error_index, bytes_copied, descriptors, skipped)` —
  新入口请求的终态结果；`status ∈ {"ok", "error", "cancelled"}`
- `StepEvent(kind, request_id, head, index, bytes_copied, skipped)` — 一次 `step` 做的事；
  `kind ∈ {"copy", "error", "cancelled"}`
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

### `SteppedCopyEngine(ring)`（可取消的步进式复制入口）

- `submit(head) -> CopyRequest`：提交前整链结构验证（`next` 合法性、环检测、每个槽位
  必须已 release 给 device、不与在途请求共享槽位），并**冻结**链上槽位及 generation，
  返回句柄。验证全部通过才改状态，被拒绝的提交不留部分变更。
- `step() -> StepEvent | None`：一次最多处理一个描述符，总是作用于最老的在途请求：
  若该请求已登记取消则确认取消（归还全部剩余槽位），否则 acquire→校验→复制→release
  一个描述符。无在途请求时返回 `None`。
- `cancel(request) -> bool`：只**登记**取消意图；返回是否新登记。已取消或已结束的
  请求重复调用是 no-op，不增加任何完成次数。
- `result(request) -> CopyRequestResult | None`：在途返回 `None`，结束后返回冻结的
  终态结果；重复读取幂等。
- 输出：`engine.output`、`engine.in_flight`。

## 新旧两个复制入口的区别

| | `CopyEngine`（旧，保持不变） | `SteppedCopyEngine`（新） |
|---|---|---|
| 驱动方式 | `run()` 一次同步跑完队列 | `step()` 一次最多一个描述符，调用方逐步推进 |
| 取消 | 不支持 | `cancel` 登记意图，下一次 `step` 确认后才结束请求 |
| 描述符错误 | 引擎 **halt**，后续排队链不再消费，需 `reset_halt()` | 只结束当前请求并归还其剩余槽位，**不影响**不相交的其他请求，无全局 halt |
| 槽位占用 | 队列模型，无占用概念 | 两个在途请求不得占用同一槽位（`SlotBusyError`） |
| 结果 | `ChainResult`，status ∈ {ok, error} | `CopyRequestResult`，status ∈ {ok, error, cancelled} |

新入口的取消语义（全部事件按调用顺序串行发生，无线程）：

1. `cancel` 只登记意图。确认之前槽位仍归 device 所有，**CPU 无法重写这些槽位**
   （`cpu_write` 抛 `OwnershipError`），未处理的描述符也不会再产生输出。
2. 下一次 `step` 确认取消：尚未处理的描述符按原有 device_acquire/device_release
   规则归还 CPU，完成状态为 `skipped`、`bytes_done=0`；请求以 `cancelled` 结束。
   已复制的前缀保留在输出中。
3. 先后次序确定胜负：最后一个描述符先复制完成则请求以 `ok` 结束，之后的
   `cancel` 是 no-op，不能改写结果；`cancel` 先登记，则下一 `step` 处理取消，
   不再复制下一段。
4. `bytes_copied` 只计**实际复制**的字节；被 `skipped` 的未复制部分**不算成功**，
   也不出现在输出里。
5. 句柄按 `request_id` 识别请求：槽位环回复用后，旧句柄仍可查询自己的终态，
   但 `cancel(旧句柄)` 绝不会作用到占用同槽位的新一代请求（no-op）。
6. 重复 `cancel`、重复读 `result` 都不增加任何槽位的完成次数——每代每个槽位
   只发布一次完成。

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
| `OwnershipError` | 非持有方写/读/acquire/release（含缺 release、取消确认前 CPU 重写槽位） |
| `VisibilityError` | 未 acquire 就读描述符或完成字段 |
| `GenerationError` | expected_generation 与已提交 generation 不符（ABA） |
| `ChainError` | 整链验证失败；携带 `.index` 与 `.reason`（`cycle` / `buffer-out-of-range` / `next-out-of-range`） |
| `SlotBusyError` | 新入口提交的在途请求占用了同一槽位 |

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
- `SteppedCopyEngine.submit` 只做结构验证（`next`、环、所有权、槽位占用）；
  buffer 范围仍是 device 侧逐描述符校验，在 `step` 时才可能报 `error`。
- `step` 总是作用于最老的在途请求（提交顺序），不提供调度策略。
- 请求在途期间其槽位归 device 所有；绕过引擎直接操作这些槽位（如外部
  `device_acquire`）属于违约用法，不在契约内。

# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 正确复制 + 可见性拒绝 + 分步取消演示（约 0.1 秒）
```

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果
- `CopyRequestResult(request_id, head, status, error, error_index, bytes_copied,
  descriptors, skipped)` — 分步请求的终态结果；`status ∈ {"ok", "error", "cancelled"}`，
  `bytes_copied` 只含实际复制的字节
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

### `SteppedCopyEngine(ring)`（可取消的分步复制入口）

- `submit(head) -> CopyRequest`：整链结构验证（`next` 合法、无环）、要求每个槽位
  已 release 给 device 且未被其他在途请求占用，然后**冻结**每个槽位的 generation
  并返回请求句柄。槽位冲突抛 `SlotBusyError`（携带 `.index` 与 `.request_id`）；
  所有校验先于任何状态变更，被拒绝的 submit 不留部分占用。
- `step() -> bool`：一次最多处理一个事件——确认队首请求已登记的取消，或复制其
  下一个描述符。无请求在途时返回 `False`。`drain()` 循环 step 直到清空。
- `CopyRequest.cancel() -> bool`：只**登记**取消意图；幂等，重复调用返回 `False`。
  设备在下一次 `step()` 确认后才结束请求。
- `CopyRequest.result() -> CopyRequestResult | None`：终态结果（在途时为 `None`）；
  重复读取返回同一对象，不增加完成次数。`CopyRequest.done` / `.cancel_requested`
  为只读状态。
- 输出：`engine.output`、`engine.results`（每个请求恰好一条 `CopyRequestResult`）、
  `engine.active`。
- `CopyRequestResult(request_id, head, status, error, error_index, bytes_copied,
  descriptors, skipped)`：`status ∈ {"ok", "error", "cancelled"}`；
  `descriptors` 为已完成前缀（含出错描述符），`skipped` 为未处理即归还的槽位。

## 新旧入口的区别

| | `CopyEngine`（旧） | `SteppedCopyEngine`（新） |
|---|---|---|
| 驱动方式 | `run()` 一次处理整条链 | `step()` 一次最多一个描述符 |
| 取消 | 不支持 | `cancel()` 登记意图，下一 `step()` 确认 |
| 描述符错误 | 引擎 halt，后续排队链不消费，需 `reset_halt()` | 只结束当前请求并归还其剩余槽位，不影响不相交请求；无 halt 标志 |
| 结果 | `ChainResult`（`ok`/`error`） | `CopyRequestResult`（`ok`/`error`/`cancelled`） |
| 槽位占用 | 提交即入队，不跟踪占用 | 提交时冻结槽位与 generation，两个在途请求不得占用同一槽位 |

两个引擎共享同一个 `DmaRing` 的所有权协议，但相互独立；旧引擎的
`run`/`halt`/`reset_halt` 语义没有任何改变。

取消与归还语义（新入口）：

1. `cancel()` 只登记意图。取消未确认前，槽位仍归 device 所有，CPU 对链内任何
   槽位的 `cpu_write`/`cpu_acquire` 仍被 `OwnershipError` 拒绝——**不能提前归还
   缓冲区**。
2. 下一 `step()` 确认取消：请求以 `cancelled` 结束，尚未处理的描述符按原有
   release/acquire 规则归还（device acquire 冻结的 generation，再以
   `status="skipped", bytes_done=0` release），CPU `cpu_acquire` 后可见
   `skipped` 完成记录，此后才能重写槽位。
3. 已复制的前缀保留在 `engine.output` 中；未处理的部分不再产生任何输出。
   `bytes_copied` 只统计**实际复制**的字节，未复制字节绝不算作成功。
4. 事件严格串行：最后一个描述符先完成则 `ok` 胜出，之后的 `cancel()` 是 no-op；
   取消先登记则下一 `step()` 处理取消，不再复制下一段。
5. 每个请求恰好发布一次终态结果；重复 `cancel()` 与重复 `result()` 读取都不
   增加完成次数。
6. 旧句柄在槽位环绕复用后仍可查询自己的终态（`result()` 不变），但
   `cancel()` 只作用于自己那一代请求，绝不能取消复用槽位的新一代请求。

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
| `OwnershipError` | 非持有方写/读/acquire/release（含缺 release、取消未确认前 CPU 重写槽位） |
| `VisibilityError` | 未 acquire 就读描述符或完成字段 |
| `GenerationError` | expected_generation 与已提交 generation 不符（ABA） |
| `ChainError` | 整链验证失败；携带 `.index` 与 `.reason`（`cycle` / `buffer-out-of-range` / `next-out-of-range`） |
| `SlotBusyError` | 提交的在途请求占用冲突；携带 `.index` 与 `.request_id` |

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
  新入口的所有事件按调用顺序串行发生，不需要线程。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()` 绕过所有权规则，仅供测试/调试，不属于协议接口。
- 旧引擎 halt 后已排队链保留在队列中，需显式 `reset_halt()` 才会继续。
- 新入口按提交顺序处理请求（队首请求未完成前不处理后续请求）；取消确认是
  单个 step 事件，一次性归还该请求全部剩余槽位。

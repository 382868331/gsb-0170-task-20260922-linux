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
python demo.py                            # 一次正确复制 + 一次可见性拒绝（约 0.1 秒）
```

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors, skipped, token)` — 单链处理结果；
  `status ∈ {"ok", "error", "cancelled"}`，`skipped` 为取消时未执行且设备已确认不再访问的描述符
- `ChainNotification(token, head, generation, status)` — 设备→CPU 的链终态通知；
  以提交时分配的唯一 token 与头槽位 generation 标记，旧通知无法被误认为属于复用槽位后的新链
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

- `submit_chain(head) -> int`（返回唯一链令牌 token）、`run()`、`step()`、`reset_halt()`
- 撤销/停止：`request_cancel(token)`（撤销未开始的链，或请求停止正在执行的链）、
  `reclaim(token) -> ChainResult`（CPU 侧消费终态通知并回收链资源）
- 输出：`engine.output`（复制字节结果）、`engine.results`（`ChainResult` 列表）、
  `engine.notifications`（设备→CPU 通知队列）、`engine.queued`、`engine.halted`

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

撤销/停止相关：`request_cancel`/`reclaim` 对未知 token 或已终态的链抛 `DmaRingError`；
`reclaim` 在设备确认终态之前抛 `OwnershipError`（已交给设备的描述符不能立即回收）。

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

## 撤销、停止与通知语义

- `submit_chain` 返回单调递增、永不复用的链 token；`request_cancel(token)`
  只记录请求：对未开始的链是撤销，对正在执行的链是停止请求。
- 请求在设备工作边界（下一次 `step()`/`run()`）生效：设备逐个 acquire 并以
  `status="skipped"`、`bytes_done=0` release 尚未执行的描述符——这就是"设备确认
  不再访问"；此前 `reclaim` 一律抛 `OwnershipError`，已交给设备的部分不会被立即回收。
- 取消链的最终 `ChainResult.status="cancelled"`，同时汇总已完成前缀
  （`descriptors`、`bytes_copied`）与未执行部分（`skipped`）。
- 每个终态（ok/error/cancelled）恰好发出一条 `ChainNotification`，携带 token 与
  提交时头槽位 generation。`reclaim` 消费并丢弃该 token 的全部通知（含陈旧/重复
  通知），且幂等：重复通知或重复 `reclaim` 不会重复 release 描述符、不会多计
  `completion_count`。token 永不复用 + generation 校验保证旧通知不影响复用槽位后的新链。
- `step()` 每次推进一个工作单元（处理一个描述符，或确认一次取消），用于在测试中
  交错"部分完成 → 停止请求 → 确认 → 槽位复用"。取消确认即使在引擎 halt 时也会处理
  （设备随时能确认不再访问），普通链处理仍需 `reset_halt()`。

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

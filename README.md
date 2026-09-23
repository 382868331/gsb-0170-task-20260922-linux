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
python demo.py                            # 链复制、可见性拒绝、请求乱序完成、边界与错误（约 0.1 秒）
```

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果
- `DescriptorCompletion(index, status, error, bytes_done)` — 请求结果中单个描述符的结局
  （`status ∈ {"ok","error","skipped"}`），按请求的提交顺序排列
- `RequestResult(request_id, status, error, error_index, bytes_copied, descriptors)` —
  多描述符请求的最终结果，仅在所有描述符都落定后发布
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

### `RequestEngine(ring)`（多描述符请求，本次新增）

把多个独立描述符合成一个请求；描述符可乱序完成，全部落定后才发布一个结果。

- `submit_request(descriptors, *, request_id=None) -> tuple[DescriptorSnapshot, ...]`：
  `descriptors` 为互不重复的环索引序列（至少一个），每个槽位须已 `cpu_release`
  且未被获取，buffer 范围须在池内。全部校验通过后才一次性原子获取；失败不获取
  任何槽位，也不消耗自动 id。省略 `request_id` 时自动分配（成功后才推进计数），
  显式 id 须为非负整数且未被占用；返回各槽位提交时的快照（含 generation）。
- `complete_descriptor(request_id, index, *, generation, status="ok",
  bytes_done=0, error=None) -> RequestResult | None`：以任意顺序上报完成；
  `generation` 必须等于获取时的世代。仅当本次调用让请求全部落定时返回
  `RequestResult`，否则返回 `None`。
- 输出与自省：`engine.results`（已发布的 `RequestResult`，按请求完成先后）、
  `engine.output`（`ok` 描述符的字节，按实际完成先后）、`engine.pending`
  （未结请求数）、`acquired_snapshot(request_id, index)`（取获取时快照）、
  `request_snapshot(request_id)`（带外调试视图，含 `states`/`remaining`/
  `completion_order`，不属于协议）。

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
| `RequestError` | 请求级规则被违反；携带 `.request_id`、`.index` 与 `.reason`（见下） |

`RequestError.reason` 取值：

- `empty-request`：描述符列表为空
- `duplicate-descriptor`：同一提交中槽位重复出现
- `descriptor-unavailable`：槽位未 release 给 device 或已被获取
- `descriptor-busy`：槽位已属于另一个在途请求
- `buffer-out-of-range`：描述符 buffer 范围越出共享池（在提交时即拒绝）
- `unknown-request`：完成上报指向未知或已结束的请求
- `not-in-request`：描述符不属于该请求
- `stale-generation`：完成世代 ≠ 获取世代（环回复用后的旧完成/ABA）
- `duplicate-completion`：同一描述符在同一请求中被第二次完成
- `duplicate-request-id`：显式 `request_id` 已被在途请求占用

`cpu_write`/`device_release`/`submit_request`/`complete_descriptor` 先完成全部
校验再改状态，失败调用不留部分变更。

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

## 请求处理语义（本次新增）

- 请求是在现有环协议之上的一层：CPU 仍按原方式 `cpu_write`/`cpu_release`
  各描述符；`RequestEngine.submit_request` 把若干**互不重复**的槽位原子获取
  成一个请求，并记录每个槽位的 generation。
- 描述符之间无 `next` 链接、无顺序约束，可以乱序 acquire/完成。请求结果只在
  **全部**描述符完成后发布一次；此前 `engine.results` 中没有该请求。
- 世代即完成身份：完成上报必须携带获取时的 generation。世代不符拒绝为
  `stale-generation`（环回复用后迟到的旧完成）；同世代第二次上报拒绝为
  `duplicate-completion`。被拒调用不改任何状态，槽位仍由该请求持有。
- 某个描述符以 `error` 完成即中止请求：出错描述符保留 `error`，此前已 `ok`
  的描述符保留其字节，仍在 pending 的描述符在**同一步**内以 `skipped`
  释放回 CPU——错误请求不泄漏任何已占槽位。结果整体状态为 `error`，携带
  `error_index`。
- 多个请求可同时在途，槽位不跨请求共享；一个请求中止不影响其他在途请求。
- 请求提交时即校验 buffer 范围（与链处理的"设备侧校验"不同：请求在获取前
  整批校验，保证不会获取一半才发现越界）。

## 输入上限

- `capacity` ≤ 64 且为 2 的幂；链长 ≤ capacity；`pool_size` ≥ 0（演示用 96 字节，
  未穷举上限规模）；描述符 `next` 为 `None` 或合法环索引。
- 单个请求的描述符数 ∈ [1, capacity] 且互不重复；请求间槽位不重叠；
  `request_id` 为非负整数（bool 拒绝）或省略自动分配。

## 浮点误差口径

本模块不产生浮点结果：输出为字节串与整数计数，无误差可言。数值**输入**拒绝
NaN/Infinity（`ValueError`），整数参数拒绝 bool（`TypeError`），非整浮点拒绝
（`TypeError`），整数值浮点（如 `4.0`）接受。

## 已知限制

- 单 CPU 单 device、同步顺序模型；不模拟并发、中断、缓存一致性协议或真实总线。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()`、`request_snapshot()`、`acquired_snapshot()` 绕过所有权规则，
  仅供测试/调试，不属于协议接口。
- 引擎 halt 后已排队链保留在队列中，需显式 `reset_halt()` 才会继续。
- 请求层只在提交时做静态校验（所有权、重复、buffer 范围）；不实现多队列共享
  缓冲、取消确认与设备复位协议（不在本次范围内）。请求一旦提交，描述符只能
  以 `ok`/`error` 完成，不能由调用方撤回；`error` 后的 `skipped` 释放是
  引擎内部行为而非可请求的取消。

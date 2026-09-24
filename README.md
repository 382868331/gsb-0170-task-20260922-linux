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
python demo.py                            # 链复制 + 请求乱序/环回/世代拒绝（远小于 90 秒）
```

## 接口（`dma_ring.py`）

### 数据结构

- `DescriptorSnapshot(index, generation, offset, length, next)` — 已提交描述符字段的不可变视图
- `CompletionSnapshot(index, generation, status, error, bytes_done)` — 已提交完成记录；
  `status ∈ {"ok", "error", "skipped"}`（`skipped` 仅由环在请求错误清理时发布）
- `ChainResult(head, status, error, error_index, bytes_copied, descriptors)` — 单链处理结果
- `RequestCompletion(index, generation, status, error, bytes_done)` — 请求内单个成员的结果
- `RequestResult(request_id, status, error, error_index, bytes_copied, completions)` —
  多描述符请求的聚合结果；`completions` 一律按**注册顺序**排列，与实际完成顺序无关
- `Owner.{CPU, DEVICE}`

### `DmaRing(capacity, pool_size)`

- `capacity`：2 的幂，∈ [1, 64]；`pool_size` ≥ 0。`ring.pool` 为共享字节池。
- CPU 侧：`cpu_write(index, *, offset, length, next=None)`（暂存，不落可见状态）、
  `cpu_release(index) -> int`（flush 并交给 device，返回新 generation）、
  `cpu_acquire(index) -> CompletionSnapshot`、`cpu_read_completion(index)`
- device 侧：`device_acquire(index, *, expected_generation=None) -> DescriptorSnapshot`、
  `device_read(index)`、
  `device_release(index, *, status, bytes_done=0, error=None, request_id=None)`
- 链与自省：`walk_chain(head) -> list[int]`（整链预验证）、
  `completion_count(index)`（每项完成次数）、`inspect(index)`（带外调试视图，不属于协议）
- 请求（本次新增）：
  - `register_request(indices) -> (request_id, generations)`：在已交给 device 的槽位上
    注册多描述符请求；`indices` 为非空列表/元组、索引不重复；登记各槽位**当前世代**。
    全部校验通过前不改任何状态（失败不留半更新）。
  - `request_is_done(request_id) -> bool`：聚合结果是否已发布。
  - `request_acquire(request_id) -> RequestResult`：读取已完成请求的聚合结果（幂等，
    可重复读）；请求仍有成员未完成时抛 `RequestError(reason="in-flight")`。
  - 成员完成仍走 `device_release(..., request_id=rid)`：每个成员完成后槽位**立即**
    归还 CPU；只有全部成员完成，请求才发布 `RequestResult`。

### `CopyEngine(ring)`（合成设备，同步复制参考）

- 链：`submit_chain(head)`、`run()`、`reset_halt()`
- 请求：`submit_request(indices, *, order=None) -> request_id`（注册并入队；
  `order` 指定设备完成顺序，必须是 `indices` 的排列，用于演示乱序完成；缺省按注册顺序）
- 输出：`engine.output`（复制字节结果）、`engine.results`（`ChainResult` 列表）、
  `engine.request_results`（`RequestResult` 列表）、`engine.queued`、`engine.halted`

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
| `TypeError` | 整数参数为 bool、非整浮点、非数值类型；`indices`/`order` 非列表元组 |
| `ValueError` | NaN/Infinity、负数 offset/length、非法 capacity/status/bytes_done、空请求、成员自报 `skipped` |
| `IndexError` | 环索引越界 |
| `OwnershipError` | 非持有方写/读/acquire/release（含缺 release） |
| `VisibilityError` | 未 acquire 就读描述符或完成字段 |
| `GenerationError` | expected_generation 与已提交 generation 不符（ABA）；或向请求提交了槽位回收后的**旧完成** |
| `ChainError` | 整链验证失败；携带 `.index` 与 `.reason`（`cycle` / `buffer-out-of-range` / `next-out-of-range`） |
| `DuplicateCompletionError` | 同一 `(request, 槽位, 世代)` 完成第二次；携带 `.request_id/.index/.generation` |
| `RequestError` | 请求注册/生命周期违规；携带 `.request_id`（注册失败时为 `None`）、`.index`、`.reason`（`duplicate-member` / `not-device-owned` / `device-held` / `already-in-request` / `bad-order` / `not-a-member` / `unknown-request` / `in-flight`） |

`cpu_write`/`device_release`/`register_request`/`submit_request` 先完成全部校验再改状态，
失败调用不留部分变更（请求级的重复/旧完成检查在所有权检查之前，槽位已归还 CPU 也能正确拒绝）。

## 请求语义（本次新增，设计取舍）

- 请求是**显式成员列表**，不沿 `next` 链遍历：`register_request([i, j, k])` 把若干已交给
  device 的槽位登记为一个请求，并快照各槽位世代。成员可被设备**乱序完成**；每个成员完成
  后槽位立刻归还 CPU，但 `RequestResult` 只在全部成员完成后发布，且按注册顺序排列。
- **世代守卫**：注册时记录每个成员的世代。完成时 `(request_id, index)` 必须匹配，且槽位当前
  世代必须与登记世代一致——若槽位已被回收复用（世代前进），该完成是旧完成，抛
  `GenerationError` 且不产生任何记录。
- **重复完成**：同一 `(request, index, generation)` 重复完成抛 `DuplicateCompletionError`；
  请求已完结后重放同世代成员同样按重复完成拒绝（槽位已回收则按旧完成拒绝）。
- **错误请求释放已占资源**：任一成员以 `error` 完成，请求立即以 `error` 终结，`error_index`
  指向首个错误成员；此前已 `ok` 的成员保留字节，仍未完成的成员由环强制以 `skipped`
  （bytes_done=0）发布完成并归还 CPU，槽位归属全部清空，环可立刻回绕复用，无需手工清理。
  `CopyEngine` 随之 halt（与坏链一致），后续排队项在 `reset_halt()` 前不被消费。
- 请求与链互斥：请求成员不能作为链头提交，链遍历 `next` 进入请求成员时整条链错误停止，
  且不 acquire、不完成该成员。
- `RequestResult.bytes_copied` 只计 `ok` 成员；设备输出字节的先后顺序取决于实际完成顺序，
  与聚合结果按注册顺序排列是两件事。

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

- `capacity` ≤ 64 且为 2 的幂；链长 ≤ capacity；请求成员数 ≤ capacity（成员不可重复）；
  `pool_size` ≥ 0（演示用 96 字节，未穷举上限规模）；描述符 `next` 为 `None` 或合法环索引。
- `request_id` 为进程内自增整数，仅在其所属 `DmaRing` 实例内有意义。

## 浮点误差口径

本模块不产生浮点结果：输出为字节串与整数计数，无误差可言。数值**输入**拒绝
NaN/Infinity（`ValueError`），整数参数拒绝 bool（`TypeError`），非整浮点拒绝
（`TypeError`），整数值浮点（如 `4.0`）接受。

## 已知限制

- 单 CPU 单 device、同步顺序模型；不模拟并发、中断、缓存一致性协议或真实总线。
- 只实现本文档所述契约子集，不宣称兼容任何行业标准 DMA 控制器。
- `inspect()` 绕过所有权规则，仅供测试/调试，不属于协议接口。
- 引擎 halt 后已排队链/请求保留在队列中，需显式 `reset_halt()` 才会继续。
- 请求是显式成员集合：成员在注册时必须全部已交给 device；不支持多队列共享缓冲、
  取消确认（请求一旦注册只能等成员完成/错误终结）或设备复位协议，这些不在本次范围内。
- 旧完成检测依赖注册时快照的世代；若调用方在请求未完结时违规回收槽位（先
  `cpu_acquire` 再 `cpu_release`），该槽位对请求的完成会被判为旧完成而拒绝，
  请求仍可凭其余成员与既有记录正常终结——模块不会阻止回收本身，但会隔离其后果。

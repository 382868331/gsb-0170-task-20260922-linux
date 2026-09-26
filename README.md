# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

环上挂两类合成设备：`CopyEngine`（把池字节收集到输出）与 `BlitEngine`
（在同一字节池内按二维带步长描述符搬运矩形图块，步长可为负）。两者
复用同一套描述符环所有权交接协议，互不修改对方语义。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线
- 隔离环境（可选但推荐）：`python3 -m venv .venv && . .venv/bin/activate`；
  无第三方依赖，`requirements.txt` 仅为说明性注释，无需 pip install

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 一次正确复制 + 一次可见性拒绝（约 0.1 秒）
python demo_iteration.py                   # BlitEngine 翻转/重叠/越界拒绝（约 0.1 秒）
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

### `BlitEngine(ring)`（合成设备，二维带步长图块搬运）

- 描述符用 `ring.cpu_write_blit(index, *, src, dst, rows, row_bytes,
  src_stride, dst_stride, next=None)` 暂存，`cpu_release` 提交；所有权交接、
  generation/ABA、可见性规则与 CopyEngine 完全共用。
- 地址生成：第 `r` 行起点为 `start + r * stride`，每行 `row_bytes` 字节；
  `src_stride`/`dst_stride` 允许负数。`rows == 0` 或 `row_bytes == 0` 为空矩形，
  是成功的零传输。
- `submit_chain(head)` 即“接收请求”：整链 acquire 并逐描述符检查**所有**生成
  地址（源与目标每一行）是否在池内；任一行越界则不写入任何字节、不入队，
  坏描述符以 `error` 完成、已 acquire 的前驱以 `skipped` 完成，并抛
  `ChainError`（带 `.index`/`.reason`，reason ∈ `cycle` / `next-out-of-range` /
  `buffer-out-of-range` / `wrong-kind`）。
- 提交成功时对源行做快照（只缓存有效源行，不复制整个池）：整次搬运按提交时
  的源字节解释，源/目标重叠结果确定；提交后池被改动不影响本次搬运。
- `run()` 按逻辑行号从小到大写、行内按列号写；目标行自身重叠时后写覆盖先写。
  未涉及的池字节（含步长填充）保持原值。`bytes_done` 只计行有效载荷
  （`rows * row_bytes`），不含行填充。
- 输出：`engine.results`（`BlitChainResult` 列表）、`engine.queued`。
  地址在提交时已验证，`run()` 不会半途失败，因此无 halt 语义。

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
- `BlitEngine` 只接受 `cpu_write_blit` 暂存的描述符（其余按 `wrong-kind` 拒绝）；
  `CopyEngine` 不解释 blit 字段，把 blit 描述符当普通 offset/length 复制处理。
  两种引擎共用同一字节池，混用时需自行保证描述符种类与引擎匹配。
- blit 行数/行字节数无单独上限，数据搬运代价为 O(rows × row_bytes)；
  地址越界检查为 O(1)（等差数列端点检查）。

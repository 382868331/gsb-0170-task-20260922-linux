# dma_ring — 合成 DMA 环形描述符所有权交接模型

面向离线设备驱动内核工程的可复用模块：纯软件模拟 DMA 描述符环的
CPU↔device 所有权交接协议。无真实硬件、无线程、无 I/O；描述符字段的
可见性用显式 release（flush）/ acquire 事件模拟。

## 环境

- Python 3.14.7（仅标准库，无需安装任何其他依赖）
- Linux 原生运行；核心、测试、演示全部离线

## 隔离 venv 准备（可选）

模块只用标准库，系统 Python 3.14.7 即可直接运行；如需隔离环境：

```sh
python3 -m venv .venv        # 创建隔离环境
. .venv/bin/activate         # 之后所有 python 命令都在 venv 中执行
```

无需 `pip install` 任何依赖。

## 运行

```sh
python -m unittest discover -s tests -v   # 必要测试
python demo.py                            # 一次正确复制 + 一次可见性拒绝（约 0.1 秒）
python demo_iteration.py                   # BlitEngine 二维搬运演示（< 1 秒，上限 90 秒）
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

与 `CopyEngine` 共用同一 `DmaRing` 与全部所有权/可见性协议，但消费
`cpu_write_blit` 暂存的二维描述符，直接在同一字节池 `ring.pool` 内搬运。

- CPU 侧新增：`cpu_write_blit(index, *, src, dst, rows, bpr, src_stride,
  dst_stride, next=None)`（暂存）、`cpu_release(index)`（同一提交/交接）。
  `src`/`dst` 为第 0 行的池内字节偏移，`rows` 为行数，`bpr` 为每行字节数，
  两个步长为相邻行的有符号地址增量（允许负数，如垂直翻转）。
- device 侧新增：`device_read_blit(index) -> BlitSnapshot`（需先
  `device_acquire`）。`BlitSnapshot(index, generation, src, dst, rows, bpr,
  src_stride, dst_stride, next)` 为已提交二维字段的不可变视图。
- 引擎接口与 `CopyEngine` 对称：`submit_chain(head)`、`run()`、
  `reset_halt()`、`results`、`queued`、`halted`；不写 `output`，搬运结果
  体现在 `ring.pool` 中。

#### 二维搬运语义

- 第 `r` 行源地址为 `src + r*src_stride`，目标为 `dst + r*dst_stride`，
  每行占 `[addr, addr+bpr)`。
- **先校验后写**：引擎接受描述符时先检查所有行的源/目标地址，任意一行
  越出池界，该描述符以 `error` 完成且一个字节都不写（链错误完成、引擎
  halt，语义同 `CopyEngine`）。
- **快照语义**：整次搬运按接受请求时的源字节解释；引擎只缓存有效的源行
  字节（不复制整个池），因此源/目标重叠（含自重叠）结果确定。
- **写序**：目标按逻辑行号从小到大、行内按列号写；目标行自身重叠时后写
  覆盖先写。未被任何写触及的池字节（含行填充/间隙）保持原值。
- **空矩形**：`rows == 0` 或 `bpr == 0` 是成功的零传输；完成记录
  `bytes_done == rows*bpr`，只计载荷字节，不含行填充。
- 向 `BlitEngine` 提交 copy 描述符（或反向混用链）会以 `error` 完成该
  描述符，不静默纠正。

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
- `walk_chain` 的 buffer 范围检查针对 copy 描述符的 `offset+length`；二维
  描述符的行地址由 `BlitEngine` 在接受请求时逐行校验。

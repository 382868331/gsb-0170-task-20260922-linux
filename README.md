# dmaring — DMA 环形描述符所有权交接（合成模型）

为离线设备驱动内核工程提供的可复用 Python 库：以纯软件方式模拟 DMA
描述符环（descriptor ring）的 CPU/设备所有权交接协议。**不涉及任何真实
硬件**——可见性用显式的 flush/acquire 事件建模。

- Python 版本：**3.14.7**（仅用标准库，无第三方依赖，安装 Python 即可运行）
- 平台：Linux 原生；核心、测试、演示全部离线（无网络、账号、密钥、
  数据库、中间件、云 API、Docker）
- 示例数据由仓库内确定性代码生成

## 快速开始

```bash
python -m unittest discover -s tests -v   # 运行全部测试
python demo.py                            # 固定输入演示（约 0.1 秒内完成）
```

## 模型与接口

### 所有权协议

```
CPU: cpu_write(slot, offset=, length=, next=)   # 仅 CPU 拥有时可写
CPU: cpu_release(slot) -> generation            # 代际+1，发布快照(flush)，交给设备
DEV: device_acquire(slot) -> DescriptorSnapshot # 仅设备拥有时可取；重复代际拒绝(ABA)
DEV: device_release(slot, status=, bytes_copied=) -> Completion  # 写完成字段，归还 CPU
CPU: cpu_acquire(slot) -> Completion            # 取回本周期的完成记录
CPU: completion(slot) -> Completion             # 未先 cpu_acquire 读取 → VisibilityError
```

描述符字段：`generation`、`offset`/`length`（buffer 范围）、
`next`（`(ring_id, slot)` 或 `None`=链尾）、`owner`（`"cpu"`/`"device"`），
以及设备回写的完成字段 `status`（`"ok"`/`"error"`）、`bytes_copied`。

### 公开对象（`dmaring` 包）

| 名称 | 说明 |
|---|---|
| `DmaRing(capacity, *, ring_id=0)` | 描述符环；容量须为 2 的幂且 ≤ 64 |
| `CoherentMemory(size)` | CPU/设备双视图内存；`flush()` CPU→设备，`invalidate()` 设备→CPU |
| `SyntheticDevice(memory)` | 合成 DMA 引擎；`process_chain(rings, head)` 走链复制 |
| `DescriptorSnapshot` / `Completion` / `ChainResult` | 不可变结果视图（NamedTuple） |
| `reference_gather_copy(source, chains)` | 独立参考实现（不调用核心），按合法交接顺序做同步复制 |
| 常量 | `OWNER_CPU` `OWNER_DEVICE` `STATUS_OK` `STATUS_ERROR` `END` `MAX_CAPACITY`(64) `MAX_MEMORY_SIZE`(1 MiB) |

### 错误语义（均可定位，绝不静默纠正）

全部继承 `DmaRingError`：

- `ValidationError`（兼 `ValueError`）：非法输入。整数参数拒绝 `bool`，
  拒绝一切非 `int`（含 `float`/`NaN`/`Infinity`），消息含参数名。
- `CapacityError`：容量非 2 的幂或越界。
- `OwnershipError`：在对方拥有描述符时执行本侧操作（含“缺 release”）。
- `VisibilityError`：未 acquire 就读完成字段，或无完成记录可读。
- `StaleGenerationError`：设备重复 acquire 已消费的代际（环回 ABA 防护）。
- `DeviceHaltedError`：链错误停机后继续提交新链。

### Scatter/Gather 整链验证

`SyntheticDevice.process_chain(rings, head)` 从 `head=(ring_id, slot)`
沿 `next` 跨环走链，逐个 acquire → 校验 buffer 范围 → 从设备视图复制
字节到输出流 → release。任一描述符出错（acquire 失败、范围越界、链环、
悬空环指针）时：该链以 `status="error"` 完成，已消费项保留完成记录，
越界描述符本身得到 error 完成，**链内剩余描述符与后续链均不消费**
（设备停机，`reset()` 后才接受新链）。结果含复制字节数与每项完成
次数（`device.completion_counts` / `completion_count(ring_id, slot)`）。

## 输入上限

- 环容量：2 的幂，1–64（`MAX_CAPACITY`）
- 内存：1–1,048,576 字节（`MAX_MEMORY_SIZE`）
- 链长：由环回检测自然约束（同一 `(ring_id, slot)` 不会走第二遍）
- 所有数值输入为整数；浮点（含 NaN/Infinity）与 `bool` 一律拒绝

## 设计取舍

- **显式可见性事件**：`CoherentMemory` 与描述符表都分 CPU/设备两份视图，
  只有 `flush`/`invalidate`/`cpu_release`/`device_release` 才发布快照，
  以此在无硬件条件下复现一致性语义。
- **代际计数防 ABA**：环回复用同一槽位且字段值相同时，只有单调递增的
  `generation` 能区分新旧描述符；设备按槽位记录已消费代际，拒绝陈旧
  acquire。
- **写时不做内存边界检查**：`cpu_write` 只校验类型与非负；越界 buffer
  由设备在走链时发现并产生 error 完成，模拟硬件在 doorbell 之后才发现
  坏描述符的真实时序。
- **批次原子性**：`cpu_write_many` 先整体校验（含所有权）再整体应用，
  失败不留部分变更。
- **参考实现独立**：`dmaring/reference.py` 不导入核心，测试另用手算
  期望值钉住参考本身，核心与参考的一致性仅作往返补充验证。

## 已知限制

- 单线程、确定性模型：不模拟并发 doorbell、乱序完成或中断。
- 仅实现本题契约的所有权/可见性子集，不宣称兼容任何真实硬件
  （如 Intel/ARM 描述符格式）或完整行业标准。
- 设备出错即停机是刻意的保守语义；真实驱动可能跳过坏描述符继续。
- 本库不产生浮点结果，故无浮点误差口径问题；字节复制结果为精确相等。
- `ChainError` 目前保留在错误分类中，链错误经 `ChainResult.error`
  报告而非抛出。

## 目录

```
dmaring/__init__.py    公共导出
dmaring/core.py        环、内存、合成设备
dmaring/errors.py      错误分类
dmaring/reference.py   独立参考实现
tests/                 unittest 测试（52 项）
demo.py                固定输入演示
```

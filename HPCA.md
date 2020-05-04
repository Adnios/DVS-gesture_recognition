## NVDIMM-C: A Byte-Addressable Non-Volatile Memory Module for Compatibility with Standard DDR Memory Interfaces

>Samsung Electronics

### Knowledge

- 非易失性双列直插式内存模块（英语：non-volatile dual in-line memory module，缩写NVDIMM）是一种用于计算机的随机存取存储器。非易失性存储器是即使断电也能保留其内容的内存，这包括意外断电、系统崩溃或正常关机。双列直插式表示该内存使用DIMM封装。NVDIMM在某些情况下可以改善应用程序的性能、数据安全性和系统崩溃恢复时间。这增强了固态硬盘（SSD）的耐用性和可靠性。DIMM全称Dual-Inline-Memory-Modules，中文名叫双列直插式存储模块，是指奔腾CPU推出后出现的新型内存条，它提供了64位的数据通道。
- 2 representative NVDIMM: a proprietary Inter DDR-T and JEDEC NVDIMM-P
    - 采用新平台成本高昂,衡量其迁移到新版本的效率平台要复杂得多
    - 查找可以由所有现有系统支持的新存储设备
- NVM
    - performance, density with dynamic random-access memory(DRAM)
    - byte-addressable
- NVDIMM-C: an NVDIMM architecture for the Compatibility with the standard DDR4 memory interfaces without any modification to the hardware and software environments
- proof-of-concept (PoC) 概念验证

### Architecture

![2 design options](https://raw.githubusercontent.com/Adnios/Picture/master/img/20200504225314.png "opt title")



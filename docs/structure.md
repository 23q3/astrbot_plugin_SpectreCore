# 🏗️ 项目结构

SpectreCore插件的项目结构如下：

```
astrbot_plugin_SpectreCore/
├── main.py                # 插件主入口，包含消息处理和命令处理逻辑
├── _conf_schema.json      # 配置项定义，包含所有可配置选项
├── metadata.yaml          # 插件元数据，用于插件市场展示
├── README.md              # 插件文档
├── CHANGELOG.md           # 更新日志
├── LICENSE                # 许可证
├── docs/                  # 详细文档
│   ├── README.md          # 文档中心索引
│   ├── commands.md        # 指令详细说明
│   ├── tips.md            # 使用技巧
│   └── structure.md       # 项目结构说明
└── utils/                 # 工具类
    ├── __init__.py        # 工具类模块初始化文件
    ├── history_storage.py # 历史消息存储
    ├── llm_utils.py       # 大语言模型工具
    ├── text_filter.py     # 文本过滤工具
    ├── persona_utils.py   # 人格处理工具
    ├── reply_decision.py  # 回复决策工具
    ├── message_utils.py   # 消息处理工具
    ├── quote_utils.py     # 模型自主引用工具
    └── image_caption.py   # 图片描述工具
```

## 核心文件说明

### 主入口文件

- **main.py**: 插件的主要实现，包含以下功能：
  - 消息处理逻辑（群聊和私聊）
  - 命令处理（help、reset、callllm等）
  - 事件过滤器（消息发送后、LLM响应等）
  - 插件初始化和配置加载

### 配置文件

- **_conf_schema.json**: 定义插件配置项的结构和默认值，包括：
  - 历史消息数量限制
  - 启用的群聊列表
  - 私聊回复开关
  - 思考过程过滤
  - 人格设置
  - 读空气功能
  - 模型自主引用
  - 函数工具开关
  - 模型频率设置
  - 图片处理配置

### 工具类

- **utils/**: 包含插件的各种工具类
  - **history_storage.py**: 负责群聊和私聊历史记录的保存和读取
  - **llm_utils.py**: 提供大语言模型调用相关的工具方法
  - **text_filter.py**: 处理大模型回复的文本过滤
  - **persona_utils.py**: 人格处理相关的工具方法
  - **reply_decision.py**: 决策是否需要对消息进行回复
  - **message_utils.py**: 消息处理和转换工具
  - **quote_utils.py**: 让模型自主决定是否引用、引用哪条消息
  - **image_caption.py**: 图片描述和转述功能

## 数据存储

插件数据存储在AstrBot的数据目录中：

```
AstrBot/data
└── chat_history/       
    └── 消息平台名称 如:aiocqhttp     # 各个消息平台的历史记录文件
      └── group/private              # 群聊和私聊文件分开存储
        └── {群号/qq号}.pkl           # 历史记录文件 
```

历史记录文件格式为pkl，包含完整的Astrbot消息对象。

## 插件工作流程

1. 插件初始化时加载配置并初始化各个工具类
2. 接收到消息时，保存到历史记录
3. 通过ReplyDecision判断是否需要回复
4. 如需回复，使用LLMUtils调用大模型
5. 处理大模型回复，应用文本过滤
6. 发送回复消息
7. 更新历史记录

## 扩展性

插件设计遵循模块化原则，各个工具类负责特定功能，便于扩展和维护：

- 添加新的消息处理功能：扩展message_utils.py
- 添加新的人格处理逻辑：扩展persona_utils.py
- 添加新的回复决策规则：扩展reply_decision.py
- 添加新的命令：在main.py中添加新的命令处理方法 
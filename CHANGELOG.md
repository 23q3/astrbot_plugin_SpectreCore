# 📋 更新日志

### v2.1.1 (2025-04-29)
- ✨ **新增黑名单关键词功能** - 添加黑名单关键词配置，可以设置不触发回复的关键词
- 🐛 **修复重置历史记录问题** - 修复重置聊天记录后提示消息被错误保存到历史记录的问题 [#41](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/41)
- 🔧 **优化配置结构** - 将黑名单关键词配置移动到model_frequency配置下，优化配置结构

### v2.1.0 (2025-04-12)
- 🔄 **数据存储格式优化** - 使用jsonpickle库替换pickle，提高数据可读性和跨平台兼容性
- 🐛 **修复Docker环境兼容性** - 采用JSON序列化格式，彻底解决Docker环境下消息历史存取问题 [#31](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/31)
- 🐛 **修复私聊时无法正确保存bot消息的问题** - 修复了在私聊场景下bot发送的消息无法被正确保存到历史记录的问题，确保私聊对话的完整性

### v2.0.1 (2025-04-11)
- 🐛 **修复Docker部署问题** - 改进路径处理方式，修复在Docker环境下无法保存/读取消息历史的问题

### v2.0.0 (2025-04-08)
- 🏗️ 完全重构 抛弃使用协议端API获取聊天记录的方式，改为基于Astrbot本身，支持了更多消息平台 [#21](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/21) [#4](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/4)
- 🔄 架构改进 采用高度模块化设计，每个功能封装在独立工具类中
- 📸 图片转述 支持图片转述功能 [#16](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/16)

### v1.0.4 (2025-03-12)
- 🐛 修正处理大模型回复时的条件判断逻辑 [#15](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/15)

### v1.0.3 (2025-03-11)
- 🐛 在处理大模型回复时增加了对角色的判断，避免调用函数工具时出错 [#15](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/15)
- 🐛 在提示词增加了bot的昵称和qq号，避免大模型不知道聊天记录中哪个是自己 [#14](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/14)

### v1.0.2 (2025-03-08)
- 🔒 添加了群组锁机制，防止并发调用大模型
- 🛠️ 优化了消息处理存储流程，极大提高了性能
- 🔍 添加了清除聊天记录的指令
- 🔍 添加了检测指令关键词不回复功能
- 📝 改进了代码结构

### v1.0.1 (2025-03-05)
- 🔍 增加了读空气功能
- 🔍 增加了函数工具开关配置
- 🔄 更换了request_llm方法调用大模型，提高兼容性
- 🛠️ 优化部分代码

### v1.0.0 (2025-03-04)
- 🎉 首次发布
- ✨ 实现基本的群聊互动功能 
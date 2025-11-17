# 📋 更新日志

### v2.1.7 (2025-11-17)

- 🐛 **修复AstrBot兼容性问题** - 移除对不存在的 `Anonymous` 等消息组件类的依赖，使用类型字符串检查替代类实例检查，兼容 AstrBot 新版本
- ⚡ **优化私聊回复机制** - 将私聊回复概率固定为1，确保历史消息格式统一
- ⚡ **优化引用消息显示** - 改进 Reply 组件处理，提供更完整的发送者信息和内容
- ⚡ **优化@消息处理** - 重构 'at' 和 'atall' 组件处理逻辑，提高代码清晰度和功能性
- [#72](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/72) [#73](https://github.com/23q3/astrbot_plugin_SpectreCore/pull/73) @Hola-Gracias

### v2.1.6 (2025-10-13)

- 🐛 **修复空消息异常** - 修复napcat发送私聊"正在输入"状态时，导致插件异常的问题 [#70](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/70)
- 🐛 **修复代码问题并改进错误处理** - 修复装饰器参数位置错误、历史记录加载失败崩溃问题，改进路径处理和日志记录

### v2.1.5 (2025-09-03)

- ✨ **新增临时禁言功能** - 添加闭嘴/说话指令，支持临时禁用自动回复功能，默认5分钟，可自定义时长 [#63](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/63)
- 🔒 **添加权限控制** - 对管理指令添加了管理员权限限制

### v2.1.4 (2025-06-28)

- ✨ **为LLM回复添加限制条件** - 在结尾提示词中添加对于回复的限制，避免LLM使用[At:id(昵称)]这样的格式 [#54](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/54)@Fossssss
- 🐛 **修复与其他插件的兼容性问题** - 移除AiocqhttpMessageEvent类型断言，使用安全检查和异常处理机制，解决其他插件构造输入时的报错问题 [#57](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/57)

### v2.1.3 (2025-06-17)

- ✨ **新增图片持久化存储功能** - 添加图片本地存储和自动清理机制，解决聊天记录中图片链接过期问题。新增 `enable_image_persistence` 和 `image_retention_days` 配置选项 [#52](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/52)
- 🐛 **修复读空气功能干扰命令执行的问题** - 将读空气处理逻辑从on_llm_response移至on_decorating_result阶段，避免在大模型回复后立即停止事件传播导致命令逻辑被中断 [#33](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/33)

### v2.1.2 (2025-06-02)

- 🐛 **修复Reply消息处理错误** - 修复在处理包含回复消息的历史记录时出现'Reply' object has no attribute 'sender_str'错误的问题 [#46](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/46)

### v2.1.1 (2025-04-29)

- ✨ **新增黑名单关键词功能** - 添加黑名单关键词配置，可以设置不触发回复的关键词
- 🐛 **修复重置历史记录问题** - 修复重置聊天记录后提示消息被错误保存到历史记录的问题 [#41](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/41)

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

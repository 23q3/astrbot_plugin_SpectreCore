# 🌟 SpectreCore (影芯) - 智能群聊互动插件

<div align="center">

![SpectreCore](https://avatars.githubusercontent.com/u/129108081?s=48&v=4)

[![version](https://img.shields.io/badge/version-v2.1.9-blue.svg?style=flat-square)](https://github.com/23q3/astrbot_plugin_SpectreCore)
[![license](https://img.shields.io/badge/license-AGPL--3.0-green.svg?style=flat-square)](LICENSE)
[![author](https://img.shields.io/badge/author-23q3-orange.svg?style=flat-square)](https://github.com/23q3)

*让 AI 成为群聊中最受欢迎的成员*

</div>

<p align="center">
  <a href="#-简介">简介</a> •
  <a href="#-特性">特性</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-文档">文档</a> •
  <a href="#-最新版本">最新版本</a> •
  <a href="#-注意事项">注意事项</a>
</p>

---

## 📝 简介

<table>
<tr>
<td>

SpectreCore (影芯) 是一个为 AstrBot 设计的高级群聊互动插件，它能让大语言模型更好地参与到群聊对话中，带来生动和沉浸式的群聊体验。通过智能的消息处理和上下文理解，让 AI 像真实群友一样自然地参与讨论。

> 💡 **灵感来源**：本项目的灵感来自 [【当你发现聊了99+的群友是DeepSeek.......】](https://www.bilibili.com/video/BV1amAneGE3P)

</td>
</tr>
</table>

## ✨ 特性

- 📸 **支持图片理解** - 支持图片转述功能，也可以直接输入图片
- 💾 **图片持久化存储** - 自动保存聊天中的图片到本地，解决图片链接过期问题，支持自动清理过期图片
- 🔍 **支持读空气** - AI 可智能判断何时应该回复，何时保持沉默
- 🤐 **临时禁言功能** - 支持通过指令临时禁用自动回复，可自定义时长
- 📨 **高级消息格式处理** - 支持让大模型理解At、引用等消息
- 🌐 **多平台兼容** - 基于Astrbot，支持所有Astrbot支持的消息平台
- 🏗️ **模块化设计** - 采用高度模块化的架构，每个功能都被封装在独立的工具类中，便于维护和扩展

## 🚀 快速开始

### 安装

1. 首先确保已部署 [AstrBot](https://github.com/Soulter/AstrBot)
2. 在插件市场中搜索 SpectreCore 点击安装
   或点击右下角加号，输入本插件仓库链接安装：

   ```
   https://github.com/23q3/astrbot_plugin_SpectreCore
   ```

3. 打开插件的配置界面，配置好相关选项，即可开始使用

## 📖 文档

- [指令说明](./docs/commands.md) - 详细的指令用法和参数说明
- [使用技巧](./docs/tips.md) - 提高插件使用效果的技巧和建议
- [项目结构](./docs/structure.md) - 项目代码结构和核心文件说明
- [更新日志](./CHANGELOG.md) - 项目版本历史和功能变化

## 📋 最新版本

### v2.1.9 (2026-01-05)

- 🐛 **修复GIF图片格式识别错误** - 修复图片URL优先级问题，优先使用file字段以保留准确格式信息，避免HTTP URL下载后MIME类型检测错误导致Gemini API返回400 [#81](https://github.com/23q3/astrbot_plugin_SpectreCore/issues/81)

## ⚠️ 注意事项

- 本插件和 AstrBot 自带的主动回复功能之间没有任何联系，在使用本插件时请关闭 AstrBot 的主动回复功能，以免重复回复
- 为避免不必要的响应，建议开启读空气功能并为 AI 提示明确的回复条件

### ⚠️ 更新到2.0.0版本的用户注意

- 删除Astrbot目录下的 data\config\astrbot_plugin_SpectreCore_config.json 否则会出现重复配置 且排版混乱(Astrbot特性导致)
- 这次更新是一次完全重构，抛弃了使用协议端api获取聊天记录的方式，而是基于Astrbot本身。

- 这导致了 对于aiocqhttp平台 插件不再支持以下功能:
  - 解析合并转发消息
  - 解析猜拳骰子结果
  - 获取QQ表情描述
  - 完全溯源的引用消息
  - 获取被@人的用户名

- 如果你有相关需求，可以下载之前的版本，但请注意旧版本已不再维护。

## 🔗 相关项目

<div align="center">
<table>
<tr>
<td align="center">
<a href="https://github.com/SengokuCola/MaiMBot">
<img src="https://avatars.githubusercontent.com/u/25811392?s=48&v=4" width="80" alt="MaiMBot"><br>
<sub><b>MaiMBot (麦麦)</b></sub>
</a><br>
<sub>一款专注于群组聊天的赛博网友QQ机器人</sub>
</td>
</tr>
</table>
</div>

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来帮助改进这个项目！

[![contributors](https://contrib.rocks/image?repo=23q3/astrbot_plugin_SpectreCore)](https://github.com/23q3/astrbot_plugin_SpectreCore/graphs/contributors)

## 📄 许可证

本项目采用 GNU Affero General Public License v3.0 (AGPL-3.0) 许可证。

详细信息请查看 [LICENSE](LICENSE) 文件。

---

<div align="center">

**[SpectreCore (影芯)](https://github.com/23q3/astrbot_plugin_SpectreCore)** | Powered by [AstrBot](https://github.com/AstrBotDevs/AstrBot)

<sub>Made with ❤️ by [23q3](https://github.com/23q3)</sub>

</div>

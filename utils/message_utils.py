from astrbot.api.all import *
from typing import List, Dict, Any, Optional
import os
import time
from datetime import datetime
from .image_caption import ImageCaptionUtils
import asyncio
import json
import traceback
from urllib.parse import urlparse, unquote


class MessageUtils:
    """
    消息处理工具类
    """

    @staticmethod
    def _file_uri_to_path(uri: str) -> Optional[str]:
        """
        将 file:// URI 转为本地路径（更稳，支持转义字符）。
        返回转换后的路径；如果不是 file uri 或解析失败，返回 None。
        """
        if not isinstance(uri, str):
            return None
        if not uri.startswith("file:"):
            return None

        try:
            parsed = urlparse(uri)
            # 典型: file:///tmp/a.png -> parsed.path = "/tmp/a.png"
            # 典型: file:///C:/a.png (Windows) -> parsed.path = "/C:/a.png"
            path = unquote(parsed.path or "")
            if not path:
                return None

            # Windows 兼容：去掉形如 "/C:/..." 的前导斜杠
            if os.name == "nt" and len(path) >= 3 and path[0] == "/" and path[2] == ":":
                path = path[1:]

            return path
        except Exception:
            return None

    @staticmethod
    async def format_history_for_llm(
        history_messages: List[AstrBotMessage],
        max_messages: int = 20,
        umo: Optional[str] = None
    ) -> str:
        """
        将历史消息列表格式化为适合输入给大模型的文本格式

        Args:
            history_messages: 历史消息列表
            max_messages: 最大消息数量，默认20条
            umo: unified_msg_origin，用于 UMO 路由

        Returns:
            格式化后的历史消息文本
        """
        if not history_messages:
            return ""

        # 限制消息数量
        if len(history_messages) > max_messages:
            history_messages = history_messages[-max_messages:]

        divider = "\n-\n"
        message_blocks: List[str] = []

        for msg in history_messages:
            # 获取发送者信息
            sender_name = "未知用户"
            sender_id = "unknown"
            if hasattr(msg, "sender") and msg.sender:
                sender_name = msg.sender.nickname or "未知用户"
                sender_id = msg.sender.user_id or "unknown"

            # 获取发送时间
            send_time = "未知时间"
            if hasattr(msg, "timestamp") and msg.timestamp:
                try:
                    time_obj = datetime.fromtimestamp(msg.timestamp)
                    send_time = time_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            # 获取消息内容 (异步调用)
            message_content = ""
            if hasattr(msg, "message") and msg.message:
                message_content = await MessageUtils.outline_message_list(msg.message, umo=umo)

            # 格式化该条消息（用 list + join，避免大量字符串累加）
            lines = [
                f"发送者: {sender_name} (ID: {sender_id})",
                f"时间: {send_time}",
                f"内容: {message_content}",
            ]
            message_blocks.append("\n".join(lines))

        return divider.join(message_blocks)

    @staticmethod
    async def outline_message_list(
        message_list: List[BaseMessageComponent],
        umo: Optional[str] = None
    ) -> str:
        """
        获取消息概要。
        修改点：
        1) 不再依赖 isinstance(Reply/Plain/...) ，只根据 component_type 分流
        2) 用 list + join 代替字符串累加，性能更好
        3) file:/// 解析使用 urlparse + unquote，避免路径截断 bug

        Args:
            message_list: 消息组件列表
            umo: unified_msg_origin，用于 UMO 路由
        """
        parts: List[str] = []

        for i in message_list:
            try:
                # 获取组件类型
                component_type = getattr(i, "type", None)
                if not component_type:
                    component_type = i.__class__.__name__
                # 兼容 enum / 常量类型
                component_type = getattr(component_type, "value", component_type)
                component_type = str(component_type).lower()

                # 特别优化 Reply 组件的处理
                if component_type == "reply":
                    parts.append(await MessageUtils._format_reply_component(i, umo=umo))
                    continue

                # 根据类型处理不同的消息组件
                if component_type == "plain":
                    parts.append(getattr(i, "text", ""))
                elif component_type == "image":
                    # 图片处理逻辑
                    try:
                        image = getattr(i, "file", None) or getattr(i, "url", None)
                        if image:
                            # 更稳的 file:// URI 转路径
                            local_path = MessageUtils._file_uri_to_path(image)
                            if local_path:
                                if not os.path.exists(local_path):
                                    logger.warning(f"持久化图片文件不存在: {local_path}")
                                    parts.append("[图片: 文件不存在]")
                                    continue
                                image = local_path

                            caption = await ImageCaptionUtils.generate_image_caption(image, umo=umo)
                            if caption:
                                parts.append(f"[图片: {caption}]")
                            else:
                                parts.append("[图片]")
                        else:
                            parts.append("[图片]")
                    except Exception as e:
                        logger.error(f"处理图片消息失败: {e}")
                        parts.append("[图片]")
                elif component_type == "face":
                    parts.append(f"[表情:{getattr(i, 'id', '')}]")
                elif component_type == "at":
                    qq = getattr(i, "qq", "")
                    name = getattr(i, "name", "")

                    # 处理全体@
                    if str(qq).lower() == "all":
                        parts.append("@全体成员")
                    # 有昵称时显示昵称+QQ
                    elif name:
                        parts.append(f"@{name}({qq})")
                    # 没有昵称时只显示QQ
                    else:
                        parts.append(f"@{qq}")
                elif component_type == "record":
                    parts.append("[语音]")
                elif component_type == "video":
                    parts.append("[视频]")
                elif component_type == "share":
                    title = getattr(i, "title", "")
                    content = getattr(i, "content", "") if getattr(i, "content", None) else ""
                    parts.append(f"[分享:《{title}》{content}]")
                elif component_type == "contact":
                    parts.append(f"[联系人:{getattr(i, 'id', '')}]")
                elif component_type == "location":
                    title = getattr(i, "title", "")
                    content = getattr(i, "content", "") if getattr(i, "content", None) else ""
                    parts.append(f"[位置:{title}{f'({content})' if content else ''}]")
                elif component_type == "music":
                    title = getattr(i, "title", "")
                    content = getattr(i, "content", "") if getattr(i, "content", None) else ""
                    parts.append(f"[音乐:{title}{f'({content})' if content else ''}]")
                elif component_type == "poke":
                    parts.append(f"[戳一戳 对:{getattr(i, 'qq', '')}]")
                elif component_type in ["forward", "node", "nodes"]:
                    parts.append("[合并转发消息]")
                elif component_type == "json":
                    # JSON处理逻辑
                    data = getattr(i, "data", None)
                    if isinstance(data, str):
                        try:
                            json_data = json.loads(data)
                            if "prompt" in json_data:
                                parts.append(f"[JSON卡片:{json_data.get('prompt', '')}]")
                            elif "app" in json_data:
                                parts.append(f"[小程序:{json_data.get('app', '')}]")
                            else:
                                parts.append("[JSON消息]")
                        except (json.JSONDecodeError, ValueError, TypeError):
                            parts.append("[JSON消息]")
                    else:
                        parts.append("[JSON消息]")
                elif component_type in ["rps", "dice", "shake"]:
                    parts.append(f"[{component_type}]")
                elif component_type == "file":
                    parts.append(f"[文件:{getattr(i, 'name', '')}]")
                elif component_type == "wechatemoji":
                    parts.append("[微信表情]")
                else:
                    # 处理被移除/未知的组件类型
                    if component_type == "anonymous":
                        parts.append("[匿名]")
                    elif component_type == "redbag":
                        parts.append("[红包]")
                    elif component_type == "xml":
                        parts.append("[XML消息]")
                    elif component_type == "cardimage":
                        parts.append("[卡片图片]")
                    elif component_type == "tts":
                        parts.append("[TTS]")
                    else:
                        parts.append(f"[{component_type}]")

            except Exception as e:
                logger.error(f"处理消息组件时出错: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                parts.append("[处理失败的消息组件]")
                continue

        return " ".join(parts)

    @staticmethod
    async def _format_reply_component(reply_component: Any, umo: Optional[str] = None) -> str:
        """
        优化格式化引用回复组件

        Args:
            reply_component: 回复组件
            umo: unified_msg_origin，用于 UMO 路由
        """
        try:
            # 构建发送者信息
            sender_id = getattr(reply_component, "sender_id", "")
            sender_nickname = getattr(reply_component, "sender_nickname", "")

            if sender_nickname:
                sender_info = f"'{sender_nickname}'({sender_id})"
            elif sender_id:
                sender_info = f"'{sender_id}'"
            else:
                sender_info = "未知用户"

            # 获取被引用消息的时间
            reply_time = ""
            raw_ts = getattr(reply_component, "time", None) or getattr(reply_component, "timestamp", None)
            if raw_ts:
                try:
                    ts = int(raw_ts)
                    time_obj = datetime.fromtimestamp(ts)
                    reply_time = time_obj.strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    pass

            # 获取被引用消息的内容
            if getattr(reply_component, "chain", None):
                reply_content = await MessageUtils.outline_message_list(reply_component.chain, umo=umo)
            elif getattr(reply_component, "message_str", None):
                reply_content = reply_component.message_str
            elif getattr(reply_component, "text", None):
                reply_content = reply_component.text
            else:
                reply_content = "[内容不可用]"

            # 限制回复内容长度，避免过长
            if len(reply_content) > 30:
                reply_content = reply_content[:30] + "..."

            # 构建格式化的回复显示
            if reply_time:
                return f"<引用「{reply_time} {sender_info}：{reply_content}」>"
            return f"<引用「{sender_info}：{reply_content}」>"

        except Exception as e:
            logger.error(f"格式化回复组件时出错: {e}")
            return "[回复消息]"

from astrbot.api.all import *
from typing import List, Dict, Any
import os
import time
from datetime import datetime
from .image_caption import ImageCaptionUtils
import asyncio
    


class MessageUtils:
    """
    消息处理工具类
    """
        
    @staticmethod
    async def format_history_for_llm(history_messages: List[AstrBotMessage], max_messages: int = 20) -> str:
        """
        将历史消息列表格式化为适合输入给大模型的文本格式
        
        Args:
            history_messages: 历史消息列表
            max_messages: 最大消息数量，默认20条
            
        Returns:
            格式化后的历史消息文本
        """
        if not history_messages:
            return ""
        
        # 限制消息数量
        if len(history_messages) > max_messages:
            history_messages = history_messages[-max_messages:]
        
        formatted_text = ""
        divider = "\n" + "-" + "\n"
        
        for idx, msg in enumerate(history_messages):
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
                except:
                    pass
            
            # 获取消息内容 (异步调用)
            message_content = await MessageUtils.outline_message_list(msg.message) if hasattr(msg, "message") and msg.message else ""
            
            # 格式化该条消息
            message_text = f"发送者: {sender_name} (ID: {sender_id})\n"
            message_text += f"时间: {send_time}\n"
            message_text += f"内容: {message_content}"
            
            # 添加到结果中
            formatted_text += message_text
            
            # 除了最后一条消息，每条消息后添加分割线
            if idx < len(history_messages) - 1:
                formatted_text += divider
        
        return formatted_text
           
    @staticmethod
    async def outline_message_list(message_list: List[BaseMessageComponent]) -> str:
        """
        获取消息概要。
        使用类型检查而不是类实例检查，避免依赖不存在的类。
        """
        outline = ""
        for i in message_list:
            try:
                # 获取组件类型
                component_type = getattr(i, 'type', None)
                if not component_type:
                    component_type = i.__class__.__name__.lower()
                
                # 特别优化 Reply 组件的处理
                if component_type == "reply" or isinstance(i, Reply):
                    outline += await MessageUtils._format_reply_component(i)
                    continue
                
                # 根据类型处理不同的消息组件
                elif component_type == "plain" or isinstance(i, Plain):
                    outline += i.text
                elif component_type == "image" or isinstance(i, Image):
                    # 图片处理逻辑
                    try:
                        image = i.file if i.file else i.url
                        if image:
                            if image.startswith("file:///"):
                                image_path = image[8:]
                                if not os.path.exists(image_path):
                                    logger.warning(f"持久化图片文件不存在: {image_path}")
                                    outline += f"[图片: 文件不存在]"
                                    continue
                                image = image_path

                            caption = await ImageCaptionUtils.generate_image_caption(image)
                            if caption:
                                outline += f"[图片: {caption}]"
                            else:
                                outline += f"[图片]"
                        else:
                            outline += f"[图片]"
                    except Exception as e:
                        logger.error(f"处理图片消息失败: {e}")
                        outline += "[图片]"
                elif component_type == "face" or isinstance(i, Face):
                    outline += f"[表情:{getattr(i, 'id', '')}]"
                elif component_type == "at" or isinstance(i, At):
                    qq = getattr(i, 'qq', '')
                    name = getattr(i, 'name', '')
                    
                    # 处理全体@
                    if str(qq).lower() == "all":
                        outline += "@全体成员"
                    # 有昵称时显示昵称+QQ
                    elif name:
                        outline += f"@{name}({qq})"
                    # 没有昵称时只显示QQ
                    else:
                        outline += f"@{qq}"
                elif component_type == "record" or isinstance(i, Record):
                    outline += "[语音]"
                elif component_type == "video" or isinstance(i, Video):
                    outline += "[视频]"
                elif component_type == "share" or isinstance(i, Share):
                    outline += f"[分享:《{getattr(i, 'title', '')}》{getattr(i, 'content', '') if hasattr(i, 'content') and i.content else ''}]"
                elif component_type == "contact" or isinstance(i, Contact):
                    outline += f"[联系人:{getattr(i, 'id', '')}]"
                elif component_type == "location" or isinstance(i, Location):
                    outline += f"[位置:{getattr(i, 'title', '')}{f'({i.content})' if hasattr(i, 'content') and i.content else ''}]"
                elif component_type == "music" or isinstance(i, Music):
                    outline += f"[音乐:{getattr(i, 'title', '')}{f'({i.content})' if hasattr(i, 'content') and i.content else ''}]"
                elif component_type == "poke" or isinstance(i, Poke):
                    outline += f"[戳一戳 对:{getattr(i, 'qq', '')}]"
                elif component_type in ["forward", "node", "nodes"] or isinstance(i, (Forward, Node, Nodes)):
                    outline += f"[合并转发消息]"
                elif component_type == "json" or isinstance(i, Json):
                    # JSON处理逻辑
                    data = getattr(i, 'data', None)
                    if isinstance(data, str):
                        try:
                            import json
                            json_data = json.loads(data)
                            if "prompt" in json_data:
                                outline += f"[JSON卡片:{json_data.get('prompt', '')}]"
                            elif "app" in json_data:
                                outline += f"[小程序:{json_data.get('app', '')}]"
                            else:
                                outline += "[JSON消息]"
                        except:
                            outline += "[JSON消息]"
                    else:
                        outline += "[JSON消息]"
                elif component_type in ["rps", "dice", "shake"] or isinstance(i, (RPS, Dice, Shake)):
                    # 这些可能是游戏类型的消息
                    outline += f"[{component_type}]"
                elif component_type == "file" or isinstance(i, File):
                    outline += f"[文件:{getattr(i, 'name', '')}]"
                elif component_type == "wechatemoji" or isinstance(i, WechatEmoji):
                    outline += "[微信表情]"
                elif component_type == "reply" or isinstance(i, Reply):
                    # 回复处理逻辑
                    if hasattr(i, 'chain') and i.chain:
                        sender_info = f"{getattr(i, 'sender_nickname', '')}({getattr(i, 'sender_id', '')})" if hasattr(i, 'sender_nickname') and i.sender_nickname else f"{getattr(i, 'sender_id', '')}"
                        reply_content = await MessageUtils.outline_message_list(i.chain)
                        outline += f"[回复({sender_info}: {reply_content})]"
                    elif hasattr(i, 'message_str') and i.message_str:
                        sender_info = f"{getattr(i, 'sender_nickname', '')}({getattr(i, 'sender_id', '')})" if hasattr(i, 'sender_nickname') and i.sender_nickname else f"{getattr(i, 'sender_id', '')}"
                        outline += f"[回复({sender_info}: {i.message_str})]"
                    elif hasattr(i, 'sender_nickname') and i.sender_nickname or hasattr(i, 'sender_id') and i.sender_id:
                        sender_info = f"{getattr(i, 'sender_nickname', '')}({getattr(i, 'sender_id', '')})" if hasattr(i, 'sender_nickname') and i.sender_nickname else f"{getattr(i, 'sender_id', '')}"
                        outline += f"[回复({sender_info})]"
                    else:
                        outline += "[回复消息]"
                else:
                    # 处理被移除的组件类型
                    if component_type == "anonymous":
                        outline += "[匿名]"
                    elif component_type == "redbag":
                        outline += "[红包]"
                    elif component_type == "xml":
                        outline += "[XML消息]"
                    elif component_type == "cardimage":
                        outline += "[卡片图片]"
                    elif component_type == "tts":
                        outline += "[TTS]"
                    else:
                        # 未知类型的消息组件
                        outline += f"[{component_type}]"
                    
            except Exception as e:
                logger.error(f"处理消息组件时出错: {e}")
                logger.error(f"错误详情: {traceback.format_exc()}")
                outline += f"[处理失败的消息组件]"
                continue
                
        return outline

    @staticmethod
    async def _format_reply_component(reply_component: Reply) -> str:
        """
        优化格式化引用回复组件
        """
        try:
            # 构建发送者信息
            sender_id = getattr(reply_component, 'sender_id', '')
            sender_nickname = getattr(reply_component, 'sender_nickname', '')
            
            sender_info = ""
            if sender_nickname:
                sender_info = f"{sender_nickname}({sender_id})"
            elif sender_id:
                sender_info = f"{sender_id}"
            else:
                sender_info = "未知用户"
            
            # 获取被引用消息的内容
            reply_content = ""
            
            # 优先使用 chain（原始消息组件）
            if hasattr(reply_component, 'chain') and reply_component.chain:
                reply_content = await MessageUtils.outline_message_list(reply_component.chain)
            # 其次使用 message_str（纯文本消息）
            elif hasattr(reply_component, 'message_str') and reply_component.message_str:
                reply_content = reply_component.message_str
            # 最后使用 text（向后兼容）
            elif hasattr(reply_component, 'text') and reply_component.text:
                reply_content = reply_component.text
            else:
                reply_content = "[内容不可用]"
            
            # 限制回复内容长度，避免过长
            if len(reply_content) > 150:
                reply_content = reply_content[:150] + "..."
            
            # 构建格式化的回复显示
            formatted_reply = f"「↪ 引用消息 {sender_info}：{reply_content}」"
            
            return formatted_reply
            
        except Exception as e:
            logger.error(f"格式化回复组件时出错: {e}")
            return "[回复消息]"

from astrbot.api.all import *
from typing import Dict, List, Optional, Any
import time
import threading
from .history_storage import HistoryStorage
from .message_utils import MessageUtils
from astrbot.core.provider.entites import ProviderRequest
from .persona_utils import PersonaUtils

class LLMUtils:
    """
    大模型调用工具类
    用于构建提示词和调用记录相关功能
    """
    
    # 使用字典保存每个聊天的大模型调用状态
    # 格式: {"{platform_name}_{chat_type}_{chat_id}": {"last_call_time": timestamp, "in_progress": True/False}}
    _llm_call_status: Dict[str, Dict[str, Any]] = {}
    _lock = threading.Lock()  # 用于线程安全的锁
    
    @staticmethod
    def get_chat_key(platform_name: str, is_private_chat: bool, chat_id: str) -> str:
        """
        获取聊天的唯一标识
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            聊天的唯一标识
        """
        chat_type = "private" if is_private_chat else "group"
        return f"{platform_name}_{chat_type}_{chat_id}"
    
    @staticmethod
    def set_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str, in_progress: bool = True) -> None:
        """
        设置大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            in_progress: 是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                LLMUtils._llm_call_status[chat_key] = {}
                
            LLMUtils._llm_call_status[chat_key]["in_progress"] = in_progress
            LLMUtils._llm_call_status[chat_key]["last_call_time"] = time.time()
    
    @staticmethod
    def is_llm_in_progress(platform_name: str, is_private_chat: bool, chat_id: str) -> bool:
        """
        检查指定聊天是否正在进行大模型调用
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            是否正在进行大模型调用
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return False
                
            return LLMUtils._llm_call_status[chat_key].get("in_progress", False)
    
    @staticmethod
    def get_last_call_time(platform_name: str, is_private_chat: bool, chat_id: str) -> Optional[float]:
        """
        获取指定聊天最后一次大模型调用的时间戳
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            最后一次调用的时间戳，如果从未调用过则返回None
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key not in LLMUtils._llm_call_status:
                return None
                
            return LLMUtils._llm_call_status[chat_key].get("last_call_time")
    
    @staticmethod
    async def call_llm(event: AstrMessageEvent, config: AstrBotConfig, context: Context) -> ProviderRequest:
        """
        构建调用大模型的请求对象
        
        Args:
            event: 消息对象
            config: 配置对象
            context: Context 对象，用于获取LLM工具管理器
            
        Returns:
            ProviderRequest 对象
        """
        platform_name = event.get_platform_name()
        is_private = event.is_private_chat()
        chat_id = event.get_group_id() if not is_private else event.get_sender_id()
        
        # 构建基础Prompt 
        # 对于aiocqhttp平台 通过调用协议端api获取bot用户名
        if platform_name == "aiocqhttp":
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import AiocqhttpMessageEvent
            assert isinstance(event, AiocqhttpMessageEvent)
            client = event.bot
            bot_name = (await client.api.get_login_info())["nickname"]
            prompt = f"你正在浏览聊天软件，你在聊天软件上的id是{event.get_self_id()}，用户名是{bot_name}，你正在"
        else:
            prompt = f"你正在浏览聊天软件，你在聊天软件上的id是{event.get_self_id()}，你正在"

        if is_private:
            sender_display_name = event.get_sender_name() if event.get_sender_name() else f"ID为 {event.get_sender_id()} 的人"
            prompt += f"和 {sender_display_name} 私聊页面中。"
        else:
            group_display_name = chat_id 
            # 尝试获取更详细的群名
            if platform_name in ["aiocqhttp", "gewechat"]:
                try:
                    group = await event.get_group() 
                    if group and group.group_name:
                        group_display_name = f"{group.group_name}({chat_id})" 
                except Exception as e:                   
                    logger.warning(f"为 {platform_name} 获取群组信息失败: {e}")
            
            prompt += f"群聊 {group_display_name} 中。"
            
        # 添加历史记录
        try:
            history_limit = config.get("group_msg_history", 10) # 从配置读取历史记录数量
            history_messages = HistoryStorage.get_history(platform_name, is_private, chat_id)
            if history_messages:
                formatted_history = (await MessageUtils.format_history_for_llm(history_messages, max_messages=history_limit))
                prompt += "\n\n以下是最近的聊天记录：\n" + formatted_history
            else:
                prompt += "\n\n你没看见任何聊天记录，看来最近没有消息。"
        except Exception as e:
            logger.error(f"获取或格式化历史记录失败: {e}")
            return None

        # 结尾提示词
        prompt += "\n(在聊天记录中，你的用户名以AstrBot被代替了)"
        prompt += "\n(如果你想回复某人，不要使用类似 [At:id(昵称)]这样的格式)"
        # 判断是否开启读空气
        if config.get("read_air", False):
            prompt += "\n\n你的反应是:\n(如果你想发送一条消息，直接输出发送的内容，如果你选择忽略，直接输出<NO_RESPONSE>)"
        else:
            prompt += "\n\n你决定发送一条消息(你输出的内容将作为消息发送)"
        
        # 准备并调用大模型
        func_tools_mgr = context.get_llm_tool_manager() if config.get("use_func_tool", False) else None
        
        # 获取配置中指定的人格
        system_prompt = ""
        contexts = []
        persona_name = config.get("persona", "")
        
        if persona_name:
            try:
                # 获取对应的人格
                persona = PersonaUtils.get_persona_by_name(context, persona_name)
                # 处理人格
                if persona:
                    system_prompt = persona.get('prompt', '')
                    if persona.get('_mood_imitation_dialogs_processed'):
                        mood_dialogs = persona.get('_mood_imitation_dialogs_processed', [])
                        system_prompt += "\n请模仿以下示例的对话风格来反应(示例中，a代表用户，b代表你)\n" + mood_dialogs

                    begin_dialogs = persona.get('_begin_dialogs_processed', [])
                    if begin_dialogs:
                        contexts.extend(begin_dialogs)
                                        
                    logger.debug(f"找到人格 '{persona_name}' ")
                else:
                    logger.warning(f"未找到名为 '{persona_name}' 的人格")
            except Exception as e:
                logger.error(f"获取人格信息失败: {e}")

        # 图片相关处理
        image_urls = []
        if image_count := config.get("image_processing", {}).get("image_count", 0):
            # 只从会发送给大模型的历史消息中提取图片（受history_limit限制）
            if history_messages:
                # 先计算将给大模型的消息范围
                messages_to_show = history_messages[-history_limit:] if len(history_messages) > history_limit else history_messages
                
                # 按时间从新到旧遍历消息
                for message in reversed(messages_to_show):
                    # 检查消息是否包含图片
                    if hasattr(message, "message") and message.message:
                        for component in message.message:
                            # 判断是否为图片组件
                            if isinstance(component, Image):
                                try:
                                    # 只使用本地路径格式
                                    if component.file:
                                        image_url = component.file
                                    else:
                                        continue  # 跳过其他格式
                                    
                                    image_urls.append(image_url)
                                        
                                    # 如果已收集足够数量的图片，停止收集
                                    if len(image_urls) >= image_count:
                                        break
                                except Exception as e:
                                    logger.warning(f"处理图片URL时出错: {e}")
                                    continue
                        
                        # 如果已收集足够数量的图片，停止遍历消息
                        if len(image_urls) >= image_count:
                            break
                
                # 如果收集到了图片，添加提示词
                if image_urls:
                    prompt += f"\n\n已经按照从晚到早的顺序为你提供了聊天记录中的{len(image_urls)}张图片，你可以直接查看并理解它们。这些图片出现在聊天记录中。"

        return event.request_llm(
            prompt=prompt,
            func_tool_manager=func_tools_mgr,
            contexts=contexts,
            system_prompt=system_prompt, 
            image_urls=image_urls, 
        )
    
    @staticmethod
    def clear_call_status(platform_name: str, is_private_chat: bool, chat_id: str) -> None:
        """
        清除指定聊天的大模型调用状态
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
        """
        chat_key = LLMUtils.get_chat_key(platform_name, is_private_chat, chat_id)
        
        with LLMUtils._lock:
            if chat_key in LLMUtils._llm_call_status:
                del LLMUtils._llm_call_status[chat_key] 

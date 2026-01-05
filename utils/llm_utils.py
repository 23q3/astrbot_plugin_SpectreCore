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

        # 准备并调用大模型
        func_tools_mgr = context.get_llm_tool_manager() if config.get("use_func_tool", False) else None

        # 获取配置中指定的人格
        system_prompt = ""
        contexts = []
        persona_name = config.get("persona", "")

        if persona_name:
            try:
                persona = PersonaUtils.get_persona_by_name(context, persona_name)
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

        # 构建环境描述（注入到 system_prompt，不污染 prompt）
        env_description = f"\n\n你正在浏览聊天软件，你在聊天软件上的id是{event.get_self_id()}"

        # 对于aiocqhttp平台，尝试获取bot用户名
        if platform_name == "aiocqhttp" and hasattr(event, "bot"):
            try:
                bot = getattr(event, "bot")
                bot_name = (await bot.api.get_login_info())["nickname"]
                env_description += f"，用户名是{bot_name}"
            except Exception as e:
                logger.warning(f"通过 event.bot 获取机器人昵称失败: {e}")

        if is_private:
            sender_display_name = event.get_sender_name() if event.get_sender_name() else f"ID为 {event.get_sender_id()} 的人"
            env_description += f"，你正在和 {sender_display_name} 私聊页面中。"
        else:
            group_display_name = chat_id
            if platform_name in ["aiocqhttp", "gewechat"]:
                try:
                    group = await event.get_group()
                    if group and group.group_name:
                        group_display_name = f"{group.group_name}({chat_id})"
                except Exception as e:
                    logger.warning(f"为 {platform_name} 获取群组信息失败: {e}")
            env_description += f"，你正在群聊 {group_display_name} 中。"

        # 添加历史记录（文本格式，注入到 system_prompt）
        # 注意：基于 message_id 精确排除当前消息，避免重复
        history_limit = config.get("group_msg_history", 10)
        history_messages = HistoryStorage.get_history(platform_name, is_private, chat_id)

        try:
            if history_messages:
                # 获取当前消息的 message_id 用于精确排除
                current_msg_id = getattr(event.message_obj, 'message_id', None) if hasattr(event, 'message_obj') else None
                if current_msg_id:
                    history_for_context = [m for m in history_messages if getattr(m, 'message_id', None) != current_msg_id]
                else:
                    # 回退到排除最后一条
                    history_for_context = history_messages[:-1] if len(history_messages) > 1 else []
                if history_for_context:
                    formatted_history = await MessageUtils.format_history_for_llm(history_for_context, max_messages=history_limit)
                    env_description += "\n\n以下是最近的聊天记录：\n" + formatted_history
                else:
                    env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"
            else:
                env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"
        except Exception as e:
            logger.error(f"获取或格式化历史记录失败: {e}")
            env_description += "\n\n你没看见任何聊天记录，看来最近没有消息。"

        # 行为指引
        env_description += "\n(在聊天记录中，你的用户名以AstrBot被代替了)"
        env_description += "\n(如果你想回复某人，不要使用类似 [At:id(昵称)]这样的格式)"

        if config.get("read_air", False):
            env_description += "\n\n现在你收到了一条新消息，你的反应是:\n(如果你想发送一条消息，直接输出发送的内容，如果你选择忽略，直接输出<NO_RESPONSE>)"
        else:
            env_description += "\n\n现在你收到了一条新消息，你决定发送一条消息回复(你输出的内容将作为消息发送)"

        # 将环境描述追加到 system_prompt
        system_prompt += env_description

        # 图片相关处理
        image_urls = []
        if image_count := config.get("image_processing", {}).get("image_count", 0):
            if history_messages:
                messages_to_show = history_messages[-history_limit:] if len(history_messages) > history_limit else history_messages

                for message in reversed(messages_to_show):
                    if hasattr(message, "message") and message.message:
                        for component in message.message:
                            if isinstance(component, Image):
                                try:
                                    url = component.file or component.url
                                    if url:
                                        image_urls.append(url)
                                        if len(image_urls) >= image_count:
                                            break
                                except Exception as e:
                                    logger.warning(f"处理图片URL时出错: {e}")
                                    continue
                        if len(image_urls) >= image_count:
                            break

                if image_urls:
                    system_prompt += f"\n\n已经按照从晚到早的顺序为你提供了聊天记录中的{len(image_urls)}张图片，你可以直接查看并理解它们。这些图片出现在聊天记录中。"

        # prompt 只保留用户当前消息，保持干净供 KB 检索
        prompt = event.get_message_outline()

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

from astrbot.api.all import *
from typing import Dict, Any, Optional
import random
from .llm_utils import LLMUtils

class ReplyDecision:
    """
    消息回复决策工具类
    用于判断是否要使用大模型回复消息
    """
    
    @staticmethod
    def should_reply(event: AstrMessageEvent, config: AstrBotConfig) -> bool:
        """
        判断是否应该回复消息
        
        Args:
            event: 消息事件
            config: 配置对象
            
        Returns:
            是否应该回复
        """
        try:
            # 获取必要信息
            platform_name = event.get_platform_name()
            is_private_chat = event.is_private_chat()
            chat_id = event.get_sender_id() if is_private_chat else event.get_group_id()
            
            # 检查是否已有大模型在处理
            if LLMUtils.is_llm_in_progress(platform_name, is_private_chat, chat_id):
                logger.debug(f"当前聊天已有大模型处理中，不进行回复")
                return False
                
            # 检查消息是否包含黑名单关键词
            blacklist_keywords = config.get("model_frequency", {}).get("blacklist_keywords", [])
            if blacklist_keywords and ReplyDecision._check_blacklist_keywords(event, blacklist_keywords):
                logger.debug("消息中包含黑名单关键词，不进行回复")
                return False
            
            # 检查配置中的回复规则
            return ReplyDecision._check_reply_rules(event, config)
        except Exception as e:
            logger.error(f"判断是否回复时发生错误: {e}")
            return False
    
    @staticmethod
    def _check_reply_rules(event: AstrMessageEvent, config: AstrBotConfig) -> bool:
        """
        检查回复规则
        
        Args:
            event: 消息事件
            config: 配置对象
            
        Returns:
            是否应该回复
        """
        # 检查是否是开启回复的群聊/私聊
        if event.is_private_chat():
            if not config.get("enabled_private", False):
                logger.debug("未开启私聊回复功能")
                return False
        else:
            if event.get_group_id() not in config.get("enabled_groups", []):
                logger.debug(f"群聊{event.get_group_id()}未开启回复功能")
                return False
            
        # 获取消息频率配置
        frequency_config = config.get("model_frequency", {})
        # 检查关键词触发
        keywords = frequency_config.get("keywords", [])
        if keywords and ReplyDecision._check_keywords(event, keywords):
            logger.debug("消息中包含关键词，触发回复")
            return True
        
        # 获取回复方法
        method = frequency_config.get("method", "概率回复")
        
        # 根据不同方法判断
        if method == "概率回复":
            prob_config = frequency_config.get("probability", {})
            probability = prob_config.get("probability", 0.1)
            
            # 使用概率计算是否回复
            should_reply = random.random() < probability
            if should_reply:
                logger.debug(f"概率触发回复，当前概率: {probability}")
            else:
                logger.debug(f"概率回复未触发，当前概率: {probability}")
            return should_reply
        
        # 为未来扩展预留接口
        # 可以在这里添加更多回复方法的判断逻辑
        
        return False
    
    @staticmethod
    def _check_keywords(event: AstrMessageEvent, keywords: list) -> bool:
        """
        检查消息是否包含关键词
        
        Args:
            event: 消息事件
            keywords: 关键词列表
            
        Returns:
            是否包含关键词
        """
        # 获取消息文本
        message_text = event.get_message_outline()
        
        # 检查是否包含关键词
        for keyword in keywords:
            if keyword in message_text:
                return True
                
        return False
        
    @staticmethod
    def _check_blacklist_keywords(event: AstrMessageEvent, blacklist_keywords: list) -> bool:
        """
        检查消息是否包含黑名单关键词
        
        Args:
            event: 消息事件
            blacklist_keywords: 黑名单关键词列表
            
        Returns:
            是否包含黑名单关键词
        """
        # 获取消息文本
        message_text = event.get_message_outline()
        
        # 检查是否包含黑名单关键词
        for keyword in blacklist_keywords:
            if keyword in message_text:
                return True
                
        return False

    @staticmethod
    async def process_and_reply(event: AstrMessageEvent, config: AstrBotConfig, context: Context):
        """
        处理消息并使用大模型回复
        
        Args:
            event: 消息事件
            config: 配置对象
            context: 上下文对象
            
        Yields:
            大模型的回复
        """
        # 获取必要信息
        platform_name = event.get_platform_name()
        is_private = event.is_private_chat()
        chat_id = event.get_sender_id() if is_private else event.get_group_id()
        
        # 标记开始处理
        LLMUtils.set_llm_in_progress(platform_name, is_private, chat_id)
        
        try:
            # 调用大模型并发送回复
            yield await LLMUtils.call_llm(event, config, context)
        finally:
            # 标记处理完成
            LLMUtils.set_llm_in_progress(platform_name, is_private, chat_id, False) 
import os
import jsonpickle
from typing import List
from astrbot.api.all import *
import time
import traceback

class HistoryStorage:
    """
    历史消息存储工具类
    
    按照平台->聊天类型->ID的层级结构存储消息
    使用jsonpickle序列化AstrBotMessage对象为JSON格式
    """
    
    # 保存配置对象的静态变量
    config = None
    # 基础存储路径
    base_storage_path = None
    
    @staticmethod
    def init(config: AstrBotConfig):
        """初始化配置对象"""
        HistoryStorage.config = config
        # 初始化基础存储路径
        HistoryStorage.base_storage_path = os.path.join(os.getcwd(), "data", "chat_history")
        HistoryStorage._ensure_dir(HistoryStorage.base_storage_path)
        logger.info(f"消息存储路径初始化: {HistoryStorage.base_storage_path}")
        
        # 配置jsonpickle
        jsonpickle.set_encoder_options('json', ensure_ascii=False, indent=2)
        jsonpickle.set_preferred_backend('json')
    
    @staticmethod
    def _ensure_dir(directory: str) -> None:
        """确保目录存在，不存在则创建"""
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    @staticmethod
    def _get_storage_path(platform_name: str, is_private_chat: bool, chat_id: str) -> str:
        """获取存储路径"""
        if not HistoryStorage.base_storage_path:
            # 确保基础路径已初始化，未初始化则初始化一次
            HistoryStorage.base_storage_path = os.path.join(os.getcwd(), "data", "chat_history")
            HistoryStorage._ensure_dir(HistoryStorage.base_storage_path)
            logger.info(f"消息存储路径初始化: {HistoryStorage.base_storage_path}")
            
        chat_type = "private" if is_private_chat else "group"
        directory = os.path.join(HistoryStorage.base_storage_path, platform_name, chat_type)
        
        HistoryStorage._ensure_dir(directory)
        return os.path.join(directory, f"{chat_id}.json")
    
    @staticmethod
    def _sanitize_message(message: AstrBotMessage) -> AstrBotMessage:
        """
        清理消息对象，移除可能导致序列化失败的属性
        
        Args:
            message: AstrBot消息对象
            
        Returns:
            清理后的消息对象
        """
        # 创建消息的浅复制以避免修改原始对象
        import copy
        sanitized_message = copy.copy(message)
        
        # 移除可能导致序列化问题的属性
        for attr in ['_client', '_callback', '_handler', '_context', 'raw_message']:
            if hasattr(sanitized_message, attr):
                setattr(sanitized_message, attr, None)
        
        return sanitized_message
    
    @staticmethod
    async def save_message(message: AstrBotMessage, platform_name: str, chat_id_override: str | None = None) -> bool:
        """
        保存消息到历史记录

        Args:
            message: AstrBot消息对象
            platform_name: 平台名称
            chat_id_override: 可选，强制指定存储用的聊天ID（用于bot消息存入正确的会话文件）

        Returns:
            是否保存成功
        """
        try:
            # 判断是群聊还是私聊
            is_private_chat = not bool(message.group_id)

            if chat_id_override is not None:
                chat_id = str(chat_id_override)
            elif is_private_chat:
                chat_id = message.sender.user_id
            else:
                chat_id = message.group_id
                
            # 获取存储路径
            file_path = HistoryStorage._get_storage_path(platform_name, is_private_chat, chat_id)
            
            # 读取现有历史记录
            history = HistoryStorage.get_history(platform_name, is_private_chat, chat_id)
            if not history:
                history = []
                
            # 处理图片持久化存储
            await HistoryStorage._process_image_persistence(message)

            # 清理消息对象，并添加到历史记录
            sanitized_message = HistoryStorage._sanitize_message(message)
            history.append(sanitized_message)

            # 限制历史记录数量
            if len(history) > 200:
                history = history[-200:]

            # 确保父目录存在
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 使用jsonpickle序列化对象
            json_data = jsonpickle.encode(history, unpicklable=True)

            # 写入文件
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_data)

            # 随机执行清理操作（避免每次都执行，减少性能影响）
            import random
            if random.random() < 0.05:  # 5% 的概率执行清理
                try:
                    HistoryStorage._cleanup_old_images()
                except Exception as e:
                    logger.error(f"执行图片清理时发生错误: {e}")

            return True
        except Exception as e:
            logger.error(f"保存消息历史记录失败: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    @staticmethod
    def is_chat_enabled(event: AstrMessageEvent) -> bool:
        """
        判断当前聊天是否启用保存功能
        
        Args:
            event: 消息事件
            
        Returns:
            是否启用
        """
        if not HistoryStorage.config:
            logger.warning("HistoryStorage配置未初始化，默认不启用保存功能")
            return False
            
        is_private = event.is_private_chat()
        if is_private:
            return HistoryStorage.config.get("enabled_private", False)
        else:
            group_id = event.get_group_id()
            if not group_id:
                return False
            group_id = str(group_id).strip()
            
            # 获取配置集合并规范化 (O(1) 查找)
            blocked_groups = {str(g).strip() for g in HistoryStorage.config.get("blocked_groups", []) if str(g).strip()}
            enabled_groups = {str(g).strip() for g in HistoryStorage.config.get("enabled_groups", []) if str(g).strip()}

            # 优先级: 黑名单 > 全局开关 > 白名单
            if group_id in blocked_groups:
                return False
            if HistoryStorage.config.get("enable_all_groups", False):
                return True
            return group_id in enabled_groups
    
    @staticmethod
    async def process_and_save_user_message(event: AstrMessageEvent) -> None:
        """
        处理用户消息并保存到历史记录
        
        Args:
            event: 消息事件
        """
        # 检查是否启用
        is_enabled = HistoryStorage.is_chat_enabled(event)
        if not is_enabled:
            chat_type = "私聊" if event.is_private_chat() else f"群聊{event.get_group_id()}"
            logger.debug(f"{chat_type}未开启回复功能")
            return
            
        # 创建消息对象
        message_obj = event.message_obj

        # 保存消息
        success = await HistoryStorage.save_message(message_obj, event.get_platform_name())
        
        chat_type = "私聊" if event.is_private_chat() else "群聊"
        if success:
            logger.debug(f"已保存{chat_type}消息到历史记录")
        else:
            logger.error(f"保存{chat_type}消息失败")
    
    @staticmethod
    def create_bot_message(chain: List[BaseMessageComponent], event: AstrMessageEvent) -> AstrBotMessage:
        """
        从消息链和事件对象创建一个机器人消息对象
        
        Args:
            chain: 消息链
            event: 触发消息的事件
            
        Returns:
            创建的AstrBotMessage对象
        """
        # 创建消息对象
        msg = AstrBotMessage()

        # 设置基本属性
        msg.message = chain
        msg.timestamp = int(time.time())
        
        # 设置消息类型和会话信息
        is_private = event.is_private_chat()
        msg.type = MessageType.FRIEND_MESSAGE if is_private else MessageType.GROUP_MESSAGE
        if not is_private:
            msg.group_id = event.get_group_id()
        
        # 设置发送者信息
        msg.sender = MessageMember(user_id=event.get_self_id(), nickname="AstrBot")

        # 生成纯文本消息
        msg.message_str = ""
        for comp in chain:
            if isinstance(comp, Plain):
                msg.message_str += comp.text
        
        # 设置其他必要字段
        msg.self_id = event.message_obj.self_id if hasattr(event.message_obj, "self_id") else "bot"
        msg.session_id = event.session_id
        msg.message_id = f"bot_{int(time.time())}"  # 创建一个唯一的消息ID
        
        return msg
    
    @staticmethod
    async def save_bot_message_from_chain(chain: List[BaseMessageComponent], event: AstrMessageEvent) -> bool:
        """
        从消息链和事件对象创建并保存机器人消息
        
        Args:
            chain: 消息链
            event: 触发消息的事件
            
        Returns:
            是否保存成功
        """
        try:
            # 检查是否启用
            is_enabled = HistoryStorage.is_chat_enabled(event)
            if not is_enabled:
                return False
                
            # 创建机器人消息对象
            bot_msg = HistoryStorage.create_bot_message(chain, event)

            # 保存消息
            is_private = event.is_private_chat()
            chat_id = event.get_sender_id() if is_private else event.get_group_id()
            return await HistoryStorage.save_message(bot_msg, event.get_platform_name(), chat_id_override=chat_id)
        except Exception as e:
            logger.error(f"保存机器人消息失败: {e}")
            return False
    
    @staticmethod
    def get_history(platform_name: str, is_private_chat: bool, chat_id: str) -> List[AstrBotMessage]:
        """
        获取历史消息记录
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            历史消息列表
        """
        try:
            file_path = HistoryStorage._get_storage_path(platform_name, is_private_chat, chat_id)
            
            if not os.path.exists(file_path):
                return []
            
            with open(file_path, "r", encoding="utf-8") as f:
                # 使用jsonpickle反序列化JSON数据
                history = jsonpickle.decode(f.read())
                
            return history
        except Exception as e:
            logger.error(f"读取消息历史记录失败: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    @staticmethod
    def clear_history(platform_name: str, is_private_chat: bool, chat_id: str) -> bool:
        """
        清空历史消息记录
        
        Args:
            platform_name: 平台名称
            is_private_chat: 是否为私聊
            chat_id: 聊天ID
            
        Returns:
            是否清空成功
        """
        try:
            file_path = HistoryStorage._get_storage_path(platform_name, is_private_chat, chat_id)
            
            if os.path.exists(file_path):
                os.remove(file_path)
                
            return True
        except Exception as e:
            logger.error(f"清空消息历史记录失败: {e}")
            return False

    @staticmethod
    async def _process_image_persistence(message: AstrBotMessage) -> None:
        """
        处理消息中的图片持久化存储

        将图片保存为文件并在 file 字段中存储相对路径

        Args:
            message: AstrBot消息对象
        """
        try:
            # 检查是否启用图片持久化存储
            if not HistoryStorage.config:
                logger.debug("配置未初始化，跳过图片持久化处理")
                return

            image_processing_config = HistoryStorage.config.get("image_processing", {})
            if not image_processing_config.get("enable_image_persistence", True):
                logger.debug("图片持久化存储已禁用，跳过处理")
                return

            if not hasattr(message, 'message') or not message.message:
                return

            # 确保图片存储目录存在（使用 AstrBot 的数据路径，兼容 Docker）
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            astrbot_data_path = get_astrbot_data_path()
            images_dir = os.path.join(astrbot_data_path, "chat_history", "images")
            HistoryStorage._ensure_dir(images_dir)

            for component in message.message:
                if isinstance(component, Image):
                    # 检查是否已经是持久化路径（file:/// 开头且指向 images 目录）
                    if component.file and component.file.startswith("file:///") and "/images/" in component.file:
                        logger.debug("图片已经是持久化路径，跳过处理")
                        continue

                    # 尝试将图片保存为持久化文件
                    try:
                        # 获取图片的本地文件路径
                        temp_file_path = await component.convert_to_file_path()
                        logger.debug(f"获取的绝对路径:{temp_file_path}")

                        if temp_file_path and os.path.exists(temp_file_path):
                            # 生成唯一的文件名
                            import uuid
                            unique_id = uuid.uuid4().hex
                            file_extension = ".jpg"  # 默认使用 jpg 扩展名

                            # 尝试从原文件获取扩展名
                            if "." in temp_file_path:
                                original_ext = os.path.splitext(temp_file_path)[1].lower()
                                if original_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
                                    file_extension = original_ext

                            # 构建持久化文件路径
                            persistent_filename = f"{unique_id}{file_extension}"
                            persistent_file_path = os.path.join(images_dir, persistent_filename)

                            # 复制文件到持久化目录
                            import shutil
                            shutil.copy2(temp_file_path, persistent_file_path)

                            # 规范化路径（兼容 Docker 环境）
                            persistent_file_path = os.path.abspath(persistent_file_path)

                            # 使用正斜杠规范化路径（兼容 Windows 和 Unix）
                            normalized_path = persistent_file_path.replace('\\', '/')

                            # 存储绝对路径到 file 字段（使用 file:/// 前缀，兼容 AstrBot）
                            component.file = f"file:///{normalized_path}"

                            logger.debug(f"成功将图片保存为持久化文件: {persistent_file_path}")
                        else:
                            logger.warning("无法获取图片的本地文件路径")
                    except Exception as e:
                        logger.error(f"保存图片为持久化文件时发生错误: {e}")
                        # 转换失败时保持原有数据不变
                        continue

        except Exception as e:
            logger.error(f"处理图片持久化存储时发生错误: {e}")
            logger.debug(traceback.format_exc())

    @staticmethod
    def _cleanup_old_images() -> None:
        """
        清理超过配置天数的图片文件

        防止图片文件无限增长
        """
        try:
            # 检查是否启用图片持久化存储
            if not HistoryStorage.config:
                logger.debug("配置未初始化，跳过图片清理")
                return

            image_processing_config = HistoryStorage.config.get("image_processing", {})
            if not image_processing_config.get("enable_image_persistence", True):
                logger.debug("图片持久化存储已禁用，跳过清理")
                return

            # 获取配置的保留天数
            retention_days = image_processing_config.get("image_retention_days", 7)
            if retention_days < 1 or retention_days > 365:
                logger.warning(f"图片保留天数配置无效: {retention_days}，使用默认值7天")
                retention_days = 7

            # 使用 AstrBot 的数据路径（兼容 Docker）
            from astrbot.core.utils.astrbot_path import get_astrbot_data_path
            astrbot_data_path = get_astrbot_data_path()
            images_dir = os.path.join(astrbot_data_path, "chat_history", "images")
            if not os.path.exists(images_dir):
                return

            current_time = time.time()
            cleanup_threshold = retention_days * 24 * 3600  # 配置的天数转换为秒
            cleaned_count = 0

            for filename in os.listdir(images_dir):
                file_path = os.path.join(images_dir, filename)
                if os.path.isfile(file_path):
                    # 检查文件创建时间
                    file_ctime = os.path.getctime(file_path)
                    if current_time - file_ctime > cleanup_threshold:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                            logger.debug(f"清理过期图片文件: {filename}")
                        except Exception as e:
                            logger.error(f"删除过期图片文件失败 {filename}: {e}")

            if cleaned_count > 0:
                logger.info(f"图片清理完成，清理了 {cleaned_count} 个超过 {retention_days} 天的图片文件")

        except Exception as e:
            logger.error(f"清理图片文件时发生错误: {e}")


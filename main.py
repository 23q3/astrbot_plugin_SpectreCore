from astrbot.api.all import *
from astrbot.api.event import filter
from .utils import *
import time

@register(
    "spectrecore",
    "23q3",
    "使大模型更好的主动回复群聊中的消息，带来生动和沉浸的群聊对话体验",
    "2.1.10",
    "https://github.com/23q3/astrbot_plugin_SpectreCore"
)
class SpectreCore(Star):
    """
    使大模型更好的主动回复群聊中的消息，带来生动和沉浸的群聊对话体验
    """
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.config = config
        # 初始化各个工具类
        HistoryStorage.init(config)
        ImageCaptionUtils.init(context, config)

    @event_message_type(EventMessageType.GROUP_MESSAGE)
    async def on_group_message(self, event: AstrMessageEvent):
        """处理群消息喵"""
        try:
            # 保存用户消息到历史记录并尝试回复
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理群消息时发生错误: {e}")

    @event_message_type(EventMessageType.PRIVATE_MESSAGE)
    async def on_private_message(self, event: AstrMessageEvent):
        """处理私聊消息喵"""
        try:
            # 保存用户消息到历史记录并尝试回复
            async for result in self._process_message(event):
                yield result
        except Exception as e:
            logger.error(f"处理私聊消息时发生错误: {e}")
            
    async def _process_message(self, event: AstrMessageEvent):
        """处理消息的通用逻辑：保存历史记录并尝试回复"""
        # 过滤空消息(napcat会发送私聊对方正在输入的状态，导致astrbot识别为空消息)
        message_outline = event.get_message_outline()
        if not message_outline or message_outline.strip() == "":
            logger.debug("收到空消息，忽略处理")
            return

        # 保存用户消息到历史记录
        await HistoryStorage.process_and_save_user_message(event)

        # 尝试自动回复
        if ReplyDecision.should_reply(event, self.config):
            async for result in ReplyDecision.process_and_reply(event, self.config, self.context):
                yield result

    @filter.after_message_sent()
    async def after_message_sent(self, event: AstrMessageEvent):
        """处理bot发送的消息喵"""
        try:           
            # 保存机器人消息
            if event._result and hasattr(event._result, "chain"):
                # 检查是否为重置历史记录的提示消息，如果是则不保存
                message_text = "".join([i.text for i in event._result.chain if hasattr(i, "text")])
                if "已成功重置" in message_text and "的历史记录喵~" in message_text:
                    return
                
                await HistoryStorage.save_bot_message_from_chain(event._result.chain, event)
                logger.debug(f"已保存bot回复消息到历史记录")
                
        except Exception as e:
            logger.error(f"处理bot发送的消息时发生错误: {e}")

    from astrbot.api.provider import LLMResponse
    @filter.on_llm_response(priority=114514)
    async def on_llm_resp(self, event: AstrMessageEvent, resp: LLMResponse):
        """处理大模型回复喵"""
        logger.debug(f"收到大模型回复喵: {resp}")
        try:
            if resp.role != "assistant":
                return
            # 只进行文本过滤，不处理读空气逻辑
            resp.completion_text = TextFilter.process_model_text(resp.completion_text, self.config)
        except Exception as e:
            logger.error(f"处理大模型回复时发生错误: {e}")

    @filter.on_decorating_result()
    async def on_decorating_result(self, event: AstrMessageEvent):
        """在消息发送前处理读空气功能喵"""
        try:
            result = event.get_result()
            if result is None or not result.chain:
                return

            # 检查是否为LLM结果且包含<NO_RESPONSE>标记
            if result.is_llm_result():
                # 获取消息文本内容
                message_text = ""
                for comp in result.chain:
                    if hasattr(comp, 'text'):
                        message_text += comp.text

                # 如果包含<NO_RESPONSE>标记，清空事件结果以阻止消息发送
                if "<NO_RESPONSE>" in message_text:
                    logger.debug(f"检测到读空气标记，阻止消息发送。事件结果: {event.get_result()}")
                    event.clear_result()
                    logger.debug(f"已清空事件结果: {event.get_result()}")

        except Exception as e:
            logger.error(f"处理消息发送前事件时发生错误: {e}")

    @filter.command_group("spectrecore",alias={'sc'})
    def spectrecore(self):
        """插件的前缀喵 可以用sc代替喵"""
        pass

    @spectrecore.command("help", alias=['帮助', 'helpme'])
    async def help(self, event: AstrMessageEvent):
        """查看插件的帮助喵"""
        help_text = (
            "SpectreCore插件帮助文档\n"
            "使用spectrecore或sc作为指令前缀 如/sc help\n"
            "使用reset指令重置当前聊天记录 如/sc reset\n"
            "   你也可以重置指定群聊天记录 如/sc reset 群号\n"
            "使用history指令可以查看最近聊天记录 如/sc history\n"
            "使用mute/闭嘴指令临时禁用自动回复 如/sc mute 5 或 /sc 闭嘴 10\n"
            "使用unmute/说话指令解除禁用 如/sc unmute 或 /sc 说话"
        )
        platform_name = event.get_platform_name()
        if platform_name in ("qq_official", "qq_official_webhook"):
            help_text += "\n强烈建议前往Github阅读README文档"
        else:
            help_text += "\n↓强烈建议您阅读Github中的README文档↓\nhttps://github.com/23q3/astrbot_plugin_SpectreCore"
        yield event.plain_result(help_text)
    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("history")
    async def history(self, event: AstrMessageEvent, count: int = 10):
        """查看最近的聊天记录喵，默认10条喵，示例/sc history 5"""
        try:
            # 获取平台名称
            platform_name = event.get_platform_name()
            
            # 判断是群聊还是私聊
            is_private = event.is_private_chat()
            
            # 获取聊天ID
            chat_id = event.get_group_id() if not is_private else event.get_sender_id()
            
            if not chat_id:
                yield event.plain_result("获取聊天ID失败喵，无法显示历史记录")
                return
                
            # 获取历史记录
            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            
            if not history:
                yield event.plain_result("暂无聊天记录喵")
                return
                
            # 限制记录数量
            if count > 20:
                count = 20  # 限制最大显示数量为20条
            
            # 只取最近的记录
            recent_history = history[-count:] if len(history) > count else history
            
            # 格式化历史记录
            formatted_history = await MessageUtils.format_history_for_llm(recent_history, umo=event.unified_msg_origin)
            
            # 添加标题
            chat_type = "私聊" if is_private else f"群聊({chat_id})"
            title = f"最近{len(recent_history)}条{chat_type}聊天记录喵：\n\n"
            
            # 发送结果
            full_content = title + formatted_history
            
            # 如果内容过长，转为图片发送
            if len(full_content) > 3000:
                image_url = await self.text_to_image(full_content)
                yield event.image_result(image_url)
            else:
                yield event.plain_result(full_content)
            
        except Exception as e:
            logger.error(f"获取历史记录时发生错误: {e}")
            yield event.plain_result(f"获取历史记录失败喵：{str(e)}")
    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("reset")
    async def reset(self, event: AstrMessageEvent, group_id: str | None = None):
        """重置历史记录喵，不带参数重置当前聊天记录，带群号则重置指定群聊记录 如/sc reset 123456"""
        try:
            # 获取平台名称
            platform_name = event.get_platform_name()
            
            # 判断是否提供了群号
            if group_id:
                # 重置指定群聊的历史记录
                is_private = False
                chat_id = group_id
                chat_type = f"群聊({group_id})"
            else:
                # 判断是群聊还是私聊
                is_private = event.is_private_chat()
                # 获取聊天ID
                chat_id = event.get_group_id() if not is_private else event.get_sender_id()
                chat_type = "私聊" if is_private else f"群聊({chat_id})"
                
                if not chat_id:
                    yield event.plain_result("获取聊天ID失败喵，无法重置历史记录")
                    return
            
            # 先检查是否存在历史记录
            history = HistoryStorage.get_history(platform_name, is_private, chat_id)
            if not history:
                yield event.plain_result(f"{chat_type}没有历史记录喵，无需重置")
                return
                
            # 重置历史记录
            success = HistoryStorage.clear_history(platform_name, is_private, chat_id)
            
            if success:
                yield event.plain_result(f"已成功重置{chat_type}的历史记录喵~")
            else:
                yield event.plain_result(f"重置{chat_type}的历史记录失败喵，可能发生错误")
                
        except Exception as e:
            logger.error(f"重置历史记录时发生错误: {e}")
            yield event.plain_result(f"重置历史记录失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("mute", alias=['闭嘴', 'shutup'])
    async def mute(self, event: AstrMessageEvent, minutes: int = 5):
        """临时禁用自动回复，默认5分钟喵，示例/sc mute 10 或 /sc 闭嘴 3"""
        try:
            # 计算禁用结束时间
            mute_until = time.time() + (minutes * 60)
            
            # 保存到配置中
            if "_temp_mute" not in self.config:
                self.config["_temp_mute"] = {}
            
            self.config["_temp_mute"]["until"] = mute_until
            self.config["_temp_mute"]["by"] = event.get_sender_id()
            self.config["_temp_mute"]["at"] = time.time()
            
            # 保存配置
            self.config.save_config()
            
            yield event.plain_result(f"好的喵，我会安静 {minutes} 分钟的~")
            
        except Exception as e:
            logger.error(f"执行闭嘴指令时发生错误: {e}")
            yield event.plain_result(f"执行闭嘴指令失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("unmute", alias=['说话', 'speak'])
    async def unmute(self, event: AstrMessageEvent):
        """解除禁用自动回复喵"""
        try:
            # 检查是否处于静默状态
            mute_info = self.config.get("_temp_mute", {})
            if not mute_info or mute_info.get("until", 0) <= time.time():
                yield event.plain_result("我现在本来就在正常说话喵~")
                return
            
            # 解除静默
            if "_temp_mute" in self.config:
                del self.config["_temp_mute"]
                self.config.save_config()
            
            yield event.plain_result("好耶！我又可以说话了喵~")
            
        except Exception as e:
            logger.error(f"解除闭嘴时发生错误: {e}")
            yield event.plain_result(f"解除闭嘴失败喵：{str(e)}")

    @filter.permission_type(filter.PermissionType.ADMIN)
    @spectrecore.command("callllm")
    async def callllm(self, event: AstrMessageEvent):
        """触发一次大模型回复 这是用来开发中测试的喵"""
        try:
            # 调用LLM工具类的方法构建并返回请求
            yield await LLMUtils.call_llm(event, self.config, self.context)
        except Exception as e:
            logger.error(f"调用大模型时发生错误: {e}")
            yield event.plain_result(f"触发大模型回复失败喵：{str(e)}")
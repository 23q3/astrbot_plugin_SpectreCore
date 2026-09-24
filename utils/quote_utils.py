from astrbot.api.all import *
from typing import Any, Dict, Optional
import re


class QuoteUtils:
    """
    模型自主引用工具类

    聊天记录中的消息和当前消息会带上编号，模型可以在回复开头用 [引用:编号] 指定要引用的消息，
    由插件插入对应的引用。根据 AstrBot 的「回复时引用发送人消息」分为两种模式：
    - 关闭时为自主引用：模型自行决定是否引用、引用哪条，不写标记就不引用
    - 开启时为必须引用：模型每次都要指定引用哪条，漏写时由 AstrBot 引用触发回复的消息
    """

    MODE_OPTIONAL = "optional"
    MODE_REQUIRED = "required"

    # 存放在 event extra 中的可引用消息表：{编号: {id, sender_id, sender_name, content}}
    EXTRA_KEY = "spectrecore_quote_targets"
    # 匹配所有 [引用:xxx] 形式的标记（包括写错的），保证标记不会被发送出去
    MARKER_PATTERN = re.compile(r"[\[【]\s*引用\s*[:：]([^\]】\n]*)[\]】]")
    # AstrBot 4.19.3 起，消息链中已有引用时不再自动插入引用和 @；更早的版本会重复插入
    MIN_ASTRBOT_VERSION = (4, 19, 3)

    _version_supported: Optional[bool] = None

    @staticmethod
    def _is_astrbot_supported() -> bool:
        """检查当前 AstrBot 版本是否支持由插件插入引用"""
        if QuoteUtils._version_supported is None:
            try:
                from astrbot.core.config.default import VERSION
                version = tuple(int(x) for x in re.findall(r"\d+", VERSION)[:3])
                QuoteUtils._version_supported = version >= QuoteUtils.MIN_ASTRBOT_VERSION
            except Exception as e:
                logger.warning(f"获取 AstrBot 版本失败: {e}")
                QuoteUtils._version_supported = False
            if not QuoteUtils._version_supported:
                logger.info("模型自主引用需要 AstrBot v4.19.3 及以上，当前版本不启用")
        return QuoteUtils._version_supported

    @staticmethod
    def _get_astrbot_config(context: Context, umo: str) -> dict:
        """获取当前会话对应的 AstrBot 配置"""
        try:
            return context.get_config(umo=umo)
        except TypeError:
            return context.get_config()

    @staticmethod
    def _is_streaming(event: AstrMessageEvent, astrbot_config: dict) -> bool:
        """流式输出不经过 AstrBot 的结果装饰阶段，插件无法移除标记和插入引用"""
        settings = astrbot_config.get("provider_settings", {}) or {}
        streaming = event.get_extra("enable_streaming")
        if streaming is None:
            streaming = settings.get("streaming_response", False)
        if not streaming:
            return False
        # 平台不支持流式且策略为关闭流式时，AstrBot 会按普通消息发送
        return not (
            settings.get("unsupported_streaming_strategy") == "turn_off"
            and not getattr(event.platform_meta, "support_streaming_message", False)
        )

    @staticmethod
    def get_mode(event: AstrMessageEvent, config: AstrBotConfig, context: Context) -> Optional[str]:
        """
        获取本次回复的引用模式，不启用时返回 None
        """
        if not config.get("smart_quote", True):
            return None
        try:
            astrbot_config = QuoteUtils._get_astrbot_config(context, event.unified_msg_origin)
            if QuoteUtils._is_streaming(event, astrbot_config):
                return None
            reply_with_quote = astrbot_config.get("platform_settings", {}).get("reply_with_quote", False)
        except Exception as e:
            logger.debug(f"读取 AstrBot 配置失败: {e}")
            return None
        if not QuoteUtils._is_astrbot_supported():
            return None
        return QuoteUtils.MODE_REQUIRED if reply_with_quote else QuoteUtils.MODE_OPTIONAL

    @staticmethod
    def build_instruction(mode: str, current_no: Optional[str]) -> str:
        """构建告诉模型如何引用消息的提示词"""
        current_hint = f"这条新消息的编号是 {current_no}，" if current_no else ""
        if mode == QuoteUtils.MODE_REQUIRED:
            return (
                f"\n(你的回复会引用一条消息。请在回复的最开头写上 [引用:编号] 标明你回应的是哪条消息，"
                f"{current_hint}聊天记录中其他消息的编号写在各条消息前)"
            )
        return (
            f"\n(如果你想像在聊天软件里那样引用某条消息来回复，可以在回复的最开头写上 [引用:编号]，"
            f"{current_hint}聊天记录中其他消息的编号写在各条消息前；不需要引用时不要写)"
        )

    @staticmethod
    def make_target(message: AstrBotMessage, content: str) -> Optional[Dict[str, Any]]:
        """
        从消息构建可引用目标

        bot 自己的消息只有插件生成的 ID（bot_ 开头），无法被引用，返回 None
        """
        message_id = str(getattr(message, "message_id", "") or "")
        if not message_id or message_id.startswith("bot_"):
            return None
        sender = getattr(message, "sender", None)
        return {
            "id": message_id,
            "sender_id": str(getattr(sender, "user_id", "") or ""),
            "sender_name": getattr(sender, "nickname", "") or "",
            "content": content,
        }

    @staticmethod
    def apply(event: AstrMessageEvent, result: MessageEventResult, context: Context) -> None:
        """
        处理模型回复中的 [引用:编号] 标记：移除标记，并插入对应消息的引用

        需在 on_decorating_result 中调用，此时 AstrBot 还没有插入自己的引用。
        没有有效标记时不做处理：自主引用模式下不引用，必须引用模式下由 AstrBot 引用触发回复的消息
        """
        targets = event.get_extra(QuoteUtils.EXTRA_KEY)
        if not targets:
            return

        chosen = None
        marker_found = False
        for comp in result.chain:
            if not isinstance(comp, Plain) or not comp.text:
                continue
            matches = list(QuoteUtils.MARKER_PATTERN.finditer(comp.text))
            if not matches:
                continue
            marker_found = True
            for match in matches:
                number = re.search(r"\d+", match.group(1))
                if chosen is None and number:
                    chosen = number.group()
            comp.text = QuoteUtils.MARKER_PATTERN.sub("", comp.text).strip()

        if not marker_found:
            return

        # 移除只剩标记的文本段，如果什么都不剩就不发送
        result.chain = [c for c in result.chain if not (isinstance(c, Plain) and not c.text)]
        if not result.chain:
            event.clear_result()
            return

        target = targets.get(chosen)
        if not target:
            logger.debug(f"模型引用的编号 {chosen} 无效，不插入引用")
            return

        astrbot_config = QuoteUtils._get_astrbot_config(context, event.unified_msg_origin)
        settings = astrbot_config.get("platform_settings", {}) or {}

        # 长消息会被 AstrBot 转为合并转发，此时 AstrBot 也不会引用，保持一致
        if event.get_platform_name() == "aiocqhttp":
            word_cnt = sum(len(c.text) for c in result.chain if isinstance(c, Plain))
            if word_cnt > settings.get("forward_threshold", 1500):
                return

        # 插入引用后 AstrBot 会跳过自己的引用和 @，这里按相同规则 @ 被引用消息的发送者
        header = [Reply(
            id=target["id"],
            sender_id=target["sender_id"],
            sender_nickname=target["sender_name"],
            message_str=target["content"],
        )]
        if (
            settings.get("reply_with_mention", False)
            and event.get_message_type() != MessageType.FRIEND_MESSAGE
            and target["sender_id"]
        ):
            header.append(At(qq=target["sender_id"], name=target["sender_name"]))
            if isinstance(result.chain[0], Plain):
                result.chain[0].text = "\n" + result.chain[0].text
        result.chain[0:0] = header
        logger.debug(f"模型选择引用编号 {chosen} 的消息: {target['id']}")

from __future__ import annotations

from .models import (
    ChatMessageModel,
    ChatState,
    CitationModel,
    ConversationModel,
    ImageArtifactModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
    SummaryArtifactModel,
    TableArtifactModel,
    VideoArtifactModel,
)


def build_assistant_seed_message() -> ChatMessageModel:
    return ChatMessageModel(
        id="msg-a1",
        role="assistant",
        time="2026-07-09 10:32:00",
        paragraphs=[
            ResponseParagraphModel(
                text="2024年Q4公司整体经营稳健，收入与利润保持增长，现金流显著改善，主要受核心业务增长、成本结构优化与回款提速三方面驱动。",
                citations=[
                    CitationModel(
                        label="来源 ① 财务摘要_Q4",
                        classification="内部·机密",
                        source_id="ARC-FIN-Q4",
                    )
                ],
            ),
            ResponseParagraphModel(
                text="收入方面，Q4实现营业收入128.7亿元，同比增长18.6%，较Q3环比增长7.3%。其中核心产品线收入同比增长21.4%，新兴业务收入占比提升至34%，成为重要增长引擎。",
                citations=[
                    CitationModel(
                        label="来源 ② 管理层简报_Q4",
                        classification="内部·机密",
                        source_id="ARC-MGT-Q4",
                    )
                ],
            ),
            ResponseParagraphModel(
                text="利润方面，Q4实现归母净利润18.9亿元，同比增长23.8%，较Q3环比增长9.7%，毛利率提升至32.6%，主要得益于产品结构优化与规模效应释放。",
                citations=[
                    CitationModel(
                        label="来源 ③ 利润分析底稿_Q4",
                        classification="财务受限",
                        source_id="ARC-PNL-Q4",
                    )
                ],
            ),
            ResponseParagraphModel(
                text="下一步建议：继续强化高毛利产品占比，优化产品组合与定价策略；加大新兴市场投入，提升长期增长动能；关注现金流结构，保持充足流动性以应对外部不确定性。",
                citations=[
                    CitationModel(
                        label="来源 ④ 经营例会纪要_Q4",
                        classification="内部·机密",
                        source_id="ARC-ACTION-Q4",
                    )
                ],
            ),
        ],
        artifacts=[
            SummaryArtifactModel(
                type="summary",
                title="摘要",
                source="来源 ⑤ 经营简报_Q4",
                bullets=[
                    "收入：128.7亿元，同比增长 +18.6%，环比 +7.3%",
                    "利润：18.9亿元，同比增长 +23.8%，环比 +9.7%",
                    "现金流：24.3亿元，环比 +41.2%",
                    "关键驱动：核心业务增长、成本优化、回款提速",
                ],
            ),
            ImageArtifactModel(
                type="image",
                title="2024年Q4经营概览",
                source="来源 ⑥ 经营图册_Q4",
                asset_key="city",
            ),
            VideoArtifactModel(
                type="video",
                title="Q4经营分析解读视频",
                source="来源 ⑦ 管理层解读_Q4",
                duration="03:42",
                asset_key="analysis",
            ),
            TableArtifactModel(
                type="table",
                title="Q4 vs Q3 关键指标对比",
                source="来源 ⑧ 资金流量表_Q4",
                columns=["指标", "Q4 2024", "Q3 2024", "环比变化", "同比变化"],
                rows=[
                    ["营业收入", "128.7", "120.0", "+7.3%", "+18.6%"],
                    ["归母净利润", "18.9", "17.2", "+9.7%", "+23.8%"],
                    ["毛利率", "32.6%", "30.5%", "+2.1pp", "+3.4pp"],
                    ["经营现金流", "24.3", "17.2", "+41.2%", "+26.5%"],
                ],
            ),
        ],
    )


def build_seed_state() -> ChatState:
    conversations = [
        ConversationModel(
            id="conv-q4",
            title="2024年Q4经营分析与趋势洞察",
            topic="经营分析",
            group="今天",
            updated_at="2026-07-09 10:32:00",
            pinned=True,
            context_summary="用户正在分析Q4经营情况，重点关注收入、利润、现金流及下一步行动建议。",
            turn_count=1,
        ),
        ConversationModel(
            id="conv-overseas",
            title="专题：海外市场进入策略",
            topic="战略专题",
            group="今天",
            updated_at="2026-07-09 09:15:00",
        ),
        ConversationModel(
            id="conv-portfolio",
            title="产品组合优化建议",
            topic="产品经营",
            group="昨天",
            updated_at="2026-07-08 18:20:00",
        ),
        ConversationModel(
            id="conv-competition",
            title="行业动态与竞争情报周报",
            topic="情报周报",
            group="昨天",
            updated_at="2026-07-08 16:45:00",
        ),
        ConversationModel(
            id="conv-efficiency",
            title="组织效率提升方案讨论",
            topic="组织管理",
            group="本周",
            updated_at="2026-07-08 14:30:00",
        ),
        ConversationModel(
            id="conv-budget",
            title="预算执行情况分析",
            topic="财务分析",
            group="本周",
            updated_at="2026-07-07 11:20:00",
        ),
        ConversationModel(
            id="conv-risk",
            title="供应链风险评估",
            topic="风险研判",
            group="更早",
            updated_at="2026-05-12 14:00:00",
        ),
    ]

    messages_by_conversation = {
        "conv-q4": [
            ChatMessageModel(
                id="msg-u1",
                role="user",
                time="2026-07-09 10:31:00",
                content="请帮我分析2024年Q4的经营情况，并与Q3进行对比，重点说明收入、利润、现金流的变化及原因，并给出下一步建议。",
            ),
            build_assistant_seed_message(),
        ],
        "conv-overseas": [
            ChatMessageModel(
                id="msg-overseas-u1",
                role="user",
                time="2026-07-09 09:15:00",
                content="东南亚市场进入策略需要先看哪些风险？",
            ),
            ChatMessageModel(
                id="msg-overseas-a1",
                role="assistant",
                time="2026-07-09 09:16:00",
                paragraphs=[
                    ResponseParagraphModel(
                        text="建议先拆成准入、渠道、合规、价格带四个维度，并优先核对本地监管限制与核心渠道的账期风险。",
                        citations=[
                            CitationModel(
                                label="来源 ① 海外市场备忘录",
                                classification="内部·机密",
                                source_id="ARC-SEA-014",
                            )
                        ],
                    )
                ],
            ),
        ],
    }

    knowledge_sources = [
        KnowledgeSourceModel(
            id="kb-001",
            name="2024_Q4_财务摘要.pdf",
            source_type="PDF",
            records=128,
            status="已索引",
            updated_at="2026-07-09 09:42:00",
            classification="财务受限",
        ),
        KnowledgeSourceModel(
            id="kb-002",
            name="管理层经营简报_Q4.xlsx",
            source_type="表格",
            records=64,
            status="已索引",
            updated_at="2026-07-09 08:25:00",
            classification="内部·机密",
        ),
        KnowledgeSourceModel(
            id="kb-003",
            name="供应链账期访谈纪要.docx",
            source_type="文档",
            records=31,
            status="待复核",
            updated_at="2026-07-08 18:10:00",
            classification="内部",
        ),
    ]

    return ChatState(
        conversations=conversations,
        messages_by_conversation=messages_by_conversation,
        knowledge_sources=knowledge_sources,
    )
